# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import json
import os
from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.actors.utils import (
    compute_logprobs,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.envs import EnvExample, EnvStep
from torchtitan.experiments.rl.envs.token_env import (
    PromptState,
    TokenEnv,
    TokenEnvConfig,
    TokenStep,
)
from torchtitan.experiments.rl.generation_scheduler import GenerationScheduler
from torchtitan.experiments.rl.grpo import (
    _build_train_step_trace_scalars,
    _raise_rollout_task_errors,
    _RolloutDropCounters,
    Provisioner,
    RLTrainer,
)
from torchtitan.experiments.rl.loss import DAPOLoss, GRPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    _TrainStepTimings,
    _WeightSyncTimings,
    build_rollout_metrics,
    build_train_step_metrics,
    rename_metric_prefix,
)
from torchtitan.experiments.rl.replay import (
    has_advantage_signal,
    ReplayBatch,
    ReplayBuffer,
    ReplayBufferStats,
    ReplayGroup,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.rollout_logging import RolloutSampleLogger
from torchtitan.experiments.rl.rollouts import do_single_rollout, run_rollout_group
from torchtitan.experiments.rl.sampling import SamplingConfig, TrainingLogprobConfig
from torchtitan.experiments.rl.sum_digits import (
    SumDigitsBuilder,
    SumDigitsDataset,
    SumDigitsExample,
)

from torchtitan.experiments.rl.trainer_microbatch import (
    has_loss_tokens,
    MetricAccumulator,
    schedule_training_microbatches,
    split_training_batch,
    split_training_batches_by_rank,
    zero_gradient_training_batch_like,
)
from torchtitan.experiments.rl.types import (
    Completion,
    OptimStepOutput,
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
    TrainingBatch,
)


def test_sampling_config_default_is_one_completion():
    assert SamplingConfig().n == 1
    assert SamplingConfig().top_p == 1.0


def test_training_logprob_config_validates_sampling_contract():
    cfg = TrainingLogprobConfig.from_sampling(
        SamplingConfig(temperature=0.7, top_p=1.0)
    )

    assert cfg.temperature == 0.7

    with pytest.raises(ValueError, match="top_p=1.0"):
        TrainingLogprobConfig.from_sampling(SamplingConfig(top_p=0.95))
    with pytest.raises(ValueError, match="temperature must be positive"):
        TrainingLogprobConfig.from_sampling(SamplingConfig(temperature=0.0))


def test_compute_logprobs_applies_sampling_temperature():
    logits = torch.tensor([[[0.0, 2.0], [3.0, 1.0], [0.5, 0.5]]])
    token_ids = torch.tensor([[0, 1, 0]])

    actual = compute_logprobs(logits, token_ids, temperature=2.0)
    expected = torch.log_softmax(logits[:, :-1, :].float() / 2.0, dim=-1)
    expected = expected.gather(2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(actual, expected)


def test_compute_logprobs_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature must be positive"):
        compute_logprobs(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.long),
            temperature=0.0,
        )


def test_grpo_loss_clamps_log_ratio_before_exp():
    loss_fn = GRPOLoss(GRPOLoss.Config(max_log_ratio=5.0))

    loss, metrics = loss_fn(
        policy_logprobs=torch.tensor([1000.0, -1000.0]),
        behavior_logprobs=torch.tensor([0.0, 0.0]),
        advantages=torch.tensor([-1.0, 1.0]),
        num_global_valid_tokens=torch.tensor(2.0),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["loss/ratio/mean"])
    assert metrics["loss/ratio/log_clipped_frac"].item() == 1.0


def test_grpo_loss_sanitizes_nonfinite_log_ratios():
    loss_fn = GRPOLoss(GRPOLoss.Config(max_log_ratio=5.0))

    loss, metrics = loss_fn(
        policy_logprobs=torch.tensor([float("inf"), float("-inf"), float("-inf")]),
        behavior_logprobs=torch.tensor([0.0, 0.0, float("-inf")]),
        advantages=torch.tensor([0.5, -0.25, 0.75]),
        num_global_valid_tokens=torch.tensor(3.0),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["loss/ratio/mean"])
    assert metrics["loss/ratio/nonfinite_frac"].item() == 1.0
    assert metrics["loss/logprob/policy_nonfinite_frac"].item() == 1.0
    assert metrics["loss/logprob/behavior_nonfinite_frac"].item() == pytest.approx(
        1 / 3
    )


def test_dapo_loss_uses_asymmetric_clip_high():
    policy_logprobs = torch.log(torch.tensor([1.25, 0.7]))
    behavior_logprobs = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])
    num_global_valid_tokens = torch.tensor(2.0)

    symmetric_loss, symmetric_metrics = GRPOLoss(GRPOLoss.Config(clip_eps=0.2))(
        policy_logprobs=policy_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        num_global_valid_tokens=num_global_valid_tokens,
    )
    dapo_loss, dapo_metrics = DAPOLoss(DAPOLoss.Config(clip_low=0.2, clip_high=0.3))(
        policy_logprobs=policy_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        num_global_valid_tokens=num_global_valid_tokens,
    )

    assert symmetric_loss.item() == pytest.approx(-0.2)
    assert dapo_loss.item() == pytest.approx(-0.225)
    assert dapo_loss < symmetric_loss
    assert symmetric_metrics["loss/ratio/clipped_high_frac"].item() == pytest.approx(
        0.5
    )
    assert dapo_metrics["loss/ratio/clipped_high_frac"].item() == 0.0
    assert dapo_metrics["loss/ratio/clipped_low_frac"].item() == pytest.approx(0.5)


def test_dapo_loss_clips_only_active_advantage_directions():
    ratios = torch.tensor([1.30, 1.25, 0.70, 0.75, 1.30, 0.70])
    advantages = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
    loss_fn = DAPOLoss(DAPOLoss.Config(clip_low=0.2, clip_high=0.28))

    loss, metrics = loss_fn(
        policy_logprobs=torch.log(ratios),
        behavior_logprobs=torch.zeros_like(ratios),
        advantages=advantages,
        num_global_valid_tokens=torch.tensor(float(ratios.numel())),
    )

    # Mirrors Forge's pg_ppo_clip semantics: upper clipping applies to positive
    # advantages, lower clipping applies to negative advantages.
    expected_token_losses = torch.tensor([-1.28, -1.25, 0.8, 0.8, 1.3, -0.7])
    assert loss.item() == pytest.approx(expected_token_losses.mean().item())
    assert metrics["loss/ratio/clipped_high_frac"].item() == pytest.approx(1 / 6)
    assert metrics["loss/ratio/clipped_low_frac"].item() == pytest.approx(2 / 6)
    assert metrics["loss/ratio/clipped_frac"].item() == pytest.approx(3 / 6)


def test_dapo_loss_supports_forge_dual_clip():
    ratios = torch.tensor([10.0, 2.0, 0.5])
    advantages = torch.tensor([-1.0, -1.0, 1.0])
    loose_dual_loss, loose_dual_metrics = DAPOLoss(
        DAPOLoss.Config(clip_low=0.2, clip_high=0.28, dual_clip_c=100.0)
    )(
        policy_logprobs=torch.log(ratios),
        behavior_logprobs=torch.zeros_like(ratios),
        advantages=advantages,
        num_global_valid_tokens=torch.tensor(float(ratios.numel())),
    )

    dual_loss, dual_metrics = DAPOLoss(
        DAPOLoss.Config(clip_low=0.2, clip_high=0.28, dual_clip_c=3.0)
    )(
        policy_logprobs=torch.log(ratios),
        behavior_logprobs=torch.zeros_like(ratios),
        advantages=advantages,
        num_global_valid_tokens=torch.tensor(float(ratios.numel())),
    )

    expected_dual_token_losses = torch.tensor([3.0, 2.0, -0.5])
    assert dual_loss.item() == pytest.approx(expected_dual_token_losses.mean().item())
    assert dual_loss < loose_dual_loss
    assert loose_dual_metrics["loss/dual_clip/clipped_frac"].item() == 0.0
    assert dual_metrics["loss/dual_clip/clipped_frac"].item() == pytest.approx(1 / 3)


def test_dapo_loss_rejects_invalid_dual_clip_constant():
    with pytest.raises(ValueError, match="dual_clip_c must be greater than 1"):
        DAPOLoss(DAPOLoss.Config(dual_clip_c=1.0))


def test_dapo_loss_zero_advantages_are_finite():
    loss_fn = DAPOLoss(DAPOLoss.Config(clip_low=0.2, clip_high=0.28))

    loss, metrics = loss_fn(
        policy_logprobs=torch.tensor([10.0, -10.0]),
        behavior_logprobs=torch.tensor([0.0, 0.0]),
        advantages=torch.zeros(2),
        num_global_valid_tokens=torch.tensor(2.0),
    )

    assert loss.item() == 0.0
    assert torch.isfinite(metrics["loss/ratio/mean"])
    assert metrics["loss/ratio/clipped_frac"].item() == 0.0


def test_dapo_loss_empty_selected_tokens_is_finite_zero():
    loss_fn = DAPOLoss(DAPOLoss.Config(clip_low=0.2, clip_high=0.28))

    loss, metrics = loss_fn(
        policy_logprobs=torch.empty(0),
        behavior_logprobs=torch.empty(0),
        advantages=torch.empty(0),
        num_global_valid_tokens=torch.tensor(1.0),
    )

    assert loss.item() == 0.0
    for value in metrics.values():
        assert torch.isfinite(value)


def test_logprob_drift_reports_nonfinite_without_nan_metrics():
    drift = verify_logprob_identity(
        generator_token_logprobs=torch.tensor([0.0, float("-inf"), 1.0, float("nan")]),
        trainer_token_logprobs=torch.tensor([0.1, 0.0, 2.0, float("nan")]),
        num_global_valid_tokens=torch.tensor(4.0),
        device=torch.device("cpu"),
    )

    assert torch.isfinite(drift.logprob_diff_mean)
    assert torch.isfinite(drift.logprob_diff_max)
    assert torch.isfinite(drift.ratio_tokens_different)
    assert torch.isfinite(drift.nonfinite_logprob_frac)
    assert drift.logprob_diff_mean.item() == pytest.approx(1.1 / 4)
    assert drift.logprob_diff_max.item() == pytest.approx(1.0)
    assert drift.ratio_tokens_different.item() == pytest.approx(2 / 4)
    assert drift.nonfinite_logprob_frac.item() == pytest.approx(2 / 4)


def _required_fwd_bwd_metrics(**overrides):
    metrics = {
        "loss/mean": 0.0,
        "loss/ratio/nonfinite_frac": 0.0,
        "loss/logprob/policy_nonfinite_frac": 0.0,
        "loss/logprob/behavior_nonfinite_frac": 0.0,
        "bit_wise/nonfinite_logprob_frac": 0.0,
    }
    metrics.update(overrides)
    return metrics


def test_forward_backward_skip_metrics_reject_nonfinite_loss_signal():
    assert (
        RLTrainer._forward_backward_skip_metrics(
            _required_fwd_bwd_metrics(),
            policy_version=3,
        )
        is None
    )

    loss_skip = RLTrainer._forward_backward_skip_metrics(
        _required_fwd_bwd_metrics(**{"loss/mean": float("nan")}),
        policy_version=3,
    )
    assert loss_skip == {
        "train/policy_version": 3.0,
        "train/skipped_nonfinite_loss": 1.0,
        "train/skipped_nonfinite_grad_norm": 0.0,
    }

    ratio_skip = RLTrainer._forward_backward_skip_metrics(
        _required_fwd_bwd_metrics(**{"loss/ratio/nonfinite_frac": 0.25}),
        policy_version=3,
    )
    assert ratio_skip == loss_skip

    with pytest.raises(KeyError, match="loss/ratio/nonfinite_frac"):
        RLTrainer._forward_backward_skip_metrics(
            {"loss/mean": 0.0},
            policy_version=3,
        )


def test_grpo_loss_microbatch_sum_matches_full_batch_with_global_denominator():
    loss_fn = GRPOLoss(GRPOLoss.Config())
    policy_logprobs = torch.tensor([-0.1, -0.2, -0.3, -0.4])
    behavior_logprobs = torch.tensor([-0.2, -0.1, -0.35, -0.45])
    advantages = torch.tensor([0.5, -0.25, 0.75, -1.0])
    num_global_valid_tokens = torch.tensor(4.0)

    full_loss, full_metrics = loss_fn(
        policy_logprobs=policy_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        num_global_valid_tokens=num_global_valid_tokens,
    )
    first_loss, first_metrics = loss_fn(
        policy_logprobs=policy_logprobs[:2],
        behavior_logprobs=behavior_logprobs[:2],
        advantages=advantages[:2],
        num_global_valid_tokens=num_global_valid_tokens,
    )
    second_loss, second_metrics = loss_fn(
        policy_logprobs=policy_logprobs[2:],
        behavior_logprobs=behavior_logprobs[2:],
        advantages=advantages[2:],
        num_global_valid_tokens=num_global_valid_tokens,
    )

    assert torch.allclose(first_loss + second_loss, full_loss)
    for key, value in full_metrics.items():
        assert torch.allclose(first_metrics[key] + second_metrics[key], value)


def test_split_training_batch_keeps_sample_boundaries():
    batch = TrainingBatch(
        token_ids=torch.arange(9).view(1, 9),
        seq_lens=[2, 3, 4],
        loss_mask=torch.ones((1, 9), dtype=torch.bool),
        behavior_logprobs=torch.arange(9, dtype=torch.float32).view(1, 9),
        advantages=torch.arange(9, dtype=torch.float32).view(1, 9),
    )

    parts = split_training_batch(
        batch,
        max_samples=2,
        max_tokens=None,
    )

    assert [part.seq_lens for part in parts] == [[2, 3], [4]]
    assert parts[0].token_ids.tolist() == [[0, 1, 2, 3, 4]]
    assert parts[1].token_ids.tolist() == [[5, 6, 7, 8]]

    token_limited_parts = split_training_batch(
        batch,
        max_samples=None,
        max_tokens=3,
    )

    assert [part.seq_lens for part in token_limited_parts] == [[2], [3], [4]]


def test_split_training_batches_returns_global_microstep_count():
    rank0 = TrainingBatch(
        token_ids=torch.arange(5).view(1, 5),
        seq_lens=[2, 3],
        loss_mask=torch.ones((1, 5), dtype=torch.bool),
        behavior_logprobs=torch.zeros((1, 5)),
        advantages=torch.zeros((1, 5)),
    )
    rank1 = TrainingBatch(
        token_ids=torch.arange(9).view(1, 9),
        seq_lens=[2, 3, 4],
        loss_mask=torch.ones((1, 9), dtype=torch.bool),
        behavior_logprobs=torch.zeros((1, 9)),
        advantages=torch.zeros((1, 9)),
    )

    splits_by_rank, max_microbatches = split_training_batches_by_rank(
        [rank0, rank1],
        max_samples=1,
        max_tokens=None,
    )

    assert max_microbatches == 3
    assert [len(splits) for splits in splits_by_rank] == [2, 3]
    assert [part.seq_lens for part in splits_by_rank[0]] == [[2], [3]]
    assert [part.seq_lens for part in splits_by_rank[1]] == [[2], [3], [4]]


def test_trainer_schedule_pads_shorter_rank_with_dummy_microbatch():
    rank0 = TrainingBatch(
        token_ids=torch.arange(5).view(1, 5),
        seq_lens=[2, 3],
        loss_mask=torch.ones((1, 5), dtype=torch.bool),
        behavior_logprobs=torch.zeros((1, 5)),
        advantages=torch.zeros((1, 5)),
    )
    rank1 = TrainingBatch(
        token_ids=torch.arange(9).view(1, 9),
        seq_lens=[2, 3, 4],
        loss_mask=torch.ones((1, 9), dtype=torch.bool),
        behavior_logprobs=torch.zeros((1, 9)),
        advantages=torch.zeros((1, 9)),
    )

    schedule = schedule_training_microbatches(
        [rank0, rank1],
        dp_rank=0,
        max_samples=1,
        max_tokens=None,
    )

    assert schedule.max_microbatches == 3
    assert [item.is_real for item in schedule.microbatches] == [True, True, False]
    assert schedule.microbatches[-1].batch.seq_lens == [2]
    assert schedule.max_seq_len == 3


def test_zero_gradient_microbatch_matches_reference_tensor_metadata():
    reference = TrainingBatch(
        token_ids=torch.arange(5, dtype=torch.long).view(1, 5),
        seq_lens=[2, 3],
        loss_mask=torch.ones((1, 5), dtype=torch.bool),
        behavior_logprobs=torch.zeros((1, 5), dtype=torch.float64),
        advantages=torch.zeros((1, 5), dtype=torch.float16),
    )

    dummy = zero_gradient_training_batch_like(reference)

    assert has_loss_tokens(dummy)
    assert dummy.token_ids.shape == (1, 2)
    assert dummy.loss_mask.tolist() == [[False, True]]
    assert dummy.token_ids.dtype == reference.token_ids.dtype
    assert dummy.loss_mask.dtype == reference.loss_mask.dtype
    assert dummy.behavior_logprobs.dtype == reference.behavior_logprobs.dtype
    assert dummy.advantages.dtype == reference.advantages.dtype
    assert dummy.advantages.sum().item() == 0


def test_metric_accumulator_uses_same_keys_for_inactive_values():
    real_accumulator = MetricAccumulator()
    dummy_accumulator = MetricAccumulator()
    loss_metrics = {
        "loss/mean": torch.tensor(1.0),
        "loss/ratio/mean": torch.tensor(2.0),
    }
    drift_metrics = {
        "bit_wise/logprob_diff/mean": torch.tensor(3.0),
        "bit_wise/ratio_tokens_different/mean": torch.tensor(4.0),
    }
    max_metrics = {
        "bit_wise/logprob_diff/max": torch.tensor(5.0),
        "train/microbatch_tokens/max": torch.tensor(6.0),
        "train/microbatch_samples/max": torch.tensor(7.0),
    }

    for active, accumulator in [
        (True, real_accumulator),
        (False, dummy_accumulator),
    ]:
        accumulator.add_sum({**loss_metrics, **drift_metrics}, active=active)
        accumulator.add_max(max_metrics, active=active)

    assert list(real_accumulator.sum_reduced_metrics) == list(
        dummy_accumulator.sum_reduced_metrics
    )
    assert list(real_accumulator.max_reduced_metrics) == list(
        dummy_accumulator.max_reduced_metrics
    )
    assert (
        sum(value.item() for value in real_accumulator.sum_reduced_metrics.values())
        == 10.0
    )
    assert (
        sum(value.item() for value in dummy_accumulator.sum_reduced_metrics.values())
        == 0.0
    )
    assert (
        sum(value.item() for value in real_accumulator.max_reduced_metrics.values())
        == 18.0
    )
    assert (
        sum(value.item() for value in dummy_accumulator.max_reduced_metrics.values())
        == 0.0
    )


def test_provisioner_respects_parent_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    provisioner = Provisioner(total_gpus=4)

    provisioner.allocate(2)()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"
    provisioner.allocate(2)()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


def test_rollout_to_replay_sample_masks_multiturn_prefix_continuation():
    rollout = RolloutOutput(
        group_id="g0",
        sample_idx=0,
        status=RolloutStatus.COMPLETED,
        reward=1.0,
        turns=[
            RolloutTurn(
                prompt_token_ids=[10, 11],
                response_token_ids=[20],
                response_logprobs=[-0.2],
                policy_version=0,
            ),
            RolloutTurn(
                prompt_token_ids=[10, 11, 20, 12],
                response_token_ids=[21, 22],
                response_logprobs=[-0.3, -0.4],
                policy_version=0,
            ),
        ],
    )

    [sample] = rollouts_to_replay_samples([rollout])

    assert sample.token_ids == [10, 11, 20, 12, 21, 22]
    assert sample.loss_mask == [0, 0, 1, 0, 1, 1]
    assert sample.behavior_logprobs == [0.0, 0.0, -0.2, 0.0, -0.3, -0.4]
    assert sample.advantage == 0.0


def test_rollout_to_replay_samples_normalizes_advantages_by_group_std():
    def rollout(sample_idx: int, reward: float) -> RolloutOutput:
        return RolloutOutput(
            group_id="g0",
            sample_idx=sample_idx,
            status=RolloutStatus.COMPLETED,
            reward=reward,
            turns=[
                RolloutTurn(
                    prompt_token_ids=[10],
                    response_token_ids=[20 + sample_idx],
                    response_logprobs=[-0.1],
                    policy_version=0,
                )
            ],
        )

    samples = rollouts_to_replay_samples([rollout(0, 0.0), rollout(1, 1.0)])

    assert [sample.advantage for sample in samples] == pytest.approx([-1.0, 1.0])


def test_replay_group_derives_and_validates_group_id():
    rollout = RolloutOutput(
        group_id="g0",
        sample_idx=0,
        status=RolloutStatus.COMPLETED,
        reward=1.0,
        turns=[
            RolloutTurn(
                prompt_token_ids=[1],
                response_token_ids=[2],
                response_logprobs=[-0.1],
                policy_version=4,
            )
        ],
    )
    sample = ReplaySample(
        token_ids=[1, 2],
        loss_mask=[0, 1],
        behavior_logprobs=[0.0, -0.1],
        advantage=0.5,
        group_id="g0",
        sample_idx=0,
        behavior_version=4,
        reward=1.0,
    )

    group = ReplayGroup.from_rollouts(samples=[sample], rollouts=[rollout])

    assert group.group_id == "g0"
    assert group.behavior_version == 4
    assert group.max_behavior_version == 4

    mismatched_rollout = dataclasses.replace(rollout, group_id="g1")
    with pytest.raises(ValueError, match="rollout group_ids"):
        ReplayGroup.from_rollouts(
            samples=[sample],
            rollouts=[rollout, mismatched_rollout],
        )

    mismatched_sample = dataclasses.replace(sample, group_id="g1")
    with pytest.raises(ValueError, match="sample group_ids"):
        ReplayGroup.from_rollouts(
            samples=[mismatched_sample],
            rollouts=[rollout],
        )


def test_max_turn_truncation_does_not_train_on_nonterminal_reward():
    class TokenEnvStub:
        async def initial_prompt(self) -> PromptState:
            return PromptState(token_ids=[1, 2], messages=[])

        async def step(self, completion: Completion) -> TokenStep:
            return TokenStep(
                env_step=EnvStep(
                    reward=0.75,
                    reward_components={"shaped": 0.75},
                    done=False,
                ),
                response_messages=[],
                next_prompt=PromptState(token_ids=[1, 2, 3], messages=[]),
            )

    async def run() -> None:
        async def completion_fn(
            *,
            prompt_token_ids: list[int],
            sampling: SamplingConfig,
            request_id: str,
        ) -> Completion:
            return Completion(
                policy_version=0,
                token_ids=[3],
                token_logprobs=[-0.1],
                finish_reason="stop",
            )

        rollout = await do_single_rollout(
            token_env=TokenEnvStub(),
            completion_fn=completion_fn,
            sampling=SamplingConfig(n=1),
            group_id="g0",
            sample_idx=0,
            max_turns=1,
        )

        assert rollout.status == RolloutStatus.TRUNCATED
        assert rollout.reward == 0.0
        assert rollout.reward_components == {"max_turns": 1.0}

    asyncio.run(run())


def test_token_env_scores_length_stopped_response():
    class EnvStub:
        def __init__(self):
            self.messages = []

        async def reset(self):
            raise AssertionError("reset is not needed for a length-stop step")

        async def step(self, assistant_message):
            self.messages.append(assistant_message)
            return EnvStep(
                reward=0.75,
                reward_components={"task_score": 0.75},
                done=True,
                status=RolloutStatus.COMPLETED,
            )

        async def close(self):
            pass

    class RendererStub:
        def parse_response(self, token_ids):
            assert token_ids == [10, 11]
            return SimpleNamespace(
                content="partial answer",
                reasoning_content=None,
                tool_calls=None,
            )

    async def run() -> None:
        env = EnvStub()
        token_step = await TokenEnv(env, RendererStub()).step(
            Completion(
                policy_version=0,
                token_ids=[10, 11],
                token_logprobs=[-0.1, -0.2],
                finish_reason="length",
            )
        )

        assert token_step.env_step.status == RolloutStatus.TRUNCATED
        assert token_step.env_step.reward == 0.75
        assert token_step.env_step.reward_components == {"task_score": 0.75}
        assert env.messages == [{"role": "assistant", "content": "partial answer"}]
        assert token_step.response_messages == [
            {"role": "assistant", "content": "partial answer"}
        ]

    asyncio.run(run())


def test_token_env_length_stop_falls_back_to_truncation_reward():
    class EnvStub:
        async def reset(self):
            raise AssertionError("reset is not needed for a length-stop step")

        async def step(self, assistant_message):
            return EnvStep(done=False, reward_components={"partial": 1.0})

        async def close(self):
            pass

    class RendererStub:
        def parse_response(self, token_ids):
            return SimpleNamespace(
                content="partial intermediate",
                reasoning_content=None,
                tool_calls=None,
            )

    async def run() -> None:
        token_step = await TokenEnv(
            EnvStub(),
            RendererStub(),
            TokenEnvConfig(truncation_reward=-0.25),
        ).step(
            Completion(
                policy_version=0,
                token_ids=[10],
                token_logprobs=[-0.1],
                finish_reason="length",
            )
        )

        assert token_step.env_step.status == RolloutStatus.TRUNCATED
        assert token_step.env_step.reward == -0.25
        assert token_step.env_step.reward_components == {
            "partial": 1.0,
            "length_stop": 1.0,
        }

    asyncio.run(run())


def test_sum_digits_dataset_and_builder_have_separate_roles():
    dataset = SumDigitsDataset.Config(seed=123).build()
    builder = SumDigitsBuilder.Config(
        correctness_reward=2.0,
        format_reward=0.5,
    ).build()
    example = dataset.sample_group(sample_step=2, group_idx=7)

    async def run() -> None:
        env = builder.build(example=example)
        reset = await env.reset()
        await env.close()

        assert example.group_id == "sum_digits/step=2/group=7"
        assert isinstance(example.payload, SumDigitsExample)
        assert example.payload.values
        assert isinstance(example.payload.target, int)
        assert reset.messages[0]["role"] == "system"
        assert reset.messages[1]["role"] == "user"

    asyncio.run(run())


def test_sum_digits_builder_rejects_stale_dict_payload():
    builder = SumDigitsBuilder.Config().build()
    example = EnvExample(
        group_id="bad",
        sample_step=0,
        group_idx=0,
        payload={"values": [12, 34], "target": 10},
    )

    with pytest.raises(ValueError, match="SumDigitsExample"):
        builder.build(example=example)


def test_run_rollout_group_closes_partially_built_envs_on_build_failure():
    class EnvStub:
        def __init__(self):
            self.closed = False

        async def reset(self):
            raise AssertionError("rollout should not start after build failure")

        async def step(self, assistant_message):
            raise AssertionError("rollout should not start after build failure")

        async def close(self):
            self.closed = True

    class BuilderStub:
        def __init__(self):
            self.first_env = EnvStub()
            self.calls = 0

        def build(self, *, example: EnvExample):
            self.calls += 1
            if self.calls == 1:
                return self.first_env
            raise RuntimeError("build failed")

    async def run() -> None:
        builder = BuilderStub()

        with pytest.raises(RuntimeError, match="build failed"):
            await run_rollout_group(
                env_builder=builder,
                example=EnvExample(group_id="g0", sample_step=0, group_idx=0),
                group_size=2,
                renderer=None,
                completion_fn=None,
                sampling=SamplingConfig(n=1),
                max_turns=1,
                token_env_config=None,
            )

        assert builder.first_env.closed

    asyncio.run(run())


def test_validation_metric_rename_handles_reward_summary_keys():
    rollouts = [
        RolloutOutput(
            group_id="g0",
            sample_idx=0,
            status=RolloutStatus.COMPLETED,
            reward=1.0,
            turns=[
                RolloutTurn(
                    prompt_token_ids=[1, 2],
                    response_token_ids=[3],
                    response_logprobs=[-0.1],
                    policy_version=0,
                )
            ],
        )
    ]

    metrics = build_rollout_metrics(
        rollouts,
        generation_metrics=[],
        prefix="rollout",
    )
    renamed = [
        rename_metric_prefix(metric, old_prefix="rollout/", new_prefix="validation/")
        for metric in metrics
    ]
    aggregate = m.MetricsProcessor._aggregate_metrics(renamed)

    assert aggregate["validation/reward/_mean"] == 1.0
    assert aggregate["validation/response_length/mean"] == 1.0


def test_generation_scheduler_coalesces_same_tick_requests():
    async def run() -> None:
        calls: list[list[list[int]]] = []
        request_id_calls: list[list[str]] = []

        async def generate_batch(
            prompts: list[list[int]],
            request_ids: list[str],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            request_id_calls.append(list(request_ids))
            return (
                [
                    Completion(
                        policy_version=0,
                        token_ids=[idx],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for idx, _prompt in enumerate(prompts)
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)

        completions = await asyncio.gather(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="a",
            ),
            scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="b",
            ),
        )

        assert calls == [[[1], [2]]]
        assert request_id_calls == [["a", "b"]]
        assert [completion.token_ids for completion in completions] == [[0], [1]]
        aggregate = m.MetricsProcessor._aggregate_metrics(scheduler.pop_metrics())
        assert aggregate["generation_scheduler/batch_size/mean"] == 2
        assert aggregate["generation_scheduler/batch_size/max"] == 2
        assert aggregate["generation_scheduler/pending_depth/mean"] == 0
        assert aggregate["generation_scheduler/pending_depth/max"] == 0
        assert aggregate["generation_scheduler/active_prompts/mean"] == 2
        assert aggregate["generation_scheduler/active_prompts/max"] == 2
        assert aggregate["generation_scheduler/queue_wait_seconds/mean"] >= 0
        assert aggregate["generation_scheduler/queue_wait_seconds/max"] >= 0

    asyncio.run(run())


def test_generation_scheduler_partitions_mixed_sampling_requests():
    async def run() -> None:
        calls: list[tuple[list[list[int]], list[str], float]] = []

        async def generate_batch(
            prompts: list[list[int]],
            request_ids: list[str],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append(
                (
                    [list(prompt) for prompt in prompts],
                    list(request_ids),
                    sampling.temperature,
                )
            )
            return (
                [
                    Completion(
                        policy_version=0,
                        token_ids=[prompt[0]],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for prompt in prompts
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        greedy = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        sampled = SamplingConfig(n=1, temperature=0.7, top_p=1.0, max_tokens=4)

        completions = await asyncio.gather(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=greedy,
                request_id="greedy-a",
            ),
            scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampled,
                request_id="sampled",
            ),
            scheduler.submit(
                prompt_token_ids=[3],
                sampling=greedy,
                request_id="greedy-b",
            ),
        )

        assert calls == [
            ([[1], [3]], ["greedy-a", "greedy-b"], 0.0),
            ([[2]], ["sampled"], 0.7),
        ]
        assert [completion.token_ids for completion in completions] == [[1], [2], [3]]
        await scheduler.close()

    asyncio.run(run())


def test_generation_scheduler_pauses_new_admission_until_resume():
    async def run() -> None:
        calls: list[list[list[int]]] = []
        active_started = asyncio.Event()
        finish_active = asyncio.Event()
        version = 0

        async def generate_batch(
            prompts: list[list[int]],
            request_ids: list[str],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            if prompts == [[1]]:
                active_started.set()
                await finish_active.wait()
            return (
                [
                    Completion(
                        policy_version=version,
                        token_ids=[prompt[0]],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for prompt in prompts
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        first = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="first",
            )
        )
        await active_started.wait()

        pause = asyncio.create_task(scheduler.pause_for_weight_sync())
        await asyncio.sleep(0)
        assert not pause.done()

        second = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="second",
            )
        )
        await asyncio.sleep(0)
        assert calls == [[[1]]]
        assert not second.done()

        finish_active.set()
        await pause
        assert not second.done()

        version = 1
        await scheduler.resume_after_weight_sync()

        assert (await first).policy_version == 0
        assert (await second).policy_version == 1
        assert calls == [[[1]], [[2]]]

    asyncio.run(run())


def test_generation_scheduler_close_drops_cancelled_queued_request():
    async def run() -> None:
        calls: list[list[list[int]]] = []

        async def generate_batch(
            prompts: list[list[int]],
            request_ids: list[str],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            return (
                [
                    Completion(
                        policy_version=0,
                        token_ids=[prompt[0]],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for prompt in prompts
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        request = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="queued",
            )
        )
        await asyncio.sleep(0)

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await scheduler.close()

        assert calls == []

    asyncio.run(run())


def test_generation_scheduler_close_waits_for_active_generation():
    async def run() -> None:
        calls: list[list[list[int]]] = []
        active_started = asyncio.Event()
        finish_active = asyncio.Event()

        async def generate_batch(
            prompts: list[list[int]],
            request_ids: list[str],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            active_started.set()
            await finish_active.wait()
            return (
                [
                    Completion(
                        policy_version=0,
                        token_ids=[prompt[0]],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for prompt in prompts
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        request = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="active",
            )
        )
        await active_started.wait()

        close = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0)
        assert not close.done()
        with pytest.raises(RuntimeError, match="generation scheduler is closed"):
            await scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="closed",
            )

        finish_active.set()
        await close

        assert (await request).token_ids == [1]
        assert calls == [[[1]]]

    asyncio.run(run())


def test_generation_scheduler_propagates_generator_exception_to_pending_batch():
    async def run() -> None:
        async def generate_batch(
            prompts: list[list[int]],
            request_ids: list[str],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            raise ValueError(f"bad batch: {prompts}")

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        first = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="first",
            )
        )
        second = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="second",
            )
        )

        with pytest.raises(ValueError, match="bad batch"):
            await first
        with pytest.raises(ValueError, match="bad batch"):
            await second
        await scheduler.close()

    asyncio.run(run())


def test_rltrainer_close_times_out_hung_actor_and_stops_mesh():
    async def run() -> None:
        class HungClose:
            def call(self):
                return asyncio.sleep(3600)

        class FakeMesh:
            def __init__(self) -> None:
                self.stopped = False

            async def stop(self) -> None:
                self.stopped = True

        class FakeMetrics:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        mesh = FakeMesh()
        metrics = FakeMetrics()
        trainer = RLTrainer.__new__(RLTrainer)
        trainer.config = SimpleNamespace(actor_close_timeout_s=0.01)
        trainer.trainer = SimpleNamespace(close=HungClose())
        trainer.generator = None
        trainer.metrics_processor = metrics
        trainer._proc_meshes = [mesh]

        await trainer.close()

        assert metrics.closed
        assert mesh.stopped
        assert trainer._proc_meshes == []

    asyncio.run(run())


def test_empty_collate_produces_no_loss_dummy_row():
    batch = RLTrainer._collate_samples([])

    assert batch.token_ids.shape == (1, 1)
    assert batch.seq_lens == [1]
    assert not batch.loss_mask.any()


def test_rollout_sample_logger_caps_groups_per_step(tmp_path):
    logger = RolloutSampleLogger(str(tmp_path), max_groups_per_step=1)
    rollouts = [
        RolloutOutput(
            group_id=group_id,
            sample_idx=sample_idx,
            status=RolloutStatus.COMPLETED,
            reward=1.0,
            turns=[
                RolloutTurn(
                    prompt_token_ids=[1, 2],
                    response_token_ids=([3] if sample_idx == 0 else [3, 4, 5]),
                    response_logprobs=(
                        [-0.1]
                        if sample_idx == 0
                        else [-0.3, float("inf"), float("nan")]
                    ),
                    policy_version=sample_idx,
                    prompt_messages=[{"role": "user", "content": "sort"}],
                    response_messages=[{"role": "assistant", "content": "done"}],
                )
            ],
        )
        for group_id, sample_idx in [("g0", 0), ("g0", 1), ("g1", 0)]
    ]

    logger.write(step=3, phase="train", rollouts=rollouts)
    logger.write(step=3, phase="train", rollouts=rollouts)
    logger.write(step=3, phase="validation", rollouts=rollouts)

    records = [
        json.loads(line)
        for line in (tmp_path / "rollout_samples.jsonl").read_text().splitlines()
    ]

    assert [
        (record["phase"], record["group_id"], record["sample_idx"])
        for record in records
    ] == [
        ("train", "g0", 0),
        ("train", "g0", 1),
        ("validation", "g0", 0),
        ("validation", "g0", 1),
    ]
    assert records[0]["turns"][0]["messages"][0] == {
        "role": "user",
        "content": "sort",
    }
    assert records[0]["turns"][0]["response_logprob_count"] == 1
    assert records[0]["turns"][0]["response_logprob_nonfinite_count"] == 0
    assert records[0]["turns"][0]["response_logprob_finite_min"] == pytest.approx(-0.1)
    assert records[0]["turns"][0]["response_logprob_finite_max"] == pytest.approx(-0.1)
    assert records[1]["turns"][0]["response_logprob_count"] == 3
    assert records[1]["turns"][0]["response_logprob_nonfinite_count"] == 2
    assert records[1]["turns"][0]["response_logprob_finite_min"] == pytest.approx(-0.3)
    assert records[1]["turns"][0]["response_logprob_finite_max"] == pytest.approx(-0.3)


def test_has_advantage_signal_detects_nonzero_group_signal():
    def sample(group_id: str, advantage: float) -> ReplaySample:
        return ReplaySample(
            token_ids=[1, 2],
            loss_mask=[0, 1],
            behavior_logprobs=[0.0, -0.1],
            advantage=advantage,
            group_id=group_id,
            sample_idx=0,
            behavior_version=0,
            reward=advantage,
        )

    assert not has_advantage_signal([sample("zero", 0.0)])
    assert has_advantage_signal([sample("signal", 0.0), sample("signal", 0.5)])


def test_rollout_drop_counters_fail_after_consecutive_no_signal_groups():
    counters = _RolloutDropCounters(max_no_signal_groups=3)

    counters.record_empty()
    counters.record_zero_advantage()
    with pytest.raises(RuntimeError, match="no trainable rollout groups admitted"):
        counters.record_zero_advantage()


def test_rollout_drop_counters_reset_after_admitted_group():
    counters = _RolloutDropCounters(max_no_signal_groups=3)

    counters.record_empty()
    counters.record_zero_advantage()
    counters.record_admitted()
    counters.record_zero_advantage()
    assert counters.consecutive_dropped_groups == 1
    assert counters.consecutive_empty_groups == 0
    assert counters.consecutive_zero_advantage_groups == 1


def test_rollout_drop_counters_emit_zero_advantage_reward_metrics():
    counters = _RolloutDropCounters(max_no_signal_groups=None)

    counters.record_zero_advantage([0.25, 0.75])
    stats = counters.pop()
    aggregate = m.MetricsProcessor._aggregate_metrics(stats.metrics)

    assert stats.empty_groups == 0
    assert stats.zero_advantage_groups == 1
    assert aggregate["rollout/dropped_zero_advantage_reward/_mean"] == 0.5
    assert aggregate["rollout/dropped_zero_advantage_reward/_max"] == 0.75


def test_replay_wait_surfaces_no_signal_producer_error():
    async def run() -> None:
        buffer = ReplayBuffer(max_groups=1)
        counters = _RolloutDropCounters(max_no_signal_groups=1)

        async def producer() -> None:
            try:
                counters.record_zero_advantage()
            except RuntimeError:
                await buffer.close()
                raise

        task = asyncio.create_task(producer())
        batch = await buffer.get_batch(min_groups=1, train_version=0)

        assert not batch.samples
        with pytest.raises(RuntimeError, match="no trainable rollout groups"):
            await _raise_rollout_task_errors([task], timeout_s=1.0)

    asyncio.run(run())


def test_train_step_metric_builder_emits_replay_timing_and_trace_scalars():
    sample = ReplaySample(
        token_ids=[1, 2],
        loss_mask=[0, 1],
        behavior_logprobs=[0.0, -0.1],
        advantage=0.5,
        group_id="g0",
        sample_idx=0,
        behavior_version=2,
        reward=1.0,
    )
    rollout = RolloutOutput(
        group_id="g0",
        sample_idx=0,
        status=RolloutStatus.COMPLETED,
        reward=1.0,
        turns=[
            RolloutTurn(
                prompt_token_ids=[1],
                response_token_ids=[2],
                response_logprobs=[-0.1],
                policy_version=2,
            )
        ],
    )
    batch = ReplayBatch(
        groups=[
            ReplayGroup(
                group_id="g0",
                samples=[sample],
                rollouts=[rollout],
                behavior_version=2,
                max_behavior_version=3,
            )
        ],
        samples=[sample],
        stats=ReplayBufferStats(
            num_groups=1,
            num_samples=1,
            num_loss_tokens=1,
            num_dropped_stale_groups=0,
            max_observed_age_steps=1,
            depth_groups=2,
        ),
    )

    fwd_bwd_metrics = {
        "loss/mean": 0.25,
        "loss/ratio/nonfinite_frac": 0.0,
        "loss/logprob/policy_nonfinite_frac": 0.1,
        "loss/logprob/behavior_nonfinite_frac": 0.2,
        "bit_wise/nonfinite_logprob_frac": 0.3,
    }
    optimizer_metrics = {"train/lr": 1e-6}
    timings = _TrainStepTimings(
        step_s=2.0,
        replay_wait_s=0.1,
        train_s=0.2,
        checkpoint_s=0.3,
        weight_sync=_WeightSyncTimings(
            admission_drain_s=0.4,
            push_s=0.5,
            pull_s=0.6,
            total_s=1.5,
        ),
    )

    metrics = build_train_step_metrics(
        samples=[sample],
        replay_batch=batch,
        rollouts=[rollout],
        live_generation_metrics=[m.Metric("generator/live/tokens", m.NoReduce(4.0))],
        fwd_bwd_metrics=fwd_bwd_metrics,
        optimizer_metrics=optimizer_metrics,
        checkpoint_saved=True,
        timings=timings,
        dropped_empty_groups=1,
        dropped_zero_advantage_groups=2,
        drop_metrics=[
            m.Metric(
                "rollout/dropped_zero_advantage_reward",
                m.SummaryStats.from_list([0.25, 0.75]),
            )
        ],
        train_version=7,
    )
    trace_scalars = _build_train_step_trace_scalars(
        replay_batch=batch,
        fwd_bwd_metrics=fwd_bwd_metrics,
        optimizer_metrics=optimizer_metrics,
        checkpoint_saved=True,
        timings=timings,
        dropped_empty_groups=1,
        dropped_zero_advantage_groups=2,
        train_version=7,
    )
    aggregate = m.MetricsProcessor._aggregate_metrics(metrics)

    assert aggregate["loss/mean"] == 0.25
    assert aggregate["checkpoint/saved"] == 1.0
    assert aggregate["perf/tokens_per_second"] == 1.0
    assert aggregate["replay/policy_version/train"] == 7.0
    assert aggregate["replay/policy_version/behavior_min"] == 2.0
    assert aggregate["replay/policy_version/behavior_max"] == 3.0
    assert aggregate["generator/live/tokens"] == 4.0
    assert aggregate["rollout/dropped_zero_advantage_reward/_mean"] == 0.5
    assert aggregate["rollout/dropped_zero_advantage_reward/_max"] == 0.75
    assert trace_scalars["replay.buffer_depth_groups"] == 2
    assert trace_scalars["rollout.dropped_zero_advantage_groups"] == 2
    assert trace_scalars["timing.weight_sync_pull_ms"] == 600.0
    assert trace_scalars["loss.ratio.nonfinite_frac"] == 0.0
    assert trace_scalars["loss.logprob.policy_nonfinite_frac"] == 0.1
    assert trace_scalars["loss.logprob.behavior_nonfinite_frac"] == 0.2
    assert trace_scalars["bit_wise.nonfinite_logprob_frac"] == 0.3


def test_train_step_metric_builder_handles_zero_step_duration():
    sample = ReplaySample(
        token_ids=[1, 2],
        loss_mask=[0, 1],
        behavior_logprobs=[0.0, -0.1],
        advantage=0.5,
        group_id="g0",
        sample_idx=0,
        behavior_version=0,
        reward=1.0,
    )
    batch = ReplayBatch(
        groups=[
            ReplayGroup(
                group_id="g0",
                samples=[sample],
                rollouts=[],
                behavior_version=0,
                max_behavior_version=0,
            )
        ],
        samples=[sample],
        stats=ReplayBufferStats(
            num_groups=1,
            num_samples=1,
            num_loss_tokens=1,
            num_dropped_stale_groups=0,
            max_observed_age_steps=0,
            depth_groups=0,
        ),
    )

    zero_timings = _TrainStepTimings(
        step_s=0.0,
        replay_wait_s=0.0,
        train_s=0.0,
        checkpoint_s=0.0,
        weight_sync=_WeightSyncTimings(
            admission_drain_s=0.0,
            push_s=0.0,
            pull_s=0.0,
            total_s=0.0,
        ),
    )

    metrics = build_train_step_metrics(
        samples=[sample],
        replay_batch=batch,
        rollouts=[],
        live_generation_metrics=[],
        fwd_bwd_metrics={
            "loss/mean": 0.0,
            "loss/ratio/nonfinite_frac": 0.0,
            "loss/logprob/policy_nonfinite_frac": 0.0,
            "loss/logprob/behavior_nonfinite_frac": 0.0,
            "bit_wise/nonfinite_logprob_frac": 0.0,
        },
        optimizer_metrics={},
        checkpoint_saved=False,
        timings=zero_timings,
        dropped_empty_groups=0,
        dropped_zero_advantage_groups=0,
        train_version=0,
    )
    aggregate = m.MetricsProcessor._aggregate_metrics(metrics)

    assert aggregate["perf/tokens_per_second"] == 0.0

    with pytest.raises(KeyError, match="loss/ratio/nonfinite_frac"):
        build_train_step_metrics(
            samples=[sample],
            replay_batch=batch,
            rollouts=[],
            live_generation_metrics=[],
            fwd_bwd_metrics={"loss/mean": 0.0},
            optimizer_metrics={},
            checkpoint_saved=False,
            timings=zero_timings,
            dropped_empty_groups=0,
            dropped_zero_advantage_groups=0,
            train_version=0,
        )


def test_optimizer_step_skipped_detects_no_new_policy_version():
    assert RLTrainer._optimizer_step_skipped(
        OptimStepOutput(
            policy_version=2,
            metrics={"train/skipped_nonfinite_grad_norm": 0.0},
        ),
        previous_policy_version=2,
    )


def test_policy_trainer_optim_step_skips_nonfinite_grad_norm():
    class FakeOptimizer:
        def __init__(self) -> None:
            self.stepped = False
            self.zeroed = False

        def step(self) -> None:
            self.stepped = True

        def zero_grad(self) -> None:
            self.zeroed = True

    class FakeScheduler:
        def __init__(self) -> None:
            self.stepped = False

        def get_last_lr(self) -> list[float]:
            return [1e-6]

        def step(self) -> None:
            self.stepped = True

    async def run() -> None:
        trainer = PolicyTrainer.__new__(PolicyTrainer)
        param = torch.nn.Parameter(torch.tensor([1.0]))
        param.grad = torch.tensor([float("nan")])
        optimizer = FakeOptimizer()
        scheduler = FakeScheduler()
        trainer.model_parts = [SimpleNamespace(parameters=lambda: [param])]
        trainer.config = SimpleNamespace(training=SimpleNamespace(max_norm=1.0))
        trainer.parallel_dims = SimpleNamespace(get_optional_mesh=lambda name: None)
        trainer.optimizers = optimizer
        trainer.lr_schedulers = SimpleNamespace(schedulers=[scheduler])
        trainer.policy_version = 7

        result = await PolicyTrainer.optim_step.__dict__["_method"](trainer)

        assert result.policy_version == 7
        assert result.metrics["train/skipped_nonfinite_grad_norm"] == 1.0
        assert not optimizer.stepped
        assert optimizer.zeroed
        assert not scheduler.stepped

    asyncio.run(run())
    assert RLTrainer._optimizer_step_skipped(
        OptimStepOutput(
            policy_version=3,
            metrics={"train/skipped_nonfinite_grad_norm": 1.0},
        ),
        previous_policy_version=2,
    )
    assert not RLTrainer._optimizer_step_skipped(
        OptimStepOutput(
            policy_version=3,
            metrics={"train/skipped_nonfinite_grad_norm": 0.0},
        ),
        previous_policy_version=2,
    )


def test_replay_buffer_blocks_until_batch_is_ready():
    async def run() -> None:
        buffer = ReplayBuffer(max_groups=1)

        def sample(idx: int) -> ReplaySample:
            return ReplaySample(
                token_ids=[idx, idx + 10],
                loss_mask=[0, 1],
                behavior_logprobs=[0.0, -0.1],
                advantage=1.0,
                group_id=f"g{idx}",
                sample_idx=0,
                behavior_version=0,
                reward=1.0,
            )

        async def producer() -> None:
            await buffer.put(
                ReplayGroup(
                    group_id="g0",
                    samples=[sample(0)],
                    rollouts=[],
                    behavior_version=0,
                    max_behavior_version=0,
                )
            )
            await buffer.put(
                ReplayGroup(
                    group_id="g1",
                    samples=[sample(1)],
                    rollouts=[],
                    behavior_version=0,
                    max_behavior_version=0,
                )
            )
            await buffer.close()

        producer_task = asyncio.create_task(producer())
        batch = await buffer.get_batch(min_groups=2, train_version=0)
        await producer_task

        assert [sample.group_id for sample in batch.samples] == ["g0", "g1"]
        assert batch.stats.num_groups == 2
        assert batch.stats.num_loss_tokens == 2

    asyncio.run(run())


def test_replay_buffer_batches_by_group_count_not_sample_count():
    async def run() -> None:
        buffer = ReplayBuffer(max_groups=2)

        def sample(idx: int) -> ReplaySample:
            return ReplaySample(
                token_ids=[idx, idx + 10],
                loss_mask=[0, 1],
                behavior_logprobs=[0.0, -0.1],
                advantage=1.0,
                group_id=f"g{idx}",
                sample_idx=idx,
                behavior_version=0,
                reward=1.0,
            )

        await buffer.put(
            ReplayGroup(
                group_id="g0",
                samples=[sample(0), sample(1), sample(2)],
                rollouts=[],
                behavior_version=0,
                max_behavior_version=0,
            )
        )
        await buffer.put(
            ReplayGroup(
                group_id="g1",
                samples=[sample(3)],
                rollouts=[],
                behavior_version=0,
                max_behavior_version=0,
            )
        )

        batch = await buffer.get_batch(min_groups=2, train_version=0)

        assert [group.group_id for group in batch.groups] == ["g0", "g1"]
        assert batch.stats.num_groups == 2
        assert batch.stats.num_samples == 4

    asyncio.run(run())


def test_replay_buffer_drops_stale_groups_and_continues():
    async def run() -> None:
        buffer = ReplayBuffer(max_groups=2, max_age_steps=1)

        def group(group_id: str, behavior_version: int) -> ReplayGroup:
            sample = ReplaySample(
                token_ids=[1, 2],
                loss_mask=[0, 1],
                behavior_logprobs=[0.0, -0.1],
                advantage=1.0,
                group_id=group_id,
                sample_idx=0,
                behavior_version=behavior_version,
                reward=1.0,
            )
            return ReplayGroup(
                group_id=group_id,
                samples=[sample],
                rollouts=[],
                behavior_version=behavior_version,
                max_behavior_version=behavior_version,
            )

        await buffer.put(group("stale", behavior_version=0))
        await buffer.put(group("fresh", behavior_version=3))

        batch = await buffer.get_batch(min_groups=1, train_version=3)

        assert [group.group_id for group in batch.groups] == ["fresh"]
        assert batch.stats.num_dropped_stale_groups == 1
        assert batch.stats.max_observed_age_steps == 3
        assert batch.stats.depth_groups == 0

    asyncio.run(run())
