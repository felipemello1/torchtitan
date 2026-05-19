# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
from types import SimpleNamespace

import pytest
import torch

from torchtitan.experiments.rl.envs import EnvExample, EnvStep
from torchtitan.experiments.rl.envs.token_env import PromptState, TokenEnv, TokenStep
from torchtitan.experiments.rl.generation_scheduler import GenerationScheduler
from torchtitan.experiments.rl.grpo import (
    _raise_rollout_task_errors,
    _RolloutDropCounters,
    _TrainStepTimings,
    _WeightSyncTimings,
    GRPOLoss,
    Provisioner,
    RLTrainer,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    build_rollout_metrics,
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
from torchtitan.experiments.rl.sampling import SamplingConfig
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
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
    TrainingBatch,
)


def test_sampling_config_default_is_one_completion():
    assert SamplingConfig().n == 1


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


def test_token_env_keeps_truncated_response_message_for_logging():
    class EnvStub:
        async def reset(self):
            raise AssertionError("reset is not needed for a length-stop step")

        async def step(self, assistant_message):
            raise AssertionError("length-stop responses should not step the env")

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
        token_step = await TokenEnv(EnvStub(), RendererStub()).step(
            Completion(
                policy_version=0,
                token_ids=[10, 11],
                token_logprobs=[-0.1, -0.2],
                finish_reason="length",
            )
        )

        assert token_step.env_step.status == RolloutStatus.TRUNCATED
        assert token_step.env_step.reward_components == {"length_stop": 1.0}
        assert token_step.response_messages == [
            {"role": "assistant", "content": "partial answer"}
        ]

    asyncio.run(run())


def test_sum_digits_dataset_and_builder_have_separate_roles():
    dataset = SumDigitsDataset.Config(seed=123).build()
    builder = SumDigitsBuilder.Config(
        correctness_reward=2.0,
        format_reward=0.5,
    ).build()
    example = dataset.sample_group(step=2, group_idx=7)

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
        step=0,
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
                example=EnvExample(group_id="g0", step=0, group_idx=0),
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

        async def generate_batch(
            prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
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
        assert [completion.token_ids for completion in completions] == [[0], [1]]
        aggregate = m.MetricsProcessor._aggregate_metrics(scheduler.pop_metrics())
        assert aggregate["generation_scheduler/batch_size/mean"] == 2
        assert aggregate["generation_scheduler/batch_size/max"] == 2
        assert aggregate["generation_scheduler/pending_depth/mean"] == 0
        assert aggregate["generation_scheduler/pending_depth/max"] == 0
        assert aggregate["generation_scheduler/active_requests/mean"] == 2
        assert aggregate["generation_scheduler/active_requests/max"] == 2

    asyncio.run(run())


def test_generation_scheduler_pauses_new_admission_until_resume():
    async def run() -> None:
        calls: list[list[list[int]]] = []
        active_started = asyncio.Event()
        finish_active = asyncio.Event()
        version = 0

        async def generate_batch(
            prompts: list[list[int]],
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
                    response_token_ids=[3],
                    response_logprobs=[-0.1],
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
        batch = await buffer.get_batch(min_samples=1, train_version=0)

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

    metrics, trace_scalars = RLTrainer._build_train_step_metrics(
        samples=[sample],
        replay_batch=batch,
        rollouts=[rollout],
        live_generation_metrics=[m.Metric("generator/live/tokens", m.NoReduce(4.0))],
        fwd_bwd_metrics={"loss/mean": 0.25},
        optimizer_metrics={"train/lr": 1e-6},
        checkpoint_saved=True,
        timings=_TrainStepTimings(
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
        ),
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
    assert trace_scalars["replay.buffer_depth_groups"] == 2
    assert trace_scalars["rollout.dropped_zero_advantage_groups"] == 2
    assert trace_scalars["timing.weight_sync_pull_ms"] == 600.0


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

    metrics, _ = RLTrainer._build_train_step_metrics(
        samples=[sample],
        replay_batch=batch,
        rollouts=[],
        live_generation_metrics=[],
        fwd_bwd_metrics={"loss/mean": 0.0},
        optimizer_metrics={},
        checkpoint_saved=False,
        timings=_TrainStepTimings(
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
        ),
        dropped_empty_groups=0,
        dropped_zero_advantage_groups=0,
        train_version=0,
    )
    aggregate = m.MetricsProcessor._aggregate_metrics(metrics)

    assert aggregate["perf/tokens_per_second"] == 0.0


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
        batch = await buffer.get_batch(min_samples=2, train_version=0)
        await producer_task

        assert [sample.group_id for sample in batch.samples] == ["g0", "g1"]
        assert batch.stats.num_groups == 2
        assert batch.stats.num_loss_tokens == 2

    asyncio.run(run())
