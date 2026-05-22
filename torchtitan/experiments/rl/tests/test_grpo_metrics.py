# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for rollout metrics, replay rows, and reducers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from torchtitan.config import BatchConfig
from torchtitan.experiments.rl.grpo import Batcher
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    _TrainStepTimings,
    _WeightSyncTimings,
    build_rollout_metrics,
    build_train_step_metrics,
    validate_train_step_fwd_bwd_metrics,
)
from torchtitan.experiments.rl.replay import ReplayBatch
from torchtitan.experiments.rl.types import (
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
)


def _turn(
    *,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
    logprobs: list[float] | None = None,
    policy_version: int = 3,
) -> RolloutTurn:
    response_ids = response_ids if response_ids is not None else [7, 8]
    return RolloutTurn(
        prompt_token_ids=prompt_ids if prompt_ids is not None else [5],
        response_token_ids=response_ids,
        response_logprobs=logprobs if logprobs is not None else [-0.3, -0.4],
        policy_version=policy_version,
        prompt_messages=[{"role": "user", "content": "question"}],
        response_messages=[{"role": "assistant", "content": "answer"}],
        finish_reason="stop",
    )


def _rollout(
    *,
    group_id: str = "g0",
    sample_idx: int = 0,
    reward: float = 1.0,
    components: dict[str, float] | None = None,
    turn: RolloutTurn | None = None,
) -> RolloutOutput:
    return RolloutOutput(
        group_id=group_id,
        sample_idx=sample_idx,
        status=RolloutStatus.COMPLETED,
        turns=[turn or _turn()],
        reward=reward,
        reward_components=components or {"correctness": reward},
    )


def test_rollout_metrics_average_observed_reward_components_only() -> None:
    rollouts = [
        _rollout(reward=1.0, components={"correctness": 1.0, "format": 0.5}),
        _rollout(sample_idx=1, reward=0.0, components={"correctness": 0.0}),
    ]

    agg = m.MetricsProcessor._aggregate_metrics(
        build_rollout_metrics("rollout", rollouts)
    )

    assert agg["rollout/reward/component/correctness/mean"] == pytest.approx(0.5)
    assert agg["rollout/reward/component/format/mean"] == pytest.approx(0.5)
    assert agg["rollout/error_rate/mean"] == pytest.approx(0.0)


def test_rollout_metrics_aggregate_across_collection_waves() -> None:
    wave0 = build_rollout_metrics(
        "rollout",
        [
            _rollout(group_id="g0", sample_idx=0, reward=1.0),
            _rollout(group_id="g0", sample_idx=1, reward=1.0),
        ],
    )
    wave1 = build_rollout_metrics(
        "rollout",
        [
            _rollout(group_id="g1", sample_idx=0, reward=1.0),
            _rollout(group_id="g1", sample_idx=1, reward=0.0),
        ],
    )

    agg = m.MetricsProcessor._aggregate_metrics([*wave0, *wave1])

    assert agg["reward/zero_std_frac/mean"] == pytest.approx(0.5)


def test_batcher_replay_sample_path_masks_advantage() -> None:
    batcher = Batcher(
        Batcher.Config(batch=BatchConfig(local_batch_size=1, global_batch_size=1)),
        pad_id=0,
    )
    sample = ReplaySample(
        token_ids=[5, 7, 8],
        loss_mask=[0, 1, 1],
        ref_logprobs=[0.0, -0.3, -0.4],
        advantage=0.5,
        group_id="g0",
        sample_idx=0,
        behavior_version=3,
        reward=1.0,
    )

    row = list(batcher._iter_training_samples([sample]))[0]

    assert row == {
        "input_ids": [5, 7, 8],
        "ref_logprobs": [0.0, -0.3, -0.4],
        "loss_mask": [0.0, 1.0, 1.0],
        "advantages": [0.0, 0.5, 0.5],
    }


def _no_reduce_values(metrics: list[m.Metric]) -> dict[str, float]:
    return {
        metric.key: metric.value.value
        for metric in metrics
        if isinstance(metric.value, m.NoReduce)
    }


def test_train_step_metrics_validation_requires_health_metrics() -> None:
    with pytest.raises(KeyError, match="ref_logprob_nonfinite_frac"):
        validate_train_step_fwd_bwd_metrics(
            {"health/loss/policy_logprob_nonfinite_frac": 0.0}
        )


def test_build_train_step_metrics_includes_replay_and_step_context() -> None:
    samples = [
        ReplaySample(
            token_ids=[1, 2, 3],
            loss_mask=[0, 1, 1],
            ref_logprobs=[0.0, -0.1, -0.2],
            advantage=0.5,
            group_id="g0",
            sample_idx=0,
            behavior_version=4,
            reward=1.0,
            metrics=(m.Metric("sample/custom", m.NoReduce(9.0)),),
        ),
        ReplaySample(
            token_ids=[4, 5],
            loss_mask=[0, 1],
            ref_logprobs=[0.0, -0.3],
            advantage=-0.5,
            group_id="g1",
            sample_idx=0,
            behavior_version=6,
            reward=0.0,
        ),
    ]
    replay_batch = ReplayBatch(
        samples=samples,
        metrics=[
            m.Metric("replay/num_samples", m.NoReduce(2.0)),
            m.Metric("replay/buffer/dropped_stale_samples", m.NoReduce(1.0)),
        ],
    )
    timings = _TrainStepTimings(
        step_s=2.0,
        replay_wait_s=0.5,
        rollout_s=0.25,
        train_s=1.0,
        checkpoint_s=0.1,
        weight_sync=_WeightSyncTimings(
            admission_drain_s=0.05,
            push_s=0.06,
            pull_s=0.07,
            total_s=0.2,
        ),
    )

    metrics = build_train_step_metrics(
        samples=samples,
        replay_batch=replay_batch,
        rollouts=[_rollout(group_id="g0", sample_idx=0, reward=1.0)],
        live_generation_metrics=[m.Metric("generator/live/queue", m.NoReduce(3.0))],
        fwd_bwd_metrics={
            "loss/mean": 1.5,
            "health/loss/policy_logprob_nonfinite_frac": 0.0,
            "health/loss/ref_logprob_nonfinite_frac": 0.0,
        },
        optimizer_metrics={"train/policy_version": 7.0},
        packing_metrics={"packing/rows": 2.0},
        checkpoint_saved=True,
        timings=timings,
        dropped_empty_groups=2,
        dropped_zero_advantage_groups=1,
        train_version=7,
    )

    values = _no_reduce_values(metrics)
    assert values["replay/policy_version/train"] == 7.0
    assert values["replay/policy_version/behavior_min"] == 4.0
    assert values["replay/policy_version/behavior_max"] == 6.0
    assert values["replay/buffer/stale_drop_rate"] == pytest.approx(1.0 / 3.0)
    assert values["perf/tokens_per_second"] == pytest.approx(2.5)
    assert values["trainer/idle_ratio"] == pytest.approx(0.25)
    assert values["timing/weight_sync_overhead_ratio"] == pytest.approx(0.1)
    assert values["checkpoint/saved"] == 1.0
    assert values["sample/custom"] == 9.0
    assert values["generator/live/queue"] == 3.0
    assert values["loss/mean"] == 1.5


class TestRLTrainerConfigWiring:
    def test_group_size_owns_fanout(self) -> None:
        from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b

        cfg = rl_grpo_qwen3_0_6b()
        assert cfg.group_size == 8
        assert not hasattr(cfg.generator.sampling, "n")

    def test_metrics_default_uses_factory(self) -> None:
        from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b

        cfg = rl_grpo_qwen3_0_6b()
        baseline = m.MetricsProcessor.Config()
        assert cfg.metrics.console_log_keys_train == baseline.console_log_keys_train
        assert (
            cfg.metrics.console_log_keys_validation
            == baseline.console_log_keys_validation
        )

    def test_metrics_defaults_are_independent_copies(self) -> None:
        from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b

        cfg = rl_grpo_qwen3_0_6b()
        cfg.metrics.console_log_keys_train.append("X")
        cfg.metrics.console_log_keys_validation.append("Y")
        fresh = rl_grpo_qwen3_0_6b()
        assert "X" not in fresh.metrics.console_log_keys_train
        assert "Y" not in fresh.metrics.console_log_keys_validation

    def test_metrics_default_wandb_disabled(self) -> None:
        from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b

        cfg = rl_grpo_qwen3_0_6b()
        assert cfg.metrics.enable_wandb is False
        assert cfg.metrics.enable_tensorboard is False


def _stub_trainer_for_reducers(dp_size: int):
    from torchtitan.experiments.rl.actors.trainer import PolicyTrainer

    inst = PolicyTrainer.__new__(PolicyTrainer)
    inst.dp_size = dp_size
    inst.device = torch.device("cpu")
    inst.parallel_dims = MagicMock()
    inst.parallel_dims.get_optional_mesh = MagicMock(return_value=None)
    return inst


class TestReducerFastPaths:
    def test_single_dp_identical(self) -> None:
        trainer = _stub_trainer_for_reducers(dp_size=1)
        out = trainer.reduce_forward_backward_metrics(
            sum_reduced_metrics={
                "loss/mean": torch.tensor(3.0),
                "bit_wise/logprob_diff/mean": torch.tensor(0.001),
                "bit_wise/ratio_tokens_different/mean": torch.tensor(0.0),
            },
            max_reduced_metrics={"bit_wise/logprob_diff/max": torch.tensor(0.005)},
        )
        assert out["loss/mean"] == pytest.approx(3.0)
        assert out["bit_wise/logprob_diff/mean"] == pytest.approx(0.001)
        assert out["bit_wise/logprob_diff/max"] == pytest.approx(0.005)
        assert out["bit_wise/ratio_tokens_different/mean"] == 0.0

    def test_unbiased_sum_reduction_across_ranks(self) -> None:
        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")

        rank0_share = torch.tensor([10.0 / 15.0], dtype=torch.float32)
        rank1_share = torch.tensor([30.0 / 15.0], dtype=torch.float32)

        def fake_all_reduce(t, *, reduceOp, group):
            if t.numel() == 1 and t.dtype == torch.float32:
                return rank0_share + rank1_share
            return t

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={"loss/mean": rank0_share[0]},
                max_reduced_metrics={"bit_wise/logprob_diff/max": torch.tensor(0.0)},
            )
        assert out["loss/mean"] == pytest.approx(40.0 / 15.0)

    def test_sum_and_max_reduce_paths(self) -> None:
        import torch.distributed.distributed_c10d as c10d

        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")
        rank1_max = torch.tensor([0.006], dtype=torch.float32)

        def fake_all_reduce(t, *, reduceOp, group):
            if reduceOp == c10d.ReduceOp.SUM.name:
                return t * 2
            if reduceOp == c10d.ReduceOp.MAX.name:
                return torch.maximum(t, rank1_max)
            raise AssertionError(f"unexpected reduceOp={reduceOp!r}")

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={"loss/mean": torch.tensor(0.5)},
                max_reduced_metrics={"bit_wise/logprob_diff/max": torch.tensor(0.003)},
            )

        assert out["loss/mean"] == pytest.approx(1.0)
        assert out["bit_wise/logprob_diff/max"] == pytest.approx(0.006)

    def test_sum_only_skips_max_collective(self) -> None:
        import torch.distributed.distributed_c10d as c10d

        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")
        seen_ops: list[str] = []

        def fake_all_reduce(t, *, reduceOp, group):
            seen_ops.append(reduceOp)
            if reduceOp == c10d.ReduceOp.SUM.name:
                return t * 2
            raise AssertionError(f"unexpected reduceOp={reduceOp!r}")

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={"loss/mean": torch.tensor(0.5)},
                max_reduced_metrics={},
            )
        assert seen_ops == [c10d.ReduceOp.SUM.name]
        assert out == {"loss/mean": pytest.approx(1.0)}

    def test_max_only_skips_sum_collective(self) -> None:
        import torch.distributed.distributed_c10d as c10d

        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")
        seen_ops: list[str] = []
        rank1_max = torch.tensor([0.006], dtype=torch.float32)

        def fake_all_reduce(t, *, reduceOp, group):
            seen_ops.append(reduceOp)
            if reduceOp == c10d.ReduceOp.MAX.name:
                return torch.maximum(t, rank1_max)
            raise AssertionError(f"unexpected reduceOp={reduceOp!r}")

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={},
                max_reduced_metrics={
                    "bit_wise/logprob_diff/max": torch.tensor(0.003),
                },
            )
        assert seen_ops == [c10d.ReduceOp.MAX.name]
        assert out == {"bit_wise/logprob_diff/max": pytest.approx(0.006)}

    def test_both_empty_returns_empty(self) -> None:
        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")
        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=AssertionError("should not be called"),
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={},
                max_reduced_metrics={},
            )
        assert out == {}
