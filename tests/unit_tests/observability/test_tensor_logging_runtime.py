# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, init_device_mesh, Replicate, Shard
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, CheckpointPolicy

from torchtitan.config import CompileConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.compile import apply_compile
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_pass
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import tag_sac_policy
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.observability.tensor_logging import (
    disable,
    init,
    log_fwd_bwd_stats,
    log_stats,
    register,
    register_fwd_bwd,
    set_enabled,
)


@pytest.fixture
def cpu_device_mesh(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'process_group'}",
        rank=0,
        world_size=1,
    )
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    finally:
        dist.destroy_process_group()


class TinyStatsModule(nn.Module):
    def __init__(self, width: int, *, track_forward_calls: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))
        self.track_forward_calls = track_forward_calls
        self.forward_calls = 0
        register(self, ["hidden"])
        register_fwd_bwd(self, ["output"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.track_forward_calls:
            self.forward_calls += 1
        hidden = torch.sin(value @ self.weight)
        log_stats(self, hidden=hidden)
        output = hidden.square()
        return log_fwd_bwd_stats(self, output=output)


class CompileStatsModule(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))
        register(self, ["hidden"])
        register_fwd_bwd(self, ["output"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = torch.sin(value @ self.weight)
        log_stats(self, hidden=hidden)
        output = hidden.square()
        return log_fwd_bwd_stats(self, output=output)


class CompileForwardStatsModule(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(width))
        register(self, ["hidden"])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = torch.sin(value @ self.weight)
        log_stats(self, hidden=hidden)
        return hidden.square()


class CompileForwardStatsRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": CompileForwardStatsModule(width=4)})

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](value)


class TinyStatsRoot(nn.Module):
    def __init__(self, *, track_forward_calls: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {
                "0": TinyStatsModule(
                    width=4,
                    track_forward_calls=track_forward_calls,
                )
            }
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](value)


class TinyRouterStatsModule(nn.Module):
    def __init__(self, expert_count: int, sequence_count: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(expert_count))
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(expert_count, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "last_per_sequence_counts",
            torch.zeros(sequence_count, expert_count, dtype=torch.int64),
            persistent=False,
        )
        register(
            self,
            [
                "expert_load",
                "experts_max_violation",
                "seq_expert_imbalance_mean",
                "seq_expert_imbalance_max",
            ],
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        logits = value @ self.weight
        assignments = F.one_hot(
            logits.argmax(dim=-1),
            num_classes=logits.shape[-1],
        )
        with torch.no_grad():
            self.tokens_per_expert.add_(assignments.sum(dim=(0, 1)))
            self.last_per_sequence_counts.copy_(assignments.sum(dim=1))
        return logits.square()


def _trace_forward_backward_step(
    module: nn.Module,
    value: torch.Tensor,
):
    def forward_backward_step(input_value: torch.Tensor) -> list[torch.Tensor]:
        output = module(input_value)
        loss = output.sum()
        gradients = torch.autograd.grad(loss, list(module.parameters()))
        return [loss, *gradients]

    with set_enabled(True):
        return minimal_fx_tracer(
            forward_backward_step,
            module=module,
        )(value)


def _rematerialize_every_forward_node(traced) -> None:
    tag_sac_policy(
        traced.gm,
        policy_fn=lambda node: CheckpointPolicy.MUST_RECOMPUTE,
    )
    selective_activation_remat_pass(traced.gm)


def _log_reconstructed_router_counts(
    router: TinyRouterStatsModule,
    reconstructed_counts: torch.Tensor,
    reconstructed_per_sequence_counts: torch.Tensor,
) -> None:
    average = reconstructed_counts.float().mean().clamp_min(1)
    expert_load = reconstructed_counts / average
    log_stats(
        router,
        expert_load=expert_load,
        experts_max_violation=(expert_load.max() - 1).view(1),
    )
    per_sequence_average = (
        reconstructed_per_sequence_counts.float().mean(dim=-1).clamp_min(1)
    )
    per_sequence_imbalance = (
        (reconstructed_per_sequence_counts / per_sequence_average.unsqueeze(-1))
        .max(dim=-1)
        .values
    )
    log_stats(
        router,
        seq_expert_imbalance_mean=per_sequence_imbalance.mean().view(1),
        seq_expert_imbalance_max=per_sequence_imbalance.max().view(1),
    )


def _run(*, activation_checkpoint: bool) -> tuple[dict, torch.Tensor, int]:
    torch.manual_seed(0)
    module = TinyStatsModule(width=4)
    value = torch.randn(3, 4, requires_grad=True)
    runtime = init(module)
    try:
        with set_enabled(True):
            if activation_checkpoint:
                output = checkpoint(
                    module,
                    value,
                    use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), disable()),
                )
            else:
                output = module(value)
            output.sum().backward()
        return runtime.raw_snapshot(), value.grad.clone(), module.forward_calls
    finally:
        runtime.close()


def _assert_snapshots_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key]["counts"], expected[key]["counts"])
        torch.testing.assert_close(actual[key]["sums"], expected[key]["sums"])
        torch.testing.assert_close(actual[key]["maximum"], expected[key]["maximum"])


def _run_torchtitan_ac(policy) -> tuple[dict, torch.Tensor, int]:
    torch.manual_seed(0)
    root = TinyStatsRoot()
    block = root.layers["0"]
    if policy is not None:
        policy.build().apply(root)

    value = torch.randn(3, 4, requires_grad=True)
    runtime = init(root)
    try:
        with set_enabled(True):
            root(value).sum().backward()
        return runtime.raw_snapshot(), value.grad.clone(), block.forward_calls
    finally:
        runtime.close()


def _run_two_live_graphs(policy) -> tuple[dict, tuple[torch.Tensor, ...], int]:
    torch.manual_seed(0)
    root = TinyStatsRoot()
    block = root.layers["0"]
    if policy is not None:
        policy.build().apply(root)

    values = tuple(torch.randn(3, 4, requires_grad=True) for _ in range(2))
    runtime = init(root)
    try:
        with set_enabled(True):
            outputs = tuple(root(value) for value in values)
            sum(output.sum() for output in outputs).backward()
        gradients = tuple(value.grad.clone() for value in values)
        return runtime.raw_snapshot(), gradients, block.forward_calls
    finally:
        runtime.close()


def _run_nested_ac(policy, *, outer_checkpoint: bool) -> tuple[dict, torch.Tensor]:
    torch.manual_seed(0)
    root = TinyStatsRoot()
    policy.build().apply(root)
    value = torch.randn(3, 4, requires_grad=True)
    runtime = init(root)
    try:
        with set_enabled(True):
            if outer_checkpoint:
                output = checkpoint(
                    root,
                    value,
                    use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), disable()),
                )
            else:
                output = root(value)
            output.sum().backward()
        return runtime.raw_snapshot(), value.grad.clone()
    finally:
        runtime.close()


def test_eager_records_forward_and_cotangent_once() -> None:
    snapshot, gradient, forward_calls = _run(activation_checkpoint=False)

    assert forward_calls == 1
    assert gradient.shape == (3, 4)
    assert snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
    assert snapshot["output.x"]["counts"].tolist() == [12, 0, 0, 1]
    assert snapshot["output.dx"]["counts"].tolist() == [12, 0, 0, 1]


def test_source_feed_forward_records_act_out_and_cotangent() -> None:
    module = FeedForward.Config(
        w1=Linear.Config(in_features=4, out_features=8),
        w2=Linear.Config(in_features=8, out_features=4),
        w3=Linear.Config(in_features=4, out_features=8),
    ).build()
    runtime = init(module)
    try:
        value = torch.randn(2, 3, 4, requires_grad=True)
        with set_enabled(True):
            module(value).sum().backward()

        snapshot = runtime.raw_snapshot()
        assert snapshot["act_out.x"]["counts"].tolist() == [48, 0, 0, 1]
        assert snapshot["act_out.dx"]["counts"].tolist() == [48, 0, 0, 1]
    finally:
        runtime.close()


def test_activation_checkpoint_recompute_is_disabled_exactly_once() -> None:
    eager, eager_gradient, eager_calls = _run(activation_checkpoint=False)
    checkpointed, checkpointed_gradient, checkpointed_calls = _run(
        activation_checkpoint=True
    )

    assert eager_calls == 1
    assert checkpointed_calls == 2
    _assert_snapshots_equal(checkpointed, eager)
    torch.testing.assert_close(checkpointed_gradient, eager_gradient)


def test_torchtitan_full_and_selective_ac_record_exactly_once() -> None:
    eager, eager_gradient, eager_calls = _run_torchtitan_ac(None)
    full, full_gradient, full_calls = _run_torchtitan_ac(FullAC.Config())
    selective, selective_gradient, selective_calls = _run_torchtitan_ac(
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[])
    )

    assert eager_calls == 1
    assert full_calls == 2
    assert selective_calls == 2
    _assert_snapshots_equal(full, eager)
    _assert_snapshots_equal(selective, eager)
    torch.testing.assert_close(full_gradient, eager_gradient)
    torch.testing.assert_close(selective_gradient, eager_gradient)


def test_repeated_checkpointed_module_with_two_live_graphs_is_exact() -> None:
    eager, eager_gradients, eager_calls = _run_two_live_graphs(None)
    full, full_gradients, full_calls = _run_two_live_graphs(FullAC.Config())
    selective, selective_gradients, selective_calls = _run_two_live_graphs(
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[])
    )

    assert eager_calls == 2
    assert full_calls == 4
    assert selective_calls == 4
    assert eager["layers.0.hidden"]["counts"].tolist() == [24, 0, 0, 2]
    assert eager["layers.0.output.x"]["counts"].tolist() == [24, 0, 0, 2]
    assert eager["layers.0.output.dx"]["counts"].tolist() == [24, 0, 0, 2]
    _assert_snapshots_equal(full, eager)
    _assert_snapshots_equal(selective, eager)
    for actual, expected in zip(full_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(selective_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "policy",
    [
        FullAC.Config(),
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[]),
    ],
    ids=["full", "selective"],
)
def test_nested_checkpoint_regions_record_exactly_once(policy) -> None:
    expected, expected_gradient = _run_nested_ac(
        policy,
        outer_checkpoint=False,
    )
    nested, nested_gradient = _run_nested_ac(
        policy,
        outer_checkpoint=True,
    )

    assert expected["layers.0.hidden"]["counts"].tolist() == [12, 0, 0, 1]
    assert expected["layers.0.output.x"]["counts"].tolist() == [12, 0, 0, 1]
    assert expected["layers.0.output.dx"]["counts"].tolist() == [12, 0, 0, 1]
    _assert_snapshots_equal(nested, expected)
    torch.testing.assert_close(nested_gradient, expected_gradient)


def test_disabled_scope_is_a_noop_and_restores_outer_scope() -> None:
    module = TinyStatsModule(width=2)
    runtime = init(module)
    try:
        value = torch.ones(1, 2, requires_grad=True)
        with set_enabled(True):
            with disable():
                module(value).sum().backward()
            module(value).sum().backward()

        snapshot = runtime.raw_snapshot()
        assert snapshot["hidden"]["counts"].tolist() == [2, 0, 0, 1]
        assert snapshot["output.x"]["counts"].tolist() == [2, 0, 0, 1]
        assert snapshot["output.dx"]["counts"].tolist() == [2, 0, 0, 1]
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_finite_statistics_match_on_cpu_and_cuda(device: str) -> None:
    owner = nn.Module().to(device)
    register(owner, ["value"])
    runtime = init(owner, device=torch.device(device))
    try:
        value = torch.tensor(
            [1.0, -2.0, 0.0, torch.nan, torch.inf, -torch.inf],
            device=device,
        )
        with set_enabled(True):
            log_stats(owner, value=value)

        snapshot = runtime.raw_snapshot()["value"]
        assert snapshot["counts"].tolist() == [6, 3, 1, 1]
        assert snapshot["sums"].tolist() == [3.0, 5.0, 17.0]
        assert snapshot["maximum"].item() == 2.0
    finally:
        runtime.close()


def test_whole_gradient_health_reuses_parameter_sufficient_statistics() -> None:
    root = nn.Module()
    root.dense = nn.Linear(2, 1, bias=False)
    root.moe = nn.Linear(2, 1, bias=False)
    register(root.dense.weight, ["dw"])
    register(root.moe.weight, ["dw"])
    runtime = init(root)
    try:
        with set_enabled(True):
            log_stats(root.dense.weight, dw=torch.tensor([[1.0, -2.0]]))
            log_stats(root.moe.weight, dw=torch.tensor([[0.0, 3.0]]))

        metrics = runtime.buffers_to_metrics(runtime.reduce_buffers())
        assert metrics["gradients.all.numel"] == 4
        assert metrics["gradients.all.observation_count"] == 2
        assert metrics["gradients.all.zero_count"] == 1
        assert metrics["gradients.all.abs_sum"] == 6.0
        assert metrics["gradients.all.abs_max"] == 3.0
        assert metrics["gradients.moe.numel"] == 2
        assert metrics["gradients.moe.abs_sum"] == 3.0
    finally:
        runtime.close()


def test_metrics_filter_matches_name_and_statistic() -> None:
    owner = nn.Module()
    register(owner, ["hidden", "other"])
    runtime = init(
        owner,
        metrics_filter_regex=r"^hidden:(?:abs_mean|abs_max)$",
    )
    try:
        with set_enabled(True):
            log_stats(
                owner,
                hidden=torch.tensor([1.0, -3.0]),
                other=torch.tensor([5.0]),
            )

        metrics = runtime.buffers_to_metrics(runtime.reduce_buffers())
        assert metrics == {
            "hidden.abs_mean": 2.0,
            "hidden.abs_max": 3.0,
        }
    finally:
        runtime.close()


def test_registered_but_unobserved_key_is_not_published() -> None:
    owner = nn.Module()
    register(owner, ["observed", "missing"])
    runtime = init(owner)
    try:
        with set_enabled(True):
            log_stats(owner, observed=torch.tensor([2.0]))

        metrics = runtime.buffers_to_metrics(runtime.reduce_buffers())
        assert metrics["observed.abs_mean"] == 2.0
        assert not any(key.startswith("missing.") for key in metrics)
    finally:
        runtime.close()


@pytest.mark.parametrize("placement", [Replicate(), Shard(0)])
def test_dtensor_records_local_forward_and_cotangent(
    cpu_device_mesh,
    placement,
) -> None:
    owner = nn.Module()
    register_fwd_bwd(owner, ["value"])
    runtime = init(owner)
    try:
        local_value = torch.tensor(
            [[1.0, -2.0], [0.0, 4.0]],
            requires_grad=True,
        )
        value = DTensor.from_local(
            local_value,
            cpu_device_mesh,
            (placement,),
            run_check=False,
        )
        with set_enabled(True):
            value = log_fwd_bwd_stats(owner, value=value)
            (value * 2).to_local().sum().backward()

        snapshot = runtime.raw_snapshot()
        assert snapshot["value.x"]["counts"].tolist() == [4, 0, 1, 1]
        assert snapshot["value.dx"]["counts"].tolist() == [4, 0, 0, 1]
        assert snapshot["value.x"]["sums"].tolist() == [7.0, 21.0, 273.0]
        assert snapshot["value.dx"]["sums"].tolist() == [8.0, 16.0, 64.0]
        assert snapshot["value.x"]["maximum"].item() == 4.0
        assert snapshot["value.dx"]["maximum"].item() == 2.0
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_graph_trainer_trace_remat_replay_and_cadence_are_exact(
    device: str,
) -> None:
    torch.manual_seed(0)
    module = TinyStatsModule(width=4, track_forward_calls=False).to(device)
    value = torch.randn(3, 4, device=device)
    runtime = init(module)
    buffer_addresses = tuple(buffer.data_ptr() for buffer in runtime.buffers.buffers())

    try:
        with set_enabled(True):
            eager_output = module(value)
            eager_loss = eager_output.sum()
            eager_gradients = torch.autograd.grad(
                eager_loss,
                tuple(module.parameters()),
            )
        eager_snapshot = runtime.raw_snapshot()
        runtime.clear()

        traced = _trace_forward_backward_step(module, value)

        metric_op = torch.ops.torchtitan.accumulate_tensor_statistics.default
        assert sum(node.target is metric_op for node in traced.gm.graph.nodes) == 3
        assert all(
            count == 0
            for statistic in runtime.raw_snapshot().values()
            for count in statistic["counts"].tolist()
        )

        _rematerialize_every_forward_node(traced)
        assert sum(node.target is metric_op for node in traced.gm.graph.nodes) == 3

        runner = run_traced(traced, module=module, _validate_runtime=True)
        with set_enabled(True):
            graph_result = runner(value)
        enabled_snapshot = runtime.raw_snapshot()
        _assert_snapshots_equal(enabled_snapshot, eager_snapshot)
        torch.testing.assert_close(graph_result[0], eager_loss)
        for actual, expected in zip(
            graph_result[1:],
            eager_gradients,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)
        assert enabled_snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert enabled_snapshot["output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert enabled_snapshot["output.dx"]["counts"].tolist() == [12, 0, 0, 1]

        runtime.clear()
        with set_enabled(False):
            runner(value)
        disabled_snapshot = runtime.raw_snapshot()
        assert all(
            count == 0
            for statistic in disabled_snapshot.values()
            for count in statistic["counts"].tolist()
        )

        with set_enabled(True):
            runner(value)
            runner(value)
        replay_snapshot = runtime.raw_snapshot()
        assert replay_snapshot["hidden"]["counts"].tolist() == [24, 0, 0, 2]
        assert replay_snapshot["output.x"]["counts"].tolist() == [24, 0, 0, 2]
        assert replay_snapshot["output.dx"]["counts"].tolist() == [24, 0, 0, 2]
        assert tuple(buffer.data_ptr() for buffer in runtime.buffers.buffers()) == (
            buffer_addresses
        )
        assert all("_tensor_logging_state" not in key for key in module.state_dict())
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is unavailable",
            ),
        ),
    ],
)
def test_graph_trainer_router_count_is_not_rematerialized(
    device: str,
) -> None:
    router = TinyRouterStatsModule(expert_count=4, sequence_count=1).to(device)
    values = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 0.0, 0.0],
                [0.0, 0.0, 8.0, 0.0],
                [0.0, 0.0, 0.0, 7.0],
                [6.0, 0.0, 0.0, 0.0],
            ]
        ],
        device=device,
    )
    runtime = init(router)
    try:
        traced = _trace_forward_backward_step(router, values)
        _rematerialize_every_forward_node(traced)
        run_traced(traced, module=router, _validate_runtime=True)(values)

        expected_counts = torch.tensor([2, 1, 1, 1], device=device)
        torch.testing.assert_close(router.tokens_per_expert, expected_counts)
        torch.testing.assert_close(
            router.last_per_sequence_counts,
            expected_counts.view(1, -1),
        )

        with set_enabled(True):
            _log_reconstructed_router_counts(
                router,
                router.tokens_per_expert,
                router.last_per_sequence_counts,
            )

        snapshot = runtime.raw_snapshot()
        assert snapshot["expert_load"]["counts"].tolist() == [4, 0, 0, 1]
        torch.testing.assert_close(
            snapshot["expert_load"]["sums"][0],
            torch.tensor(4.0),
        )
        torch.testing.assert_close(
            snapshot["expert_load"]["maximum"],
            torch.tensor(1.6),
        )
        assert snapshot["experts_max_violation"]["counts"].tolist() == [1, 0, 0, 1]
        torch.testing.assert_close(
            snapshot["experts_max_violation"]["sums"][0],
            torch.tensor(0.6),
        )
        assert snapshot["seq_expert_imbalance_mean"]["counts"].tolist() == [
            1,
            0,
            0,
            1,
        ]
        torch.testing.assert_close(
            snapshot["seq_expert_imbalance_mean"]["sums"][0],
            torch.tensor(1.6),
        )

        reduced_buffers = runtime.reduce_buffers()
        metrics = runtime.buffers_to_metrics(reduced_buffers)
        assert metrics["expert_load.numel"] == 4
        assert metrics["expert_load.observation_count"] == 1
        assert metrics["expert_load.abs_mean"] == pytest.approx(1.0)
        assert metrics["expert_load.abs_max"] == pytest.approx(1.6)
        assert metrics["experts_max_violation.abs_mean"] == pytest.approx(0.6)
        assert metrics["seq_expert_imbalance_mean.abs_mean"] == pytest.approx(1.6)

        runtime.clear()
        assert all(
            count == 0
            for statistic in runtime.raw_snapshot().values()
            for count in statistic["counts"].tolist()
        )
    finally:
        runtime.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_graph_trainer_cudagraph_replay_obeys_device_cadence() -> None:
    torch.manual_seed(0)
    module = TinyStatsModule(width=4, track_forward_calls=False).cuda()
    value = torch.randn(3, 4, device="cuda")
    runtime = init(module)
    try:
        traced = _trace_forward_backward_step(module, value)
        _rematerialize_every_forward_node(traced)
        traced.gm = cudagraph_pass(traced.gm, traced.example_inputs)
        runner = run_traced(traced, module=module, _validate_runtime=True)

        with set_enabled(True):
            runner(value)  # warmup
            runner(value)  # capture
        captured_snapshot = runtime.raw_snapshot()
        assert captured_snapshot["hidden"]["counts"].tolist() == [24, 0, 0, 2]
        assert captured_snapshot["output.x"]["counts"].tolist() == [24, 0, 0, 2]
        assert captured_snapshot["output.dx"]["counts"].tolist() == [24, 0, 0, 2]

        runtime.clear()
        with set_enabled(False):
            runner(value)
        assert all(
            count == 0
            for statistic in runtime.raw_snapshot().values()
            for count in statistic["counts"].tolist()
        )

        with set_enabled(True):
            runner(value)
        replay_snapshot = runtime.raw_snapshot()
        assert replay_snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert replay_snapshot["output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert replay_snapshot["output.dx"]["counts"].tolist() == [12, 0, 0, 1]
    finally:
        runtime.close()


def test_compile_fullgraph_records_forward_statistics() -> None:
    if not torch.cuda.is_available():
        return

    module = CompileForwardStatsModule(width=4).cuda()
    runtime = init(module)
    try:
        compiled = torch.compile(module, fullgraph=True)
        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        with set_enabled(True):
            compiled(value).sum().backward()

        snapshot = runtime.raw_snapshot()
        assert snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
    finally:
        runtime.close()
        torch.compiler.reset()


def test_compile_fullgraph_forward_cadence_has_two_stable_graphs() -> None:
    if not torch.cuda.is_available():
        return

    compiled_graphs: list[torch.fx.GraphModule] = []

    def record_graph(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    module = CompileForwardStatsModule(width=4).cuda()
    runtime = init(module)
    try:
        compiled = torch.compile(module, backend=record_graph, fullgraph=True)
        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        for enabled in (True, True, False, False, True):
            with set_enabled(enabled):
                compiled(value).sum().backward()

        assert len(compiled_graphs) == 2
        snapshot = runtime.raw_snapshot()
        assert snapshot["hidden"]["counts"].tolist() == [36, 0, 0, 3]
    finally:
        runtime.close()
        torch.compiler.reset()


@pytest.mark.parametrize(
    "policy",
    [
        FullAC.Config(),
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[]),
    ],
    ids=["full", "selective"],
)
def test_compile_fullgraph_with_ac_records_forward_exactly_once(policy) -> None:
    if not torch.cuda.is_available():
        return

    root = CompileForwardStatsRoot().cuda()
    policy.build().apply(root)
    apply_compile(
        root,
        CompileConfig(
            enable=True,
            components=["model"],
            backend="aot_eager",
        ),
    )
    runtime = init(root)
    try:
        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        with set_enabled(True):
            root(value).sum().backward()

        snapshot = runtime.raw_snapshot()
        assert snapshot["layers.0.hidden"]["counts"].tolist() == [12, 0, 0, 1]
    finally:
        runtime.close()
        torch.compiler.reset()


def test_compile_fullgraph_records_forward_and_cotangent() -> None:
    if not torch.cuda.is_available():
        return

    module = CompileStatsModule(width=4).cuda()
    runtime = init(module)
    try:
        compiled = torch.compile(module, fullgraph=True)
        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        with set_enabled(True):
            compiled(value).sum().backward()

        snapshot = runtime.raw_snapshot()
        assert snapshot["hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["output.dx"]["counts"].tolist() == [12, 0, 0, 1]
    finally:
        runtime.close()
        torch.compiler.reset()


@pytest.mark.parametrize(
    "policy",
    [
        FullAC.Config(),
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[]),
    ],
    ids=["full", "selective"],
)
def test_compile_fullgraph_with_ac_records_exactly_once(policy) -> None:
    if not torch.cuda.is_available():
        return

    root = TinyStatsRoot(track_forward_calls=False).cuda()
    policy.build().apply(root)
    apply_compile(root, CompileConfig(enable=True, components=["model"]))
    runtime = init(root)
    try:
        value = torch.randn(3, 4, device="cuda", requires_grad=True)
        with set_enabled(True):
            root(value).sum().backward()

        snapshot = runtime.raw_snapshot()
        assert snapshot["layers.0.hidden"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["layers.0.output.x"]["counts"].tolist() == [12, 0, 0, 1]
        assert snapshot["layers.0.output.dx"]["counts"].tolist() == [12, 0, 0, 1]
    finally:
        runtime.close()
        torch.compiler.reset()
