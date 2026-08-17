# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.compile import apply_compile, CompileConfig
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import tag_sac_policy
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.models.common.moe import MoE
from torchtitan.observability import tensor_logging
from torchtitan.observability.tensor_logging.runtime import (
    _include_tensor_logging_calls_for_capture,
)


class _DeterministicRouter(nn.Module):
    def __init__(self, expert_count: int) -> None:
        super().__init__()
        self.expert_count = expert_count

    def forward(
        self,
        value: torch.Tensor,
        _expert_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = value[..., : self.expert_count].sigmoid()
        top_scores, top_indices = scores.topk(1, dim=-1)
        return top_scores, top_indices, scores


def test_forward_mutation_op_names_all_resolve() -> None:
    from torchtitan.distributed.activation_checkpoint import (
        _FORWARD_MUTATION_OP_NAMES,
        _registered_forward_mutation_ops,
    )

    assert len(_registered_forward_mutation_ops()) == len(_FORWARD_MUTATION_OP_NAMES)


class _TinyRoutedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        value: torch.Tensor,
        top_scores: torch.Tensor,
        _top_indices: torch.Tensor,
        _tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        return value * top_scores * self.scale


def _build_source_moe(expert_count: int = 4) -> MoE:
    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.expert_sequence_parallel_size = 1
    moe.routed_experts = _TinyRoutedExperts()
    moe.router = _DeterministicRouter(expert_count)
    moe.shared_experts = None
    moe.load_balance_coeff = 1e-3
    moe.register_buffer("expert_bias_E", torch.zeros(expert_count))
    moe.register_buffer(
        "tokens_per_expert_E",
        torch.zeros(expert_count, dtype=torch.int64),
        persistent=False,
    )
    moe.register_buffer(
        "_sequence_expert_counts_SE",
        torch.zeros(4, expert_count, dtype=torch.int64),
        persistent=False,
    )
    moe.register_buffer(
        "_recorded_sequence_count",
        torch.zeros(1, dtype=torch.int64),
        persistent=False,
    )
    tensor_logging.register_fwd_bwd(moe, ["input_normed", "routed_output"])
    return moe


class _TinyMoEBlock(nn.Module):
    def __init__(self, *, track_forward_calls: bool = True) -> None:
        super().__init__()
        self.moe_enabled = True
        self.moe = _build_source_moe()
        self.track_forward_calls = track_forward_calls
        self.forward_calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.track_forward_calls:
            self.forward_calls += 1
        return self.moe(value)


class _TinyMoERoot(nn.Module):
    def __init__(self, *, track_forward_calls: bool = True) -> None:
        super().__init__()
        self.layers = nn.ModuleDict(
            {"0": _TinyMoEBlock(track_forward_calls=track_forward_calls)}
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](value)


class _ReplacingSelectiveAC(SelectiveAC):
    def get_save_ops(self) -> set:
        return set()


def _run(policy) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    torch.manual_seed(0)
    root = _TinyMoERoot()
    block = root.layers["0"]
    if policy is not None:
        policy.build().apply(root)

    value = torch.randn(2, 3, 4, requires_grad=True)
    state = tensor_logging.init(root)
    try:
        with tensor_logging.set_enabled(True):
            root(value).sum().backward()
        return (
            block.moe.tokens_per_expert_E.clone(),
            block.moe._sequence_expert_counts_SE.clone(),
            value.grad.clone(),
            block.forward_calls,
        )
    finally:
        state.close()


def test_source_moe_expert_counts_are_exact_once_under_ac() -> None:
    eager, eager_sequences, eager_gradient, eager_calls = _run(None)
    full, full_sequences, full_gradient, full_calls = _run(FullAC.Config())
    selective, selective_sequences, selective_gradient, selective_calls = _run(
        SelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[])
    )
    (
        replacing_selective,
        replacing_sequences,
        replacing_gradient,
        replacing_calls,
    ) = _run(_ReplacingSelectiveAC.Config(force_recompute_mm_shapes_by_fqns=[]))

    assert eager_calls == 1
    assert full_calls == 2
    assert selective_calls == 2
    assert replacing_calls == 2
    assert eager.dtype == torch.int64
    assert eager_sequences.dtype == torch.int64
    assert eager.sum().item() == 6
    assert eager_sequences.sum().item() == 6
    torch.testing.assert_close(full, eager)
    torch.testing.assert_close(selective, eager)
    torch.testing.assert_close(replacing_selective, eager)
    torch.testing.assert_close(full_sequences, eager_sequences)
    torch.testing.assert_close(selective_sequences, eager_sequences)
    torch.testing.assert_close(replacing_sequences, eager_sequences)
    torch.testing.assert_close(full_gradient, eager_gradient)
    torch.testing.assert_close(selective_gradient, eager_gradient)
    torch.testing.assert_close(replacing_gradient, eager_gradient)


def test_source_moe_router_window_appends_only_selected_forwards() -> None:
    moe = _build_source_moe()
    forward_1 = torch.tensor(
        [[[4.0, 0.0, 0.0, 0.0]], [[0.0, 4.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    forward_2 = torch.tensor(
        [[[0.0, 0.0, 4.0, 0.0]], [[0.0, 0.0, 0.0, 4.0]]],
        requires_grad=True,
    )
    state = tensor_logging.init(moe)
    try:
        # Graph capture includes the call on every replay; the device flag decides
        # whether this optimizer step contributes to the metric window.
        with (
            _include_tensor_logging_calls_for_capture(),
            tensor_logging.set_enabled(False),
        ):
            moe(forward_1)
        with tensor_logging.set_enabled(True):
            moe(forward_1)
            moe(forward_2)

        torch.testing.assert_close(
            moe._sequence_expert_counts_SE,
            torch.eye(4, dtype=torch.int64),
        )
        assert moe._recorded_sequence_count.item() == 4
    finally:
        state.close()


def test_source_moe_expert_counts_ignore_eval_forwards() -> None:
    root = _TinyMoERoot(track_forward_calls=False)
    value = torch.tensor(
        [
            [[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]],
            [[0.0, 0.0, 10.0, 0.0], [0.0, 0.0, 0.0, 10.0]],
        ]
    )
    counts = root.layers["0"].moe.tokens_per_expert_E

    root.eval()
    root(value)
    torch.testing.assert_close(counts, torch.zeros(4, dtype=torch.int64))

    root.train()
    root(value)
    torch.testing.assert_close(counts, torch.ones(4, dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_source_moe_metric_mutations_compile_fullgraph_with_full_ac() -> None:
    root = _TinyMoERoot(track_forward_calls=False).cuda()
    FullAC.Config().build().apply(root)
    apply_compile(
        root,
        compile_config=CompileConfig(
            enable=True,
            components=["model"],
            backend="inductor",
        ),
        parallel_dims=ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        ),
    )
    value = torch.randn(2, 3, 4, device="cuda", requires_grad=True)
    state = tensor_logging.init(root, device=torch.device("cuda"))
    try:
        with tensor_logging.set_enabled(True):
            root(value).sum().backward()

        assert root.layers["0"].moe.tokens_per_expert_E.sum().item() == 6
        assert root.layers["0"].moe._sequence_expert_counts_SE.sum().item() == 6
    finally:
        state.close()
        torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_full_ac_preserves_expert_counts_without_tensor_logging() -> None:
    torch.manual_seed(0)
    eager_root = _TinyMoERoot(track_forward_calls=False).cuda()
    compiled_root = _TinyMoERoot(track_forward_calls=False).cuda()
    compiled_root.load_state_dict(eager_root.state_dict())
    value = torch.randn(2, 3, 4, device="cuda")

    eager_value = value.clone().requires_grad_()
    eager_root(eager_value).sum().backward()

    FullAC.Config().build().apply(compiled_root)
    apply_compile(
        compiled_root,
        compile_config=CompileConfig(
            enable=True,
            components=["model"],
            backend="inductor",
        ),
        parallel_dims=ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
        ),
    )
    compiled_value = value.clone().requires_grad_()
    try:
        compiled_root(compiled_value).sum().backward()

        torch.testing.assert_close(
            compiled_root.layers["0"].moe.tokens_per_expert_E,
            eager_root.layers["0"].moe.tokens_per_expert_E,
        )
        torch.testing.assert_close(compiled_value.grad, eager_value.grad)
    finally:
        torch.compiler.reset()


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
def test_source_moe_expert_counts_are_exact_under_graph_remat(device: str) -> None:
    torch.manual_seed(0)
    eager_root = _TinyMoERoot(track_forward_calls=False).to(device)
    graph_root = _TinyMoERoot(track_forward_calls=False).to(device)
    graph_root.load_state_dict(eager_root.state_dict())
    value = torch.randn(2, 3, 4, device=device, requires_grad=True)

    eager_state = tensor_logging.init(eager_root)
    try:
        eager_value = value.clone().requires_grad_()
        with tensor_logging.set_enabled(True):
            eager_output = eager_root(eager_value)
            eager_gradient = torch.autograd.grad(
                eager_output.sum(),
                tuple(eager_root.parameters()),
            )
        eager_counts = eager_root.layers["0"].moe.tokens_per_expert_E.clone()
        eager_sequence_counts = eager_root.layers[
            "0"
        ].moe._sequence_expert_counts_SE.clone()
        assert eager_sequence_counts.sum().item() == 6
    finally:
        eager_state.close()

    def forward_backward_step(input_value: torch.Tensor) -> list[torch.Tensor]:
        output = graph_root(input_value)
        loss = output.sum()
        gradients = torch.autograd.grad(loss, tuple(graph_root.parameters()))
        return [loss, *gradients]

    graph_state = tensor_logging.init(graph_root)
    try:
        with tensor_logging.set_enabled(True):
            traced = minimal_fx_tracer(
                forward_backward_step,
                module=graph_root,
            )(value)
            tag_sac_policy(
                traced.gm,
                policy_fn=lambda node: CheckpointPolicy.MUST_RECOMPUTE,
            )
            selective_activation_remat_pass(traced.gm)
            graph_result = run_traced(
                traced,
                module=graph_root,
                _validate_runtime=True,
            )(value)

        graph_counts = graph_root.layers["0"].moe.tokens_per_expert_E
        graph_sequence_counts = graph_root.layers["0"].moe._sequence_expert_counts_SE
        torch.testing.assert_close(graph_counts, eager_counts)
        torch.testing.assert_close(graph_sequence_counts, eager_sequence_counts)
        torch.testing.assert_close(graph_result[0], eager_output.sum())
        for actual, expected in zip(graph_result[1:], eager_gradient, strict=True):
            torch.testing.assert_close(actual, expected)
    finally:
        graph_state.close()
