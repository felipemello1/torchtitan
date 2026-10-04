# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import warnings
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.pipelining._backward import (
    stage_backward_input,
    stage_backward_weight,
)
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.components.loss import (
    ChunkedLossWrapper,
    cross_entropy_loss,
    CrossEntropyLoss,
)
from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    run_eager_on_cuda_graph_stream,
    wrap_with_cuda_graph,
)
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common import linear as linear_module
from torchtitan.models.common.decoder_sharding import set_decoder_sharding_config
from torchtitan.models.common.linear import FP32OutputLinear
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.observability.sdc_replayer import SDCReplayer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("grad_output_pieces", [2, 3])
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_fp32_output_linear_compiles(input_dtype, weight_dtype, grad_output_pieces):
    layer = FP32OutputLinear.Config(
        in_features=128,
        out_features=16,
        bias=True,
        grad_output_pieces=grad_output_pieces,
    ).build()
    layer = layer.to(device="cuda", dtype=weight_dtype)
    compiled = torch.compile(layer, fullgraph=True)
    input_TD = torch.randn(
        32, 128, device="cuda", dtype=input_dtype, requires_grad=True
    )

    output_TE = compiled(input_TD)
    output_TE.sum().backward()

    assert output_TE.dtype is torch.float32
    assert input_TD.grad is not None
    assert input_TD.grad.dtype is input_dtype
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype is weight_dtype


def _lm_head(in_features: int = 256, out_features: int = 1024) -> FP32OutputLinear:
    lm_head = FP32OutputLinear.Config(
        in_features=in_features, out_features=out_features
    ).build()
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    return lm_head.to(device="cuda", dtype=torch.bfloat16)


def test_forward_matches_fp64_reference():
    lm_head = _lm_head()
    x = torch.randn(3, 5, 256, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(x.double(), lm_head.weight.double())

    out = lm_head(x)

    assert out.dtype == torch.float32
    assert out.shape == reference.shape
    torch.testing.assert_close(out.double(), reference, rtol=1e-4, atol=1e-4)


def _relative_error(actual, exact):
    return ((actual.double() - exact).norm() / exact.norm()).item()


def _backward_errors_vs_bf16_floor(function, num_tokens, in_features, out_features):
    """Backward of ``function`` vs exact fp64 gradients, as multiples of the bf16 floor."""
    x = torch.randn(num_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(out_features, in_features, device="cuda") * 0.02).bfloat16()
    grad_output = torch.randn(num_tokens, out_features, device="cuda")

    x_exact = x.double().requires_grad_()
    weight_exact = weight.double().requires_grad_()
    F.linear(x_exact, weight_exact).backward(grad_output.double())

    x_leaf = x.clone().requires_grad_()
    weight_leaf = weight.clone().requires_grad_()
    function(x_leaf, weight_leaf).backward(grad_output)

    # Autograd rounds both gradients to bf16 (default grad_dtype), so the bf16-rounded exact ones are
    # the floor.
    ratios = []
    for grad, exact in (
        (x_leaf.grad, x_exact.grad),
        (weight_leaf.grad, weight_exact.grad),
    ):
        assert grad.dtype == torch.bfloat16
        ratios.append(
            _relative_error(grad, exact) / _relative_error(exact.bfloat16(), exact)
        )
    return ratios


# out_features > num_tokens takes the LM-head layout, out_features < num_tokens the router one.
@pytest.mark.parametrize("grad_output_pieces", [2, 3])
@pytest.mark.parametrize(
    "num_tokens,in_features,out_features", [(64, 256, 1024), (512, 256, 16)]
)
def test_backward_error_stays_at_bf16_rounding_floor(
    num_tokens, in_features, out_features, grad_output_pieces
):
    ratios = _backward_errors_vs_bf16_floor(
        lambda input, weight: linear_module._FP32OutputLinearFunction.apply(
            input, weight, grad_output_pieces
        ),
        num_tokens,
        in_features,
        out_features,
    )
    assert max(ratios) < 1.05, ratios


def test_compiled_backward_keeps_lo_half():
    # torch.compile folds a bf16 round trip away inside fused kernels; if the split used one,
    # lo would compile to zero and the error would rise to that of a bf16 grad_output.
    # Compile a wrapper: compiling any ``Function.apply`` directly breaks later compiles of other
    # autograd Functions in the same process (test_qwen3_5_deltanet fails after it).
    def linear(input, weight):
        return linear_module._FP32OutputLinearFunction.apply(input, weight, 2)

    # fullgraph=True: the LM-head backward must trace without a break (its split guard).
    ratios = _backward_errors_vs_bf16_floor(
        torch.compile(linear, fullgraph=True), 64, 256, 1024
    )
    assert max(ratios) < 1.05, ratios


@pytest.mark.parametrize("compile", [False, True])
# 1 token takes the LM-head layout, 4 tokens the router one.
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("grad_output_pieces", [2, 3])
def test_third_piece_keeps_what_two_pieces_drop(
    grad_output_pieces, num_tokens, compile
):
    # (1 + 2^-8 + 2^-20) - (1 + 2^-8) = 2^-20: 2 pieces round the 2^-20 away, 3 keep it. Compiled,
    # this also catches a split that inductor folds (the third piece would compile to zero).
    grad_output = torch.tensor([[1 + 2**-8 + 2**-20, 1 + 2**-8]], device="cuda")
    grad_output = grad_output.repeat(num_tokens, 1)
    weight = torch.tensor([[1.0] * 8, [-1.0] * 8], device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(
        num_tokens, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def linear(input, weight):
        return linear_module._FP32OutputLinearFunction.apply(
            input, weight, grad_output_pieces
        )

    (torch.compile(linear) if compile else linear)(x, weight).backward(grad_output)

    expected = 2**-20 if grad_output_pieces == 3 else 0.0
    assert torch.equal(x.grad, torch.full_like(x.grad, expected))


@pytest.mark.parametrize("grad_output_pieces", [2, 3])
def test_compiled_split_matches_eager_split(grad_output_pieces):
    # The LM-head layout's split is compiled. It must give the eager pieces bit for bit, including
    # signed zeros, ties (1 + 2^-8 sits halfway between two bf16s) and tiny and huge values.
    # Earlier tests fill the split's Dynamo cache; past the recompile limit it would run eagerly.
    torch._dynamo.reset()
    torch.manual_seed(0)
    grad_output = torch.randn(64, 1024, device="cuda")
    grad_output *= torch.logspace(-30, 30, 1024, device="cuda")
    grad_output[0, :4] = torch.tensor([0.0, -0.0, 1 + 2**-8, -(1 + 2**-8)])
    pieces = linear_module._split_into_bf16_pieces(
        grad_output, grad_output_pieces=grad_output_pieces
    )

    stacked = linear_module._compiled_split_into_stacked_bf16_pieces(
        grad_output, grad_output_pieces=grad_output_pieces
    )

    assert torch.equal(stacked.view(torch.int16), torch.cat(pieces).view(torch.int16))


def test_compiled_split_runs_eagerly_past_the_recompile_limit():
    # Each call below needs a new graph (a new piece count, or a token count without the backward's
    # mark). Past Dynamo's recompile limit it must run eagerly; with fullgraph=True it would raise
    # inside backward.
    torch.manual_seed(0)
    with torch._dynamo.config.patch(recompile_limit=1):
        for num_tokens, grad_output_pieces in ((64, 2), (96, 2), (64, 3), (1, 3)):
            grad_output = torch.randn(num_tokens, 1024, device="cuda")
            pieces = linear_module._split_into_bf16_pieces(
                grad_output, grad_output_pieces=grad_output_pieces
            )

            stacked = linear_module._compiled_split_into_stacked_bf16_pieces(
                grad_output, grad_output_pieces=grad_output_pieces
            )

            assert torch.equal(
                stacked.view(torch.int16), torch.cat(pieces).view(torch.int16)
            )
    # Past the limit, Dynamo never compiles the split again in this process.
    torch._dynamo.reset()


def test_lm_head_backward_compiles_once_for_all_token_counts():
    # Chunk token counts change between steps (RL). The eager backward must run the compiled split,
    # with one graph for all counts.
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    weight = torch.randn(
        1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    for num_tokens in (64, 65, 100, 128, 200, 333, 500, 512, 700, 999):
        x = torch.randn(
            num_tokens, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        _lm_head_forward_backward(
            x, weight, torch.randn(num_tokens, 1024, device="cuda")
        )

    assert counters["stats"]["unique_graphs"] == 1


def _lm_head_forward_backward(x, weight, grad_output):
    output = linear_module._FP32OutputLinearFunction.apply(x, weight, 2)
    return torch.autograd.grad(output, (x, weight), grad_output)


@pytest.mark.parametrize("tracing_mode", ["real", "fake", "symbolic"])
def test_lm_head_backward_traces_with_make_fx(tracing_mode):
    # make_fx can't trace into the compiled split: fake and symbolic tracing would fail, and real
    # tracing would bake the split's result into the graph.
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    traced = make_fx(_lm_head_forward_backward, tracing_mode=tracing_mode)(
        x, weight, torch.randn(64, 1024, device="cuda")
    )

    grad_output = torch.randn(64, 1024, device="cuda")
    for actual, expected in zip(
        traced(x, weight, grad_output),
        _lm_head_forward_backward(x, weight, grad_output),
    ):
        assert torch.equal(actual, expected)


def test_lm_head_backward_runs_under_fake_tensor_mode():
    # Memory estimators run forward + backward under a bare FakeTensorMode, where the compiled
    # split's kernel would read fake data pointers.
    with FakeTensorMode():
        x = torch.randn(
            64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(
            1024, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        grad_input, grad_weight = _lm_head_forward_backward(
            x, weight, torch.randn(64, 1024, device="cuda")
        )
    # Surface an illegal memory access in this test, not a later one.
    torch.cuda.synchronize()

    assert grad_input.shape == x.shape
    assert grad_weight.shape == weight.shape


# out_features > num_tokens takes the LM-head layout, out_features < num_tokens the router one.
@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("grad_output_pieces", [2, 3])
@pytest.mark.parametrize("num_tokens,out_features", [(64, 1024), (512, 16)])
def test_weight_grad_stays_fp32_when_grad_dtype_is_fp32(
    num_tokens, out_features, grad_output_pieces, batch_invariant, monkeypatch
):
    # grad_dtype = fp32 stands in for FSDP (https://github.com/pytorch/pytorch/pull/194434): the
    # fp32 grad_weight skips the bf16 rounding. Batch-invariant mode (RL) takes the fp32 fallback.
    monkeypatch.setattr(
        linear_module, "is_in_batch_invariant_mode", lambda: batch_invariant
    )
    x = torch.randn(num_tokens, 256, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(out_features, 256, device="cuda") * 0.02).bfloat16()
    weight.requires_grad_()
    weight.grad_dtype = torch.float32
    grad_output = torch.randn(num_tokens, out_features, device="cuda")

    linear_module._FP32OutputLinearFunction.apply(
        x, weight, grad_output_pieces
    ).backward(grad_output)

    exact = grad_output.double().T @ x.double()
    assert weight.grad.dtype == torch.float32
    floor = _relative_error(exact.bfloat16(), exact)
    assert _relative_error(weight.grad, exact) < 0.01 * floor


def _num_addmm_calls(fn) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities) as prof:
        fn()
    events = prof.key_averages()
    return sum(event.count for event in events if event.key == "aten::addmm")


def _chunks(num_chunks: int = 3):
    """LM-head-shaped (num_tokens < out_features) inputs and grad_outputs, one per chunk."""
    inputs = [torch.randn(64, 256, device="cuda").bfloat16() for _ in range(num_chunks)]
    grad_outputs = [torch.randn(64, 1024, device="cuda") for _ in range(num_chunks)]
    return inputs, grad_outputs


def _run_chunks(lm_head, inputs, grad_outputs, *, accumulate: bool):
    """One forward and backward per chunk, as ChunkedLossWrapper runs them."""
    for input_TD, grad_output_TO in zip(inputs, grad_outputs):
        with (
            linear_module.accumulate_into_weight_grad()
            if accumulate
            else contextlib.nullcontext()
        ):
            output_TO = lm_head(input_TD)
        output_TO.backward(grad_output_TO)


@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
def test_chunked_loss_adds_into_weight_grad_inside_the_gemm(grad_dtype):
    # ChunkedLossWrapper runs one backward per chunk under accumulate_into_weight_grad: each chunk
    # after the first adds into lm_head.weight.grad with addmm(out=), not in a separate kernel.
    torch.manual_seed(0)
    lm_head = _lm_head()
    lm_head.weight.grad_dtype = grad_dtype
    chunked_loss = ChunkedLossWrapper.Config(num_chunks=3).build()
    chunked_loss.set_lm_head(lm_head)
    hidden = torch.randn(192, 256, device="cuda").bfloat16()
    labels = torch.randint(0, 1024, (192,), device="cuda")

    def separate_adds():
        for x, chunk_labels in zip(hidden.chunk(3), labels.chunk(3)):
            F.cross_entropy(lm_head(x), chunk_labels, reduction="sum").backward()

    def chunked_loss_step():
        loss, _ = chunked_loss(hidden.clone().requires_grad_(), labels)
        loss.backward()

    # Called directly, outside the context, autograd adds each chunk's grad_weight.
    assert _num_addmm_calls(separate_adds) == 0
    expected = lm_head.weight.grad
    lm_head.weight.grad = None
    assert _num_addmm_calls(chunked_loss_step) == 2
    torch.testing.assert_close(lm_head.weight.grad, expected)


def test_multi_output_chunked_loss_adds_into_weight_grad(monkeypatch):
    # MTP: two lm_head calls per chunk share one backward, and each adds into .grad. Correct, but
    # summed in a different order than autograd's, so not bitwise the same.
    torch.manual_seed(0)
    lm_head = _lm_head()
    lm_head.weight.grad_dtype = torch.float32
    chunked_loss = ChunkedLossWrapper.Config(
        num_chunks=3, loss_fn=MTPLoss.Config()
    ).build()
    chunked_loss.set_lm_head(lm_head)
    hidden = tuple(torch.randn(192, 256, device="cuda").bfloat16() for _ in range(2))
    labels = tuple(torch.randint(0, 1024, (192,), device="cuda") for _ in range(2))

    def step():
        pred = tuple(hidden_state.clone().requires_grad_() for hidden_state in hidden)
        loss, _ = chunked_loss(pred, labels)
        loss.backward()

    with monkeypatch.context() as patch:
        patch.setattr(
            linear_module, "accumulate_into_weight_grad", contextlib.nullcontext
        )
        assert _num_addmm_calls(step) == 0
    separate_adds = lm_head.weight.grad
    lm_head.weight.grad = None
    assert _num_addmm_calls(step) == 4
    torch.testing.assert_close(lm_head.weight.grad, separate_adds)


def test_input_only_backward_keeps_weight_grad():
    # Zero-bubble PP's input pass is autograd.grad wrt the stage input: it runs this backward but
    # drops grad_weight. A fused add would still change weight.grad, so it's off by default.
    lm_head = _lm_head()
    lm_head.weight.grad = torch.ones_like(lm_head.weight)
    (input_TD,), (grad_output_TO,) = _chunks(1)
    input_TD.requires_grad_()

    torch.autograd.grad(lm_head(input_TD), input_TD, grad_output_TO)

    assert torch.equal(lm_head.weight.grad, torch.ones_like(lm_head.weight))


def test_zero_bubble_weight_pass_raises_in_the_context():
    # Misuse is loud: zero-bubble PP's weight pass is autograd.grad wrt the weight, and the fused
    # backward returns no grad_weight, so autograd reports the weight as unused.
    lm_head = _lm_head()
    lm_head.weight.grad = torch.zeros_like(lm_head.weight)
    (input_TD,), (grad_output_TO,) = _chunks(1)
    input_TD.requires_grad_()
    with linear_module.accumulate_into_weight_grad():
        loss = (lm_head(input_TD) * grad_output_TO).sum()

    _, param_groups = stage_backward_input(
        [loss], None, [input_TD], lm_head.parameters()
    )
    with pytest.raises(RuntimeError, match="not have been used"):
        stage_backward_weight(lm_head.parameters(), param_groups)


def test_stacked_weight_leaves_weight_grad_to_autograd():
    # num_linears > 1 hands the Function a flattened view of the weight: a non-leaf, whose .grad
    # autograd never fills. Reading it would warn on every chunk.
    lm_head = FP32OutputLinear.Config(
        in_features=256, out_features=512, num_linears=2
    ).build()
    lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
    lm_head.weight.grad = torch.zeros_like(lm_head.weight)
    inputs, _ = _chunks()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not a leaf Tensor.*")
        for input_TD in inputs:
            with linear_module.accumulate_into_weight_grad():
                output = lm_head(input_TD)
            output.sum().backward()


@pytest.mark.parametrize("tracer", ["compile", "make_fx"])
def test_traced_backward_leaves_weight_grad_to_autograd(tracer):
    # A traced backward can't write into weight.grad, so it returns grad_weight and autograd adds
    # it. Dynamo sets is_compiling; graph_trainer's make_fx tracer only has a proxy mode.
    torch.manual_seed(0)
    lm_head = _lm_head()
    inputs, grad_outputs = _chunks()
    if tracer == "compile":
        _run_chunks(lm_head, inputs, grad_outputs, accumulate=False)
        separate_adds = lm_head.weight.grad
        # Trace while weight.grad exists, so a traced backward would see it.
        lm_head.weight.grad = torch.zeros_like(lm_head.weight)
        # Other tests fill Dynamo's recompile cache for this class; a full one silently runs eager.
        torch._dynamo.reset()
        compiled = torch.compile(lm_head, fullgraph=True)
        _run_chunks(compiled, inputs[:1], grad_outputs[:1], accumulate=True)
        num_addmm_calls = _num_addmm_calls(
            lambda: _run_chunks(compiled, inputs[1:], grad_outputs[1:], accumulate=True)
        )
        assert num_addmm_calls == 0
        torch.testing.assert_close(lm_head.weight.grad, separate_adds)
    else:
        lm_head.weight.grad = torch.zeros_like(lm_head.weight)

        def grad_weight(input_TD, grad_output_TO):
            with linear_module.accumulate_into_weight_grad():
                output_TO = lm_head(input_TD)
            return torch.autograd.grad(output_TO, lm_head.weight, grad_output_TO)[0]

        with torch.autograd.set_multithreading_enabled(False):
            graph = make_fx(grad_weight)(inputs[0], grad_outputs[0])
        assert not any("addmm" in str(node.target) for node in graph.graph.nodes)
        assert torch.equal(lm_head.weight.grad, torch.zeros_like(lm_head.weight))


@pytest.mark.parametrize("accumulate", [False, True])
def test_weight_grad_accumulation_replays_in_a_cuda_graph(accumulate):
    # torchtitan captures forward + backward in a CUDA graph and zeroes .grad in place between
    # steps. Replays must keep adding into that buffer, even while something else holds it (as
    # SDCReplayer does).
    torch.manual_seed(0)
    lm_head = _lm_head()
    lm_head.weight.grad_dtype = torch.float32
    inputs, grad_outputs = _chunks()
    _run_chunks(lm_head, inputs, grad_outputs, accumulate=False)
    expected = lm_head.weight.grad.clone()

    # Warm up on a side stream, as capture requires.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        lm_head.weight.grad.zero_()
        _run_chunks(lm_head, inputs, grad_outputs, accumulate=accumulate)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    held_grad = lm_head.weight.grad
    held_grad.zero_()
    with torch.cuda.graph(graph):
        _run_chunks(lm_head, inputs, grad_outputs, accumulate=accumulate)

    for _ in range(3):
        lm_head.weight.grad.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert lm_head.weight.grad is held_grad
        torch.testing.assert_close(held_grad, expected)


@pytest.mark.parametrize("accumulate", [False, True])
def test_sdc_replayer_checks_captured_chunked_loss_steps(accumulate, monkeypatch):
    # SDCReplayer keeps each parameter's entry .grad across a checked step and replays the step.
    # num_steps=-1 checks every step, including the CUDA graph capture (step 3, after 2 eager
    # warmup steps, as training_engine runs them).
    if not accumulate:
        monkeypatch.setattr(
            linear_module, "accumulate_into_weight_grad", contextlib.nullcontext
        )
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(256, 256, bias=False).to("cuda", torch.bfloat16)
    model.lm_head = _lm_head()
    model.lm_head.weight.grad_dtype = torch.float32
    chunked_loss = ChunkedLossWrapper.Config(num_chunks=4).build()
    chunked_loss.set_lm_head(model.lm_head)
    hidden = torch.randn(256, 256, device="cuda").bfloat16()
    labels = torch.randint(0, 1024, (256,), device="cuda")

    def step(*, hidden, labels):
        loss, _ = chunked_loss(model.decoder(hidden), labels)
        loss.backward()
        return loss

    graphed_step = wrap_with_cuda_graph(step)
    replayer = SDCReplayer(
        SDCReplayer.Config(num_steps=-1, num_replays=1),
        modules=[model],
        device=torch.device("cuda"),
    )
    try:
        for step_index in range(1, 6):
            model.zero_grad(set_to_none=False)
            run = (
                graphed_step
                if step_index > 2
                else partial(run_eager_on_cuda_graph_stream, step)
            )
            # Raises SDCReplayMismatch if a replay's loss or gradients differ.
            replayer.run_fwd_bwd(
                lambda: run(hidden=hidden, labels=labels),
                step=step_index,
                get_loss=lambda loss: loss,
            )
    finally:
        cuda_graph_teardown()


# out_features > num_tokens takes the LM-head layout, out_features < num_tokens the router one.
@pytest.mark.parametrize("num_tokens,out_features", [(64, 1024), (512, 16)])
def test_fp32_input_matches_the_bf16_forward_and_gets_an_fp32_grad_input(
    num_tokens, out_features
):
    # TP hands the lm_head an upcast bf16 input, so it can sum the partial grad_inputs in fp32.
    x = torch.randn(num_tokens, 256, device="cuda").bfloat16()
    weight = (torch.randn(out_features, 256, device="cuda") * 0.02).bfloat16()
    grad_output = torch.randn(num_tokens, out_features, device="cuda")
    x_fp32 = x.float().requires_grad_()

    output = linear_module._FP32OutputLinearFunction.apply(x_fp32, weight, 2)
    output.backward(grad_output)

    assert torch.equal(
        output, linear_module._FP32OutputLinearFunction.apply(x, weight, 2)
    )
    exact = grad_output.double() @ weight.double()
    floor = _relative_error(exact.bfloat16(), exact)
    assert _relative_error(x_fp32.grad, exact) < 0.01 * floor


def test_backward_handles_zero_tokens():
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    linear_module._FP32OutputLinearFunction.apply(x, weight, 3).sum().backward()

    assert x.grad.shape == x.shape
    assert torch.equal(weight.grad, torch.zeros_like(weight.grad))


def _run_fsdp_keeps_fp32_weight_grad(rank, world_size, port, compile):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        # (num_tokens, out_features, grad_output_pieces): the LM head ships 2 pieces, routers 3.
        for num_tokens, out_features, grad_output_pieces in (
            (64, 1024, 2),
            (512, 16, 3),
        ):
            # Same data on every rank, so FSDP's average is the local gradient.
            torch.manual_seed(0)
            layer = FP32OutputLinear.Config(
                in_features=256,
                out_features=out_features,
                grad_output_pieces=grad_output_pieces,
            ).build()
            layer = layer.cuda()
            torch.nn.init.normal_(layer.weight, std=0.02)
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
            forward = torch.compile(layer) if compile else layer
            x = torch.randn(num_tokens, 256, device="cuda").bfloat16()
            grad_output = torch.randn(num_tokens, out_features, device="cuda")

            forward(x).backward(grad_output)

            exact_grad = grad_output.double().T @ x.double()
            floor = _relative_error(exact_grad.bfloat16(), exact_grad)
            grad = layer.weight.grad.full_tensor()
            assert _relative_error(grad, exact_grad) < 0.01 * floor
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize(
    "compile",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="AOTAutograd rounds grad_weight to bf16 (https://github.com/pytorch/pytorch/pull/197381)",
            ),
        ),
    ],
)
def test_fsdp_keeps_fp32_weight_grad(compile):
    mp.spawn(
        _run_fsdp_keeps_fp32_weight_grad,
        args=(2, get_free_port(), compile),
        nprocs=2,
        join=True,
    )


def _run_tp_sums_lm_head_grad_input_in_fp32(rank, world_size, port, enable_sp, chunked):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=world_size,
            pp=1,
            ep=1,
            world_size=world_size,
            enable_sequence_parallel=enable_sp,
        )
        parallelism_context.build_mesh()
        # The decoder's root norm and lm_head; an identity norm keeps the hidden state as is.
        num_tokens, dim, vocab = 64, 256, 4096
        config = SimpleNamespace(
            tok_embeddings=SimpleNamespace(),
            norm=Identity.Config(),
            lm_head=FP32OutputLinear.Config(
                in_features=dim, out_features=vocab, grad_output_pieces=2
            ),
        )
        set_decoder_sharding_config(config, enable_sp=enable_sp)
        norm = config.norm.build()
        # Same data on every rank.
        torch.manual_seed(0)
        lm_head = config.lm_head.build()
        torch.nn.init.normal_(lm_head.weight, std=0.02)
        lm_head = lm_head.to(device="cuda", dtype=torch.bfloat16)
        weight = lm_head.weight.detach().clone()
        norm._parallelize(parallelism_context)
        lm_head._parallelize(parallelism_context)
        x = torch.randn(num_tokens, dim, device="cuda").bfloat16()
        labels = torch.randint(0, vocab, (num_tokens,), device="cuda")
        x_local = x.chunk(world_size)[rank] if enable_sp else x
        grad_logits = []

        def save_grad_logits(module, args, logits):
            logits.register_hook(grad_logits.append)

        lm_head.register_forward_hook(save_grad_logits)

        def step():
            grad_logits.clear()
            lm_head.weight.grad = None
            x_leaf = x_local.clone().requires_grad_()
            with parallelism_context.activate_spmd():
                with torch.no_grad():
                    # No gradient to reduce, e.g. a generator: the boundary keeps bf16.
                    assert norm(x_leaf).dtype == torch.bfloat16
                # Nor for an input that doesn't require grad, e.g. a frozen reference model.
                assert norm(x_local).dtype == torch.bfloat16
                hidden = norm(x_leaf)
                if chunked:
                    loss_fn = ChunkedLossWrapper.Config(
                        num_chunks=4,
                        loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab),
                    ).build()
                    loss_fn.set_lm_head(lm_head)
                    loss, _ = loss_fn(hidden, labels)
                else:
                    loss = cross_entropy_loss(
                        lm_head(hidden), labels, global_vocab_size=vocab
                    )
                loss.backward()
            return hidden.dtype, loss, x_leaf.grad, lm_head.weight.grad

        hidden_dtype, loss, grad_input, grad_weight = step()
        # Exact grad_input: the fp32 grad_logits of every vocab shard, times the full weight.
        grad_logits_local = torch.cat(grad_logits)
        grad_logits_shards = [
            torch.empty_like(grad_logits_local) for _ in range(world_size)
        ]
        dist.all_gather(grad_logits_shards, grad_logits_local)
        exact = torch.cat(grad_logits_shards, dim=1).double() @ weight.double()
        exact = exact.chunk(world_size)[rank] if enable_sp else exact
        assert hidden_dtype == torch.float32
        # Summing bf16 partials rounds each rank's partial too: ~1.4x the floor at TP=2.
        floor = _relative_error(exact.bfloat16(), exact)
        assert _relative_error(grad_input, exact) < 1.05 * floor

        # Only grad_input changes: the bf16 boundary gives the same loss and grad_weight.
        norm._sharding_config.out_dst_grad_dtype = None
        hidden_dtype, bf16_loss, _, bf16_grad_weight = step()
        assert hidden_dtype == torch.bfloat16
        assert torch.equal(loss, bf16_loss)
        assert torch.equal(grad_weight, bf16_grad_weight)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_tp_sums_lm_head_grad_input_in_fp32(enable_sp, chunked):
    mp.spawn(
        _run_tp_sums_lm_head_grad_input_in_fp32,
        args=(2, get_free_port(), enable_sp, chunked),
        nprocs=2,
        join=True,
    )


def _run_fsdp_chunked_loss_adds_into_weight_grad(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        # Same data on every rank, so FSDP's average is the local gradient.
        torch.manual_seed(0)
        lm_head = FP32OutputLinear.Config(
            in_features=256, out_features=1024, grad_output_pieces=2
        ).build()
        lm_head = lm_head.cuda()
        torch.nn.init.normal_(lm_head.weight, std=0.02)
        fully_shard(lm_head, mesh=mesh, mp_policy=mp_policy)
        chunked_loss = ChunkedLossWrapper.Config(num_chunks=4).build()
        chunked_loss.set_lm_head(lm_head)
        hidden = torch.randn(256, 256, device="cuda").bfloat16().requires_grad_()
        labels = torch.randint(0, 1024, (256,), device="cuda")

        def step():
            loss, _ = chunked_loss(hidden, labels)
            loss.backward()

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                linear_module, "accumulate_into_weight_grad", contextlib.nullcontext
            )
            assert _num_addmm_calls(step) == 0
        separate_adds = lm_head.weight.grad.full_tensor()
        lm_head.zero_grad(set_to_none=True)
        # Chunks 0-2 skip FSDP's gradient sync, so chunks 1-3 find an fp32 .grad to add into.
        assert _num_addmm_calls(step) == 3
        torch.testing.assert_close(lm_head.weight.grad.full_tensor(), separate_adds)

        weight = lm_head.weight.full_tensor().detach().bfloat16()
        exact_grad = 0
        for x, chunk_labels in zip(hidden.detach().chunk(4), labels.chunk(4)):
            logits = torch.mm(x, weight.T, out_dtype=torch.float32).requires_grad_()
            F.cross_entropy(logits, chunk_labels, reduction="sum").backward()
            exact_grad = exact_grad + logits.grad.double().T @ x.double()
        floor = _relative_error(exact_grad.bfloat16(), exact_grad)
        grad = lm_head.weight.grad.full_tensor()
        assert _relative_error(grad, exact_grad) < 0.01 * floor
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_fsdp_chunked_loss_adds_into_weight_grad():
    mp.spawn(
        _run_fsdp_chunked_loss_adds_into_weight_grad,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )


def test_batch_invariant_mode_computes_in_fp32(monkeypatch):
    lm_head = _lm_head()
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    monkeypatch.setattr(linear_module, "is_in_batch_invariant_mode", lambda: True)

    out = lm_head(x)
    out.sum().backward()

    assert torch.equal(out, F.linear(x.detach().float(), lm_head.weight.float()))
    assert x.grad.dtype is torch.bfloat16


def test_backward_runs_under_autocast_with_fp32_params():
    # Autocast makes the fp32 fallback's output bf16; the backward must still run.
    layer = FP32OutputLinear.Config(in_features=64, out_features=32).build().cuda()
    x = torch.randn(16, 64, device="cuda", requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x)
    out.float().sum().backward()

    assert x.grad.dtype is torch.float32
    assert layer.weight.grad.dtype is torch.float32
