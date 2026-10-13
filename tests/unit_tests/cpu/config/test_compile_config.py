# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.loss import compute_logprobs, cross_entropy_loss, mse_loss
from torchtitan.distributed import local_compile as local_compile_module
from torchtitan.distributed.local_compile import apply_local_compile, local_compile
from torchtitan.models.common.moe import _microbatch_load_balance_local_stats


@pytest.fixture(autouse=True)
def reset_local_compile():
    apply_local_compile([])
    yield
    apply_local_compile([])


def test_apply_local_compile_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"foo.*registered values"):
        apply_local_compile(["foo"])


def test_local_compile_binds_once_from_existing_config(monkeypatch) -> None:
    compiled_calls = []

    @local_compile("test_local_compile", batch_invariant=False)
    def fn(value: int) -> tuple[str, int]:
        return "eager", value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))

        def compiled(value: int) -> tuple[str, int]:
            return "compiled", value

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["test_local_compile"])

    assert fn(3) == ("compiled", 3)
    # One compile for small inputs, one for large inputs.
    assert len(compiled_calls) == 2
    for _, kwargs in compiled_calls:
        assert callable(kwargs["backend"])
        assert kwargs["fullgraph"] is True
        assert kwargs["isolate_recompiles"] is True


def test_local_compile_forwards_compile_kwargs(monkeypatch) -> None:
    compiled_calls = []
    options = {"max_autotune": True}

    @local_compile(
        "test_compile_kwargs",
        batch_invariant=False,
        backend="eager",
        dynamic=True,
        fullgraph=True,
        options=options,
    )
    def fn(value: int) -> int:
        return value

    def fake_compile(reference, **kwargs):
        compiled_calls.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["test_compile_kwargs"])

    assert fn(3) == 3
    assert [kwargs for _, kwargs in compiled_calls] == [
        {
            "backend": "eager",
            "fullgraph": True,
            "isolate_recompiles": True,
            "dynamic": True,
            "options": options,
        }
    ] * 2


def test_local_compile_dispatches_on_largest_tensor_argument(monkeypatch) -> None:
    @local_compile("test_size_dispatch", batch_invariant=False)
    def fn(scale: torch.Tensor, x: torch.Tensor) -> str:
        return "eager"

    sizes = iter(["small", "large"])

    def fake_compile(reference, **kwargs):
        size = next(sizes)
        return lambda *args, **kwargs: size

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(local_compile_module, "_LARGE_INPUT_NUMEL", 8)
    apply_local_compile(["test_size_dispatch"])

    assert fn(torch.ones(1), torch.ones(8)) == "small"
    assert fn(torch.ones(1), torch.ones(9)) == "large"
    assert fn(torch.ones(1), x=torch.ones(9)) == "large"


def test_local_compile_sizes_get_separate_inductor_cache_entries(monkeypatch) -> None:
    @local_compile("test_cache_tag", batch_invariant=False)
    def fn(x: torch.Tensor) -> torch.Tensor:
        return x * 2 + 1

    tags = []

    def fake_compile_fx(gm, example_inputs, config_patches=None):
        tags.append(torch.compiler.config.cache_key_tag)
        return gm

    monkeypatch.setattr(local_compile_module, "compile_fx", fake_compile_fx)
    monkeypatch.setattr(local_compile_module, "_LARGE_INPUT_NUMEL", 8)
    apply_local_compile(["test_cache_tag"])
    try:
        for numel in (4, 16):
            x = torch.arange(numel, dtype=torch.float32)
            torch.testing.assert_close(fn(x), x * 2 + 1)
    finally:
        torch._dynamo.reset()

    assert tags == [
        "local_compile:test_cache_tag:small",
        "local_compile:test_cache_tag:large",
    ]


def test_local_compile_rejects_fullgraph_false() -> None:
    with pytest.raises(ValueError, match="fullgraph=True"):
        local_compile(
            "test_fullgraph",
            batch_invariant=False,
            fullgraph=False,
        )


def test_local_compile_rejects_non_batch_invariant_region(monkeypatch) -> None:
    @local_compile("test_non_batch_invariant", batch_invariant=False)
    def fn(value: int) -> int:
        return value

    monkeypatch.setattr(
        "torchtitan.distributed.local_compile.is_in_batch_invariant_mode",
        lambda: True,
    )
    with pytest.raises(
        ValueError, match=r"test_non_batch_invariant.*local_compile_regions"
    ):
        apply_local_compile(["test_non_batch_invariant"])


def test_local_compile_inside_compiled_code_adds_no_graphs(monkeypatch) -> None:
    @local_compile("test_nested", batch_invariant=False)
    def inner(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    graphs = []

    def outer_backend(gm, example_inputs):
        graphs.append(gm)
        return gm

    monkeypatch.setattr(local_compile_module, "_LARGE_INPUT_NUMEL", 8)
    apply_local_compile(["test_nested"])
    outer = torch.compile(lambda x: inner(x) + 1, backend=outer_backend, dynamic=True)
    try:
        for numel in (4, 16):
            x = torch.arange(numel, dtype=torch.float32)
            torch.testing.assert_close(outer(x), x * 2 + 1)
    finally:
        torch._dynamo.reset()

    # One dynamic graph: the region traces into the outer graph without a size guard.
    assert len(graphs) == 1


def test_loss_functions_use_local_compile(monkeypatch) -> None:
    compiled_names = []

    def fake_compile(reference, **kwargs):
        del kwargs
        compiled_names.append(reference.__name__)
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["loss"])

    assert compiled_names == [
        name
        for fn in (cross_entropy_loss, mse_loss, compute_logprobs)
        for name in (fn.__name__,) * 2
    ]


def test_moe_aux_loss_uses_local_compile(monkeypatch) -> None:
    compiled = []

    def fake_compile(reference, **kwargs):
        compiled.append((reference, kwargs))
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    apply_local_compile(["moe_aux_loss"])

    assert [reference for reference, _ in compiled] == [
        _microbatch_load_balance_local_stats.__wrapped__
    ] * 2
    assert all(kwargs["fullgraph"] is True for _, kwargs in compiled)
