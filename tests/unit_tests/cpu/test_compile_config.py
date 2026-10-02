# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import re

import pytest
import torch

from torchtitan.components.loss import cross_entropy_loss, mse_loss
from torchtitan.distributed.local_compile import local_compile, LocalCompileConfig
from torchtitan.models.common.activation import SiTUGLU, SwiGLU


@pytest.fixture(autouse=True)
def reset_local_compile():
    LocalCompileConfig(regions=[]).apply_local_compile()
    yield
    LocalCompileConfig(regions=[]).apply_local_compile()


def test_local_compile_config_default() -> None:
    config = LocalCompileConfig()
    assert config.regions == [
        "gated_rmsnorm",
        "loss",
        "swiglu",
        "situglu",
        "cos_sin_rope",
    ]


def test_local_compile_config_loss_only() -> None:
    config = LocalCompileConfig(regions=["loss"])
    assert config.regions == ["loss"]


def test_local_compile_config_empty_regions() -> None:
    config = LocalCompileConfig(regions=[])
    assert config.regions == []


def test_local_compile_config_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match=r"foo.*registered values"):
        LocalCompileConfig(regions=["foo"]).apply_local_compile()


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
    LocalCompileConfig(regions=["test_local_compile"]).apply_local_compile()

    assert fn(3) == ("compiled", 3)
    assert len(compiled_calls) == 1
    assert "backend" not in compiled_calls[0][1]
    assert compiled_calls[0][1]["fullgraph"] is True


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
    LocalCompileConfig(regions=["test_compile_kwargs"]).apply_local_compile()

    assert fn(3) == 3
    assert compiled_calls[0][1] == {
        "backend": "eager",
        "fullgraph": True,
        "dynamic": True,
        "options": options,
    }


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
    with pytest.raises(ValueError, match=r"test_non_batch_invariant.*compile.regions"):
        LocalCompileConfig(regions=["test_non_batch_invariant"]).apply_local_compile()


def test_loss_functions_use_local_compile(monkeypatch) -> None:
    compiled_names = []

    def fake_compile(reference, **kwargs):
        del kwargs
        compiled_names.append(reference.__name__)
        return reference

    monkeypatch.setattr(torch, "compile", fake_compile)
    LocalCompileConfig(regions=["loss"]).apply_local_compile()

    assert compiled_names == [cross_entropy_loss.__name__, mse_loss.__name__]


@pytest.mark.parametrize(
    "region, activation",
    [("swiglu", SwiGLU.Config()), ("situglu", SiTUGLU.Config(beta=4.0))],
)
def test_glu_regions_keep_hidden_size_static(monkeypatch, region, activation) -> None:
    graph_gate_shapes = []

    def record_gate_shape(gm, example_inputs):
        del example_inputs
        gate = next(
            node.meta["example_value"]
            for node in gm.graph.find_nodes(op="placeholder")
            if isinstance(node.meta["example_value"], torch.Tensor)
        )
        # "(s68, 48)" -> "(s, 48)": symbol names vary across runs.
        shape = str(tuple(gate.shape))
        graph_gate_shapes.append(re.sub(r"s\d+", "s", shape))
        return gm.forward

    torch._dynamo.reset()
    try:
        monkeypatch.setattr(
            torch,
            "compile",
            functools.partial(torch.compile, backend=record_gate_shape),
        )
        LocalCompileConfig(regions=[region]).apply_local_compile()
        act = activation.build()

        def call(rows: int, hidden: int) -> None:
            gate, up = torch.randn(rows, 2, hidden).unbind(-2)
            act(gate, up)

        # Two hidden sizes (e.g. dense and shared expert) at two token counts.
        for tokens in (8, 4):
            for _ in range(2):
                call(tokens, 48)
                call(tokens, 16)

        # The hidden size stays static: one graph per hidden size, first with a
        # static and then with a dynamic token count.
        assert graph_gate_shapes == ["(8, 48)", "(8, 16)", "(s, 48)", "(s, 16)"]
    finally:
        torch._dynamo.reset()
