# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config import CompileConfig


def test_compile_config_default() -> None:
    config = CompileConfig()
    assert config.components == ["model", "loss"]


def test_compile_config_model_only() -> None:
    config = CompileConfig(components=["model"])
    assert config.components == ["model"]


def test_compile_config_loss_only() -> None:
    config = CompileConfig(components=["loss"])
    assert config.components == ["loss"]


def test_compile_config_empty_components() -> None:
    config = CompileConfig(components=[])
    assert config.components == []


def test_compile_config_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match=r"foo.*allowed values are.*loss.*model"):
        CompileConfig(components=["foo"])


def test_compile_config_async_tp_requires_model_compile() -> None:
    with pytest.raises(
        ValueError,
        match="Async TP requires 'model' in --compile.components",
    ):
        CompileConfig(enable_async_tensor_parallel=True, components=["loss"])


def test_compile_config_lowers_cat_as_concat_kernel_by_default() -> None:
    assert CompileConfig().inductor_options == {
        "max_pointwise_cat_inputs": 0,
        "max_complex_pointwise_cat_inputs": 0,
    }


@pytest.mark.parametrize(
    ("backend", "expects_options"), [("inductor", True), ("aot_eager", False)]
)
def test_apply_compile_passes_inductor_options_to_blocks(
    backend: str, expects_options: bool, monkeypatch
) -> None:
    import torch

    from torchtitan.distributed.compile import apply_compile

    calls = []
    monkeypatch.setattr(
        torch.nn.Module, "compile", lambda self, **kwargs: calls.append(kwargs)
    )
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleDict({"0": torch.nn.Linear(2, 2)})
    config = CompileConfig(backend=backend)

    apply_compile(model, compile_config=config, parallel_dims=_single_device_dims())

    expected = config.inductor_options if expects_options else None
    assert calls == [{"backend": backend, "fullgraph": True, "options": expected}]


def _single_device_dims():
    from types import SimpleNamespace

    return SimpleNamespace(tp_enabled=False)
