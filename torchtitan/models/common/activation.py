# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from torchtitan.config.configurable import Configurable
from torchtitan.config.function import Function
from torchtitan.distributed.local_compile import local_compile


class BinaryActivationFn(Function[torch.Tensor], ABC):
    """Base class for configurable two-input activation functions."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
        pass

    @abstractmethod
    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass


class UnaryActivationFn(Function[torch.Tensor], ABC):
    """Base class for configurable one-input activation functions."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):  # pyrefly: ignore[bad-override]
        pass

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass


class Sigmoid(UnaryActivationFn):
    """Sigmoid activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return torch.sigmoid(x)


class SiLU(UnaryActivationFn):
    """SiLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return F.silu(x)


class Softmax(UnaryActivationFn):
    """Softmax activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        dim: int = -1

    def __init__(self, config: Config) -> None:
        self.dim = config.dim

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return F.softmax(x, dim=self.dim)


class SqrtSoftplus(UnaryActivationFn):
    """Square root of softplus activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(UnaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        return F.softplus(x).sqrt()


# Each hidden size gets one graph per grad mode and per static/dynamic row
# count. Routed calls with 0 rows (an EP rank that receives no tokens) or 1 row
# also specialize, since Dynamo never treats sizes 0 and 1 as dynamic. So these
# regions can need more than torch.compile's default limit of 8 graphs.
_GLU_RECOMPILE_LIMIT = 16


def _mark_hidden_dim_static(gate: torch.Tensor, up: torch.Tensor) -> None:
    """Keep the hidden size static when one compiled function sees several.

    Example: Kimi K3's situglu serves the dense FFN (33792), the shared expert
    (6144) and the routed experts (3072). Without the mark, automatic dynamic
    shapes make the hidden size symbolic once a second size arrives, and the
    kernel indexes with a runtime divisor. With it, each hidden size gets its
    own graph.
    """
    if torch.compiler.is_compiling():
        torch._dynamo.mark_static(gate, -1)
        torch._dynamo.mark_static(up, -1)


class SwiGLU(BinaryActivationFn):
    """SwiGLU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        pass

    def __init__(self, config: Config) -> None:
        pass

    @local_compile("swiglu", batch_invariant=True, recompile_limit=_GLU_RECOMPILE_LIMIT)
    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        _mark_hidden_dim_static(gate, up)
        return F.silu(gate) * up


class SiTUGLU(BinaryActivationFn):
    """Kimi's SiTU-GLU activation, evaluated in FP32."""

    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        beta: float = 1.0
        linear_beta: float | None = None

    def __init__(self, config: Config) -> None:
        self.beta = config.beta
        self.linear_beta = config.linear_beta

    @local_compile(
        "situglu", batch_invariant=True, recompile_limit=_GLU_RECOMPILE_LIMIT
    )
    def __call__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        _mark_hidden_dim_static(gate, up)
        input_dtype = gate.dtype
        gate = gate.float()
        up = up.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(input_dtype)
