# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inductor post-grad passes for the generator's vLLM compile."""

import torch
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)

from vllm.compilation.passes.inductor_pass import InductorPass
from vllm.config.utils import Range

aten = torch.ops.aten

_unfuse_residual_addmm_patterns = PatternMatcherPass(pass_name="unfuse_residual_addmm")


class UnfuseResidualAddmmPass(InductorPass):
    """Undo inductor's ``x + mm(a, b) -> addmm(x, a, b)`` in single-token graphs.

    ``addmm`` first copies the residual ``x`` into its output, so each residual
    add costs a copy kernel. At one token that copy is a whole extra launch per
    projection (128 per step at Qwen3.5-27B), and the unfused add instead fuses
    into the next RMSNorm kernel:

        addmm:   memcpy(x -> out), gemm(beta=1)
        unfused: gemm(beta=0), add fused into the next norm

    Larger token counts keep ``addmm``, which measures faster there.
    """

    def __call__(self, graph: torch.fx.Graph) -> None:
        _unfuse_residual_addmm_patterns.apply(graph)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= 1

    def uuid(self) -> str:
        return InductorPass.hash_source(
            self, _is_residual_addmm, _unfuse_residual_addmm
        )


def _is_residual_addmm(match: Match) -> bool:
    inp = match.kwargs["inp"].meta["val"]
    mat1, mat2 = (node.meta["val"] for node in match.args)
    # A broadcast bias (stride 0) needs no copy, so it stays fused.
    return tuple(inp.shape) == (mat1.shape[0], mat2.shape[1]) and 0 not in inp.stride()


@register_graph_pattern(
    CallFunction(aten.addmm.default, KeywordArg("inp"), Arg(), Arg()),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=_unfuse_residual_addmm_patterns,
    extra_check=_is_residual_addmm,
)
def _unfuse_residual_addmm(
    match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node, *, inp: torch.fx.Node
) -> None:
    def repl(inp, mat1, mat2):
        return aten.mm(mat1, mat2) + inp

    match.replace_by_example(repl, [inp, mat1, mat2])
