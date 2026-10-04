# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``split_mm``'s kernel compiled for H100 with no GPU needed, and its SASS checked for wgmma races.

Here rather than in cpu/: CPU CI installs torch without Triton.
"""

import importlib.util

import pytest

from tests.unit_tests.gpu import sass_hazard_check


def _synthetic_sass(body: str, before_hgmma: str = "") -> str:
    lines = [
        "        /*0000*/ WARPGROUP.ARRIVE ;",
        before_hgmma,
        "        /*0010*/ HGMMA.64x256x16.F32.BF16 R24, R164, gdesc[UR28].tnspB, R24, gsb0 ;",
        f"        /*0040*/ {body}",
        "        /*0060*/ WARPGROUP.DEPBAR.LE gsb0, 0x0 ;",
        "        /*0070*/ EXIT ;",
    ]
    return "\n".join(line for line in lines if line)


# The HGMMA reads A from R164-R167 and accumulates into R24-R151 until the DEPBAR.LE 0x0.
@pytest.mark.parametrize(
    "sass,num_hazards",
    [
        (_synthetic_sass("NOP ;"), 0),
        (_synthetic_sass("MOV R200, RZ ;"), 0),
        (_synthetic_sass("VIADD R165, R3, 0x1 ;"), 1),
        (_synthetic_sass("LOP3.LUT P0, R166, R3, 0x1, RZ, 0xc0, !PT ;"), 1),
        (_synthetic_sass("MOV R30, RZ ;"), 1),
        (_synthetic_sass("MOV R140, RZ ;"), 1),
        (_synthetic_sass("FADD R200, R30, R31 ;"), 1),
        (_synthetic_sass("STS.64 [R2], R4 ;"), 1),
        (_synthetic_sass("ATOMS.ADD RZ, [R2], R4 ;"), 1),
        (_synthetic_sass("LDGSTS.E.BYPASS.128 [R2], desc[UR4][R6.64] ;"), 1),
        (_synthetic_sass("HMMA.16816.F32.BF16 R162, R4, R8, R162 ;"), 1),
        (_synthetic_sass("I2F.F64.S32 R163, R3 ;"), 1),
        (_synthetic_sass("BRX R4 -0x50 ;"), 1),
        (_synthetic_sass("CALL.REL.NOINC `(.L_x_9) ;"), 1),
        (
            _synthetic_sass(
                "WARPGROUP.DEPBAR.LE gsb0, 0x1 ;\n        /*0050*/ VIADD R165, R3, 0x1 ;"
            ),
            1,
        ),
        (_synthetic_sass("NOP ;", "        /*0008*/ VIADD R165, R3, 0x1 ;"), 1),
    ],
    ids=[
        "unrelated op",
        "unrelated register",
        "A operand",
        "A operand after a predicate",
        "accumulator low half",
        "accumulator high half",
        "accumulator read",
        "shared store",
        "shared atomic",
        "copy into shared memory",
        "4-register HMMA result",
        "2-register F64 result",
        "indirect branch",
        "call",
        "after a partial wait",
        "after the arrive, before the HGMMA",
    ],
)
def test_checker_finds_each_hazard_class(sass, num_hazards):
    hazards = sass_hazard_check.find_hazards(sass_hazard_check.parse_sass(sass))

    assert len(hazards) == num_hazards, hazards


# (M, N, K, a is transposed): grad_input runs a = grad_output, grad_weight a = grad_output.T.
_SHAPES = {
    "lm head grad_input": (2048, 4096, 151936, False),
    "lm head grad_weight": (151936, 4096, 2048, True),
    "short-k grad_input": (2048, 4096, 4096, False),
    "router grad_input": (16384, 2048, 128, False),
    "router grad_weight, split-k": (128, 2048, 16384, True),
    "odd shape": (77, 136, 4104, False),
    "odd shape, transposed": (76, 136, 4104, True),
    "over 2^31 outputs": (532480, 4096, 64, True),
    "few-token grad_input, split-k, promoted": (64, 4096, 151936, False),
    "long-chunk grad_weight, promoted": (151936, 4096, 8192, True),
}


def compile_for_h100(
    m: int, n: int, k: int, a_is_transposed: bool, num_pieces: int
) -> bytes:
    """The cubin split_mm launches for an M x N x K product on an H100 (132 SMs), built without a GPU.

    Mirrors Triton's specialization of integer arguments: 1 becomes a constexpr, multiples of 16
    are marked divisible.
    """
    import triton

    from torchtitan.models.common import split_mm as split_mm_module
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon._runtime import GluonASTSource

    (
        num_splits,
        k_tiles_per_split,
        promote,
        block_n,
        num_buffers,
    ) = split_mm_module._plan(m, n, k, num_pieces, a_is_transposed, num_sms=132)
    a_block = [64, 128] if a_is_transposed else [128, 64]
    a_layout = gl.NVMMASharedLayout.get_default_for(a_block, gl.float32)
    b_layout = gl.NVMMASharedLayout.get_default_for([64, block_n], gl.bfloat16)
    out_dtype = "bf16" if num_splits == 1 and not a_is_transposed else "fp32"
    integers = {
        "M": m,
        "N": n,
        "K": k,
        "k_tiles_per_split": k_tiles_per_split,
        "stride_os": m * n,
        "stride_om": n,
        "stride_on": 1,
    }
    constants = {
        "NUM_PIECES": num_pieces,
        "PROMOTE_EVERY": split_mm_module._PROMOTE_EVERY if promote else 0,
        "NUM_BUFFERS": num_buffers,
        "A_IS_TRANSPOSED": a_is_transposed,
        "BLOCK_M": split_mm_module._BLOCK_M,
        "GROUP_M": 8,
    }
    signature = {
        "a_desc": f"tensordesc<fp32{a_block},{a_layout!r}>",
        "b_desc": f"tensordesc<bf16{[64, block_n]},{b_layout!r}>",
        "out_ptr": f"*{out_dtype}",
    }
    constexprs, attrs = {}, {(2,): [["tt.divisibility", 16]]}
    arg_names = split_mm_module._split_mm_kernel.arg_names
    for name, value in integers.items():
        index = arg_names.index(name)
        if value == 1:
            signature[name], constexprs[(index,)] = "constexpr", 1
            continue
        signature[name] = "i32" if value < 2**31 else "i64"
        attrs[(index,)] = [["tt.divisibility", 16]] if value % 16 == 0 else []
    for name, value in constants.items():
        signature[name], constexprs[(arg_names.index(name),)] = "constexpr", value
    assert list(signature) == arg_names
    source = GluonASTSource(
        split_mm_module._split_mm_kernel, signature, constexprs, attrs
    )
    target = GPUTarget("cuda", 90, 32)
    return triton.compile(source, target=target, options={"num_warps": 4}).asm["cubin"]


@pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="needs Triton's Gluon"
)
@pytest.mark.parametrize("num_pieces", [2, 3])
@pytest.mark.parametrize("shape", list(_SHAPES.values()), ids=list(_SHAPES))
def test_split_mm_kernel_has_no_wgmma_hazards(shape, num_pieces):
    # Skip only without Gluon: a Gluon API break must fail importing split_mm, not skip.
    pytest.importorskip("triton.experimental.gluon")
    cubin = compile_for_h100(*shape, num_pieces=num_pieces)
    try:
        sass = sass_hazard_check.disassemble(cubin)
    except FileNotFoundError:
        pytest.skip("needs nvdisasm")
    parsed = sass_hazard_check.parse_sass(sass)

    # No HGMMA would mean the disassembly format changed, not a race-free kernel.
    assert any(opcode.startswith("HGMMA") for _, opcode, _, _ in parsed[0])
    assert sass_hazard_check.find_hazards(parsed) == []
