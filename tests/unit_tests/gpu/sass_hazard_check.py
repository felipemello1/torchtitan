# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static check of a Hopper cubin for wgmma (HGMMA) operand races, used on ``split_mm``'s kernel.

An async HGMMA reads its A registers, its shared-memory operands and its accumulator after issue,
until a ``WARPGROUP.DEPBAR.LE gsb0, 0x0`` (wait for all groups). ptxas does not extend the A
registers' live range to that wait, so it may reuse them for the next tile's split, which raced
(NaN) in an earlier version of the kernel. Walking the SASS control flow, this reports:

- forward, from each HGMMA to a full wait: writes to its A or accumulator registers, reads of its
  accumulator, shared-memory writes (``STS``, ``ATOMS``, ``LDGSTS``; conservative: a write to
  another buffer counts), and calls and indirect branches, which the walk can't follow;
- backward, from each HGMMA to the ``WARPGROUP.ARRIVE`` before it: writes to those registers.

A non-zero wait (``DEPBAR.LE gsb0, 0x2``) leaves the newest groups in flight, so it does not end a
walk. Destination widths come from the opcode (``.64``, ``.128``, ``LDSM``, ``HMMA``, ``.F64``);
other multi-register writers count as one register. Not modeled: the mbarrier stage protocol,
cross-warp or cross-proxy ordering, and TMA writes into shared memory.

Example:
    hazards = find_hazards(parse_sass(disassemble(cubin_bytes)))
    # ["HGMMA@0x2190 (A=R88): A operand written at 0x21c0 by F2FP.BF16.F32.PACK_AB R88, ..."]
"""

import os
import re
import shutil
import subprocess
import tempfile

_INSTRUCTION = re.compile(
    r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Za-z0-9_.]*)\s*([^;]*);"
)
_PREDICATE = re.compile(r"^!?U?P(T|\d+)$")
# Predicate-first opcodes whose second operand is a source, not a destination.
_PREDICATE_ONLY = (
    "ISETP",
    "UISETP",
    "FSETP",
    "DSETP",
    "HSETP2",
    "PLOP3",
    "UPLOP3",
    "FCHK",
    "VOTEU",
    "VOTE",
    "ELECT",
    "SYNCS.PHASECHK",
    "R2P",
    "PSETP",
)
_NO_REGISTER_RESULT = (
    "ST",
    "BRA",
    "BAR",
    "WARPGROUP",
    "EXIT",
    "RED",
    "UTMA",
    "NOP",
    "RET",
    "CALL",
    "MEMBAR",
    "FENCE",
    "DEPBAR",
    "ERRBAR",
    "CCTL",
    "WARPSYNC",
    "BSSY",
    "BSYNC",
    "YIELD",
    "LDGDEPBAR",
    "BRX",
    "JMX",
)


def disassemble(cubin: bytes) -> str:
    """SASS text of a cubin, from Triton's bundled ``nvdisasm`` (else the one on PATH)."""
    import triton

    bundled = os.path.join(
        os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "nvdisasm"
    )
    nvdisasm = bundled if os.path.exists(bundled) else shutil.which("nvdisasm")
    if nvdisasm is None:
        raise FileNotFoundError("nvdisasm")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as file:
        file.write(cubin)
        file.flush()
        return subprocess.run(
            [nvdisasm, "-c", file.name], capture_output=True, text=True, check=True
        ).stdout


def parse_sass(sass: str) -> tuple[list[tuple[int, str, str, bool]], dict[str, int]]:
    """``[(address, opcode, operands, predicated)]`` and ``{label: instruction index}``."""
    instructions, labels = [], {}
    for line in sass.splitlines():
        label = re.match(r"^(\.L_x_\d+):", line)
        if label:
            labels[label.group(1)] = len(instructions)
            continue
        match = _INSTRUCTION.search(line)
        if match:
            instructions.append(
                (
                    int(match.group(1), 16),
                    match.group(3),
                    match.group(4).strip(),
                    bool(match.group(2)),
                )
            )
    return instructions, labels


def find_hazards(parsed: tuple[list, dict[str, int]]) -> list[str]:
    """One line per hazard; see the module docstring for what counts."""
    instructions, labels = parsed
    predecessors: dict[int, list[int]] = {}
    for j in range(len(instructions)):
        for successor in _successors(instructions, labels, j):
            predecessors.setdefault(successor, []).append(j)
    hazards = []
    for i, (address, opcode, operands, _) in enumerate(instructions):
        if not opcode.startswith("HGMMA"):
            continue
        accumulator, a_registers, a_operand = _hgmma_registers(opcode, operands)
        prefix = f"HGMMA@{address:#06x} (A={a_operand})"
        # ======== Forward: the HGMMA runs until a full wait ========
        stack, seen = [i + 1], set()
        while stack:
            j = stack.pop()
            if j in seen or j >= len(instructions):
                continue
            seen.add(j)
            j_address, j_opcode, j_operands, _ = instructions[j]
            if j_opcode.startswith("WARPGROUP.DEPBAR") and re.search(
                r"gsb0,\s*0x0\b", j_operands
            ):
                continue
            where = f"at {j_address:#06x} by {j_opcode} {j_operands}"
            if j_opcode.startswith(("BRX", "JMX", "CALL")):
                hazards.append(f"{prefix}: branch the walk can't follow {where}")
            elif j_opcode.startswith(("STS", "ATOMS", "LDGSTS")):
                hazards.append(f"{prefix}: shared store {where}")
            elif not j_opcode.startswith("HGMMA"):
                clobbered = _written_registers(j_opcode, j_operands) & (
                    a_registers | accumulator
                )
                if clobbered & a_registers:
                    hazards.append(f"{prefix}: A operand written {where}")
                elif clobbered:
                    hazards.append(f"{prefix}: accumulator written {where}")
                elif _read_registers(j_operands) & accumulator:
                    hazards.append(f"{prefix}: accumulator read {where}")
            stack.extend(_successors(instructions, labels, j))
        # ======== Backward: from the last WARPGROUP.ARRIVE (wgmma.fence) to the HGMMA ========
        stack, seen = list(predecessors.get(i, [])), set()
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            j_address, j_opcode, j_operands, _ = instructions[j]
            if j_opcode.startswith("WARPGROUP.ARRIVE"):
                continue
            if not j_opcode.startswith("HGMMA") and _written_registers(
                j_opcode, j_operands
            ) & (a_registers | accumulator):
                hazards.append(
                    f"{prefix}: written after the last WARPGROUP.ARRIVE at "
                    f"{j_address:#06x} by {j_opcode} {j_operands}"
                )
            stack.extend(predecessors.get(j, []))
    return hazards


def _register_range(token: str, width: int) -> set[int]:
    match = re.fullmatch(r"-?\|?R(\d+)\|?(\.[A-Za-z0-9_]+)*", token.strip())
    if match is None:
        return set()
    return set(range(int(match.group(1)), int(match.group(1)) + width))


def _destination_width(opcode: str) -> int:
    if ".128" in opcode or re.search(r"LDSM\.\d+\.\w+\.4", opcode):
        return 4
    if re.search(r"LDSM\.\d+\.\w+\.2", opcode):
        return 2
    if opcode.startswith(("HMMA", "IMMA", "DMMA")):
        return 4
    if ".64" in opcode or ".WIDE" in opcode or ".F64" in opcode:
        return 2
    if opcode.startswith(("DADD", "DFMA", "DMUL", "CS2R")) and ".32" not in opcode:
        return 2
    return 1


def _written_registers(opcode: str, operands: str) -> set[int]:
    if opcode.startswith(_NO_REGISTER_RESULT) or not operands:
        return set()
    tokens = [token.strip() for token in operands.split(",")]
    registers = _register_range(tokens[0], _destination_width(opcode))
    # LOP3.LUT P0, R5, ...: a predicate first, then a register result.
    if (
        _PREDICATE.match(tokens[0])
        and len(tokens) > 1
        and not opcode.startswith(_PREDICATE_ONLY)
    ):
        registers |= _register_range(tokens[1], _destination_width(opcode))
    return registers


def _read_registers(operands: str) -> set[int]:
    return {
        int(match.group(1))
        for token in operands.split(",")[1:]
        for match in re.finditer(r"\bR(\d+)\b", token)
    }


def _hgmma_registers(opcode: str, operands: str) -> tuple[set[int], set[int], str]:
    """Accumulator registers (64 x N fp32: N / 2 per thread), A registers (4), the A operand."""
    tokens = [token.strip() for token in operands.split(",")]
    shape = re.search(r"HGMMA\.64x(\d+)x\d+\.F32", opcode)
    accumulator = _register_range(tokens[0], int(shape.group(1)) // 2 if shape else 128)
    a_registers = _register_range(tokens[1], 4) if tokens[1].startswith("R") else set()
    return accumulator, a_registers, tokens[1]


def _successors(instructions: list, labels: dict[str, int], j: int) -> list[int]:
    _, opcode, operands, predicated = instructions[j]
    if opcode.startswith("BRA"):
        target = re.search(r"(\.L_x_\d+)", operands)
        successors = [labels[target.group(1)]] if target else []
        return successors + ([j + 1] if predicated else [])
    if opcode.startswith("EXIT") and not predicated:
        return []
    return [j + 1]
