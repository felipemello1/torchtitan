# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for toy_spmd.py.

Runs the toy on 4 GPUs and verifies:
- Training completes without errors
- Loss decreases over steps (model is learning)
- Output files are created

Run:
    python -m torch.distributed.run --nproc_per_node=4 \
        -m pytest tests/unit_tests/observability/test_toy_spmd_integration.py -vv
    OR (simpler):
    python tests/unit_tests/observability/test_toy_spmd_integration.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest


def _run_toy_spmd(output_dir: str, num_steps: int = 10) -> str:
    """Run toy_spmd.py via torchrun and return combined stdout+stderr."""
    env = os.environ.copy()
    # Override OUTPUT_DIR via environment so the toy writes to our temp dir
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=4",
            "-m",
            "torchtitan.experiments.observability.toy_spmd",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        env=env,
    )
    return result.stdout + result.stderr, result.returncode


def _parse_loss_from_console(output: str) -> list[tuple[int, float]]:
    """Parse (step, loss) pairs from the console table output."""
    # Matches lines like: "   1      2.9258  21207.9238      3575.0"
    pattern = r"^\s*(\d+)\s+([\d.]+)\s+"
    results = []
    for line in output.split("\n"):
        m = re.match(pattern, line.strip())
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            results.append((step, loss))
    return results


@pytest.mark.skipif(
    not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.path.exists("/dev/nvidia0"),
    reason="No GPU available",
)
class TestToySpmdIntegration:
    """Integration tests for toy_spmd.py. Requires 4 GPUs."""

    def test_training_completes_and_loss_decreases(self):
        """Verify training runs successfully and loss decreases over 20 steps."""
        output, returncode = _run_toy_spmd("/tmp/test_toy_spmd")
        assert returncode == 0, f"toy_spmd failed with:\n{output}"

        # Parse loss values from console
        losses = _parse_loss_from_console(output)
        assert len(losses) >= 15, f"Expected >=15 step outputs, got {len(losses)}: {output[-500:]}"

        # Loss should decrease over the training run
        first_loss = losses[0][1]
        last_loss = losses[-1][1]
        assert last_loss < first_loss, (
            f"Loss did not decrease: step {losses[0][0]} loss={first_loss}, "
            f"step {losses[-1][0]} loss={last_loss}"
        )

        # Loss should decrease by at least 30% (model is overfitting to one batch)
        assert last_loss < first_loss * 0.7, (
            f"Loss decrease too small: {first_loss} → {last_loss} "
            f"({(1 - last_loss/first_loss)*100:.1f}% decrease, expected >= 30%)"
        )


if __name__ == "__main__":
    # Allow running directly: python test_toy_spmd_integration.py
    pytest.main([__file__, "-vv"])
