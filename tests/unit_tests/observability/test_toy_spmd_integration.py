# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for toy_spmd.py.

Runs the toy on 4 GPUs and verifies:
- Training completes without errors
- Loss decreases over steps (model is learning)
- JSONL output files have correct structure at the field level

Run:
    python -m pytest tests/unit_tests/observability/test_toy_spmd_integration.py -vv
"""

import glob
import json
import os
import re
import subprocess
import sys

import pytest

# Path to torchtitan repo root (this test lives in tests/unit_tests/observability/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TOY_OUTPUT_DIR = os.path.join(
    REPO_ROOT, "torchtitan", "experiments", "observability", "outputs", "toy_spmd"
)


def _run_toy_spmd() -> tuple[str, int]:
    """Run toy_spmd.py via torchrun and return (combined output, returncode)."""
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
        cwd=REPO_ROOT,
    )
    return result.stdout + result.stderr, result.returncode


def _parse_loss_from_console(output: str) -> list[tuple[int, float]]:
    """Parse (step, loss) pairs from console output.

    Handles both plain table format and ANSI-colored MetricsProcessor format.
    """
    results = []
    for line in output.split("\n"):
        # Plain table: "   1      3.6902      1.1520      3695.5"
        m = re.search(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            results.append((int(m.group(1)), float(m.group(2))))
            continue
        # ANSI-colored: "step:  1  loss:  3.69016  ..."
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step:\s*(\d+)\s+loss:\s+([\d.]+)", clean)
        if m:
            results.append((int(m.group(1)), float(m.group(2))))
    return results


def _load_jsonl(filepath: str) -> list[dict]:
    """Load all JSON objects from a JSONL file."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@pytest.mark.skipif(
    not os.path.exists("/dev/nvidia0"),
    reason="No GPU available",
)
class TestToySpmdIntegration:
    """Integration tests for toy_spmd.py. Requires 4 GPUs."""

    @pytest.fixture(autouse=True, scope="class")
    def run_toy(self, request):
        """Run toy_spmd once for all tests in the class."""
        output, returncode = _run_toy_spmd()
        request.cls.output = output
        request.cls.returncode = returncode
        yield

    def test_training_completes(self):
        """Training runs without errors."""
        assert self.returncode == 0, f"toy_spmd failed with:\n{self.output[-2000:]}"

    def test_loss_decreases(self):
        """Loss decreases by at least 30% (overfitting to single batch)."""
        assert self.returncode == 0, "Training failed"
        losses = _parse_loss_from_console(self.output)
        assert len(losses) >= 3, f"Expected >=3 logged steps, got {len(losses)}"

        first_loss = losses[0][1]
        last_loss = losses[-1][1]
        assert last_loss < first_loss * 0.7, (
            f"Loss decrease too small: {first_loss:.4f} → {last_loss:.4f} "
            f"({(1 - last_loss / first_loss) * 100:.1f}%)"
        )

    def test_system_jsonl_files_created(self):
        """One system JSONL file per rank (4 total)."""
        assert self.returncode == 0, "Training failed"
        sys_dir = os.path.join(TOY_OUTPUT_DIR, "system_logs")
        jsonl_files = sorted(glob.glob(os.path.join(sys_dir, "*.jsonl")))
        assert len(jsonl_files) == 4, f"Expected 4 JSONL files, got {len(jsonl_files)}: {jsonl_files}"

    def test_system_jsonl_structure(self):
        """System JSONL entries have correct fields: rank, step, source, event types."""
        assert self.returncode == 0, "Training failed"
        sys_dir = os.path.join(TOY_OUTPUT_DIR, "system_logs")
        jsonl_files = sorted(glob.glob(os.path.join(sys_dir, "*.jsonl")))
        assert len(jsonl_files) == 4

        all_ranks_seen = set()
        for filepath in jsonl_files:
            records = _load_jsonl(filepath)
            assert len(records) > 0, f"Empty JSONL: {filepath}"

            for record in records:
                # System JSONL uses 4-column format: int, normal, double, normvector
                assert "int" in record, f"Missing 'int' column in {filepath}"
                assert "normal" in record, f"Missing 'normal' column in {filepath}"

                int_fields = record["int"]
                normal_fields = record["normal"]

                # Rank must be present and valid (0-3)
                assert "rank" in int_fields, f"Missing rank in {filepath}"
                rank = int_fields["rank"]
                assert 0 <= rank <= 3, f"Invalid rank {rank}"
                all_ranks_seen.add(rank)

                # Source must be "trainer"
                assert normal_fields.get("source") == "trainer", (
                    f"Expected source='trainer', got '{normal_fields.get('source')}'"
                )

                # log_type_name must be a known event type
                event_type = normal_fields.get("log_type_name")
                assert event_type is not None, f"Missing log_type_name in {filepath}"

        # All 4 ranks must be represented
        assert all_ranks_seen == {0, 1, 2, 3}, f"Missing ranks: {all_ranks_seen}"

    def test_system_jsonl_has_fwd_bwd_and_optim_events(self):
        """System JSONL contains Forward/Backward and Optimizer span events."""
        assert self.returncode == 0, "Training failed"
        sys_dir = os.path.join(TOY_OUTPUT_DIR, "system_logs")
        rank0_file = os.path.join(sys_dir, "trainer_rank_0_system.jsonl")
        assert os.path.exists(rank0_file)

        records = _load_jsonl(rank0_file)
        event_types = {r["normal"].get("log_type_name") for r in records if "normal" in r}

        assert "fwd_bwd_start" in event_types, f"Missing fwd_bwd_start. Events: {event_types}"
        assert "fwd_bwd_end" in event_types, f"Missing fwd_bwd_end. Events: {event_types}"
        assert "optim_start" in event_types, f"Missing optim_start. Events: {event_types}"
        assert "optim_end" in event_types, f"Missing optim_end. Events: {event_types}"

    def test_system_jsonl_step_numbers(self):
        """System JSONL events cover all 20 training steps."""
        assert self.returncode == 0, "Training failed"
        sys_dir = os.path.join(TOY_OUTPUT_DIR, "system_logs")
        rank0_file = os.path.join(sys_dir, "trainer_rank_0_system.jsonl")
        records = _load_jsonl(rank0_file)

        steps_seen = {r["int"]["step"] for r in records if "step" in r.get("int", {})}
        for step in range(1, 21):
            assert step in steps_seen, f"Step {step} missing from system JSONL"


    def _parse_mp_console_losses(self):
        """Parse loss values from MetricsProcessor colored console output."""
        clean = re.sub(r"\x1b\[[0-9;]*m", "", self.output)
        losses = []
        for line in clean.split("\n"):
            m = re.search(r"step:\s*(\d+)\s+loss:\s+([\d.]+)", line)
            if m:
                losses.append((int(m.group(1)), float(m.group(2))))
        return losses

    def test_tensor_metric_loss_logged(self):
        """Tensor metric reduction produces loss values on log steps."""
        assert self.returncode == 0, "Training failed"
        tensor_losses = self._parse_mp_console_losses()
        assert len(tensor_losses) >= 3, (
            f"Expected >= 3 tensor metric loss entries, got {len(tensor_losses)}"
        )
        for step, loss in tensor_losses:
            assert loss > 0, f"Non-positive tensor loss at step {step}: {loss}"
        assert tensor_losses[-1][1] < tensor_losses[0][1], (
            f"Tensor loss not decreasing: {tensor_losses[0]} -> {tensor_losses[-1]}"
        )

    def test_mp_console_output_matches_table(self):
        """MetricsProcessor console loss matches the per-step table loss."""
        assert self.returncode == 0, "Training failed"
        mp_losses = dict(self._parse_mp_console_losses())
        table_losses = dict(_parse_loss_from_console(self.output))
        for step, mp_loss in mp_losses.items():
            if step in table_losses:
                assert abs(mp_loss - table_losses[step]) < 0.01, (
                    f"Step {step}: MP loss {mp_loss:.5f} != table loss {table_losses[step]:.5f}"
                )

    def test_experiment_jsonl_files_created(self):
        """One experiment JSONL file per rank (4 total)."""
        assert self.returncode == 0, "Training failed"
        exp_dir = os.path.join(TOY_OUTPUT_DIR, "experiment_logs")
        if not os.path.exists(exp_dir):
            pytest.skip("No experiment_logs dir yet")
        jsonl_files = sorted(glob.glob(os.path.join(exp_dir, "*.jsonl")))
        assert len(jsonl_files) == 4, f"Expected 4 experiment JSONL files, got {len(jsonl_files)}"

    def test_experiment_jsonl_has_expected_keys(self):
        """Experiment JSONL contains expected metric keys with correct fields."""
        assert self.returncode == 0, "Training failed"
        exp_dir = os.path.join(TOY_OUTPUT_DIR, "experiment_logs")
        if not os.path.exists(exp_dir):
            pytest.skip("No experiment_logs dir yet")

        rank0_files = sorted(glob.glob(os.path.join(exp_dir, "*rank_0*.jsonl")))
        assert len(rank0_files) >= 1, "No rank 0 experiment JSONL"

        records = _load_jsonl(rank0_files[0])
        assert len(records) > 0, "Empty experiment JSONL"

        keys_seen = set()
        for record in records:
            assert "step" in record, f"Missing step: {record}"
            assert "rank" in record, f"Missing rank: {record}"
            assert record["source"] == "trainer"
            assert record["rank"] == 0
            if "key" in record:
                keys_seen.add(record["key"])

        expected_keys = {
            "trainer/grad_norm_max",
            "trainer/learning_rate_mean",
            "trainer/throughput/tps_mean",
            "trainer/memory/max_reserved_gib_max",
        }
        for key in expected_keys:
            assert key in keys_seen, f"Missing key '{key}'. Found: {sorted(keys_seen)}"

    def test_experiment_jsonl_loss_values(self):
        """Loss values in experiment JSONL are positive and decreasing."""
        assert self.returncode == 0, "Training failed"
        exp_dir = os.path.join(TOY_OUTPUT_DIR, "experiment_logs")
        if not os.path.exists(exp_dir):
            pytest.skip("No experiment_logs dir yet")

        rank0_files = sorted(glob.glob(os.path.join(exp_dir, "*rank_0*.jsonl")))
        records = _load_jsonl(rank0_files[0])

        loss_by_step = {}
        for record in records:
            if record.get("key") == "trainer/loss_mean" and record.get("all_reduced"):
                loss_by_step[record["step"]] = record["value"]

        assert len(loss_by_step) >= 3, f"Expected >= 3 loss entries, got {len(loss_by_step)}"

        for step, loss in loss_by_step.items():
            assert loss > 0, f"Non-positive loss at step {step}: {loss}"

        steps = sorted(loss_by_step.keys())
        first_loss = loss_by_step[steps[0]]
        last_loss = loss_by_step[steps[-1]]
        assert last_loss < first_loss, (
            f"Loss not decreasing: step {steps[0]}={first_loss:.4f}, step {steps[-1]}={last_loss:.4f}"
        )

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
