# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for profiler output from toy_spmd.

Verifies trace files, memory snapshots, host memory snapshots, and
profile_annotation names. Run toy_spmd first to generate output files.
"""

import gzip
import json
import os
from glob import glob

import pytest

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "outputs", "toy_spmd"
)


@pytest.fixture
def profiling_dir():
    path = os.path.join(OUTPUT_DIR, "profiling", "traces")
    if not os.path.isdir(path):
        pytest.fail(
            f"No profiling traces at {path}. Run toy_spmd first to generate outputs."
        )
    return path


@pytest.fixture
def memory_snapshot_dir():
    path = os.path.join(OUTPUT_DIR, "memory_snapshot")
    if not os.path.isdir(path):
        pytest.fail(
            f"No memory snapshots at {path}. Run toy_spmd first to generate outputs."
        )
    return path


@pytest.fixture
def host_memory_dir():
    path = os.path.join(OUTPUT_DIR, "host_memory")
    if not os.path.isdir(path):
        pytest.fail(
            f"No host memory snapshots at {path}. Run toy_spmd first to generate outputs."
        )
    return path


class TestTorchProfilerTraces:
    def test_trace_files_at_expected_steps(self, profiling_dir):
        """Traces should exist at profile_freq steps (10, 20)."""
        iteration_dirs = sorted(os.listdir(profiling_dir))
        assert "iteration_10" in iteration_dirs
        assert "iteration_20" in iteration_dirs

    def test_trace_files_per_rank(self, profiling_dir):
        """Each iteration should have traces for all 4 ranks."""
        for step in ["iteration_10", "iteration_20"]:
            step_dir = os.path.join(profiling_dir, step)
            traces = glob(os.path.join(step_dir, "*.json.gz"))
            assert len(traces) == 4, f"Expected 4 rank traces at {step}, got {len(traces)}"

    def test_trace_contains_profile_annotations(self, profiling_dir):
        """Trace should contain forward_backward and optimizer annotations."""
        trace_path = glob(os.path.join(profiling_dir, "iteration_10", "*.json.gz"))[0]
        with gzip.open(trace_path, "rt") as f:
            trace = json.load(f)

        event_names = {e.get("name", "") for e in trace.get("traceEvents", [])}
        assert "forward_backward" in event_names, (
            f"Missing forward_backward annotation. Found: {sorted(n for n in event_names if not n.startswith('##'))[:20]}"
        )
        assert "optimizer" in event_names, (
            f"Missing optimizer annotation. Found: {sorted(n for n in event_names if not n.startswith('##'))[:20]}"
        )


class TestMemorySnapshot:
    def test_snapshot_files_at_expected_steps(self, memory_snapshot_dir):
        """Memory snapshots at steps 4 and 5 (start_step=3, stop_step=5)."""
        step_dirs = sorted(os.listdir(memory_snapshot_dir))
        assert "step_4" in step_dirs
        assert "step_5" in step_dirs

    def test_snapshot_files_per_rank(self, memory_snapshot_dir):
        for step in ["step_4", "step_5"]:
            step_dir = os.path.join(memory_snapshot_dir, step)
            files = glob(os.path.join(step_dir, "*.pickle"))
            assert len(files) == 4, f"Expected 4 rank snapshots at {step}, got {len(files)}"


class TestHostMemory:
    def test_host_memory_files_exist(self, host_memory_dir):
        """Host memory snapshots at interval steps (10, 20)."""
        files = glob(os.path.join(host_memory_dir, "*.txt"))
        assert len(files) >= 4, f"Expected at least 4 host memory files, got {len(files)}"

    def test_host_memory_has_content(self, host_memory_dir):
        files = glob(os.path.join(host_memory_dir, "*.txt"))
        for fp in files:
            with open(fp) as f:
                content = f.read()
            assert "Top" in content, f"Host memory file {fp} missing allocation data"
