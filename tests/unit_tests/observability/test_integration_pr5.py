# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR5: validates profiling annotations in toy_spmd.py.

Prerequisites:
    torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
"""

import json
import os

import pytest

from torchtitan.observability import profile_annotation

OUTPUT_DIR = "/tmp/toy_spmd_output"


class TestPR5Integration:
    def test_profile_annotation_noop_when_not_profiling(self):
        """profile_annotation should be a no-op (bare yield) when no profiler active."""
        with profile_annotation("test_region"):
            x = 1 + 1
        assert x == 2  # just verifying no exception

    def test_system_jsonl_has_fwd_bwd_span(self):
        """System JSONL should still have Forward/Backward spans (PR5 doesn't break PR1)."""
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        if not os.path.exists(sys_dir):
            pytest.skip("Run toy_spmd.py first")
        files = [f for f in os.listdir(sys_dir) if f.endswith(".jsonl")]
        assert files, "No system JSONL files"
        with open(os.path.join(sys_dir, files[0])) as f:
            events = [json.loads(line) for line in f if line.strip()]
        # JSONL uses 4-column Scuba format: normal.log_type_name has event names
        type_names = set()
        for e in events:
            if "normal" in e:
                type_names.add(e["normal"].get("log_type_name", ""))
        assert "fwd_bwd_start" in type_names, (
            f"Missing fwd_bwd span. Type names: {type_names}"
        )

    def test_system_jsonl_has_optim_span(self):
        """System JSONL should have Optimizer spans."""
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        if not os.path.exists(sys_dir):
            pytest.skip("Run toy_spmd.py first")
        files = [f for f in os.listdir(sys_dir) if f.endswith(".jsonl")]
        with open(os.path.join(sys_dir, files[0])) as f:
            events = [json.loads(line) for line in f if line.strip()]
        type_names = set()
        for e in events:
            if "normal" in e:
                type_names.add(e["normal"].get("log_type_name", ""))
        assert "optim_start" in type_names, (
            f"Missing optim span. Type names: {type_names}"
        )

    def test_previous_prs_still_work(self):
        """All prior PR outputs should exist."""
        if not os.path.exists(OUTPUT_DIR):
            pytest.skip("Run toy_spmd.py first")
        assert os.path.exists(os.path.join(OUTPUT_DIR, "system_logs")), "PR1 missing"
        assert os.path.exists(os.path.join(OUTPUT_DIR, "losses.json")), "PR0 missing"
        assert os.path.exists(os.path.join(OUTPUT_DIR, "experiment_logs")), "PR4 missing"
