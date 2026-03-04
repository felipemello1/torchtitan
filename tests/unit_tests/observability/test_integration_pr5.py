# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test for PR5: validates profiling annotations in toy_spmd.py.

Tests that:
- profile_annotation works as a no-op when no profiler is active
- System JSONL still has expected spans from earlier PRs

Prerequisites:
    torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd
"""

import json
import os

from torchtitan.observability import profile_annotation

OUTPUT_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "torchtitan", "experiments", "observability", "outputs", "toy_spmd",
))

SYS_LOG_DIR = os.path.join(OUTPUT_DIR, "system_logs")


class TestPR5Integration:
    def test_profile_annotation_noop_when_not_profiling(self):
        """profile_annotation should be a no-op (bare yield) when no profiler active."""
        with profile_annotation("test_region"):
            x = 1 + 1
        assert x == 2

    def test_system_jsonl_has_fwd_bwd_span(self):
        """System JSONL should still have Forward/Backward spans (PR5 doesn't break PR1)."""
        assert os.path.isdir(SYS_LOG_DIR), (
            f"system_logs dir not found: {SYS_LOG_DIR}. "
            "Run: torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd"
        )
        files = [f for f in os.listdir(SYS_LOG_DIR) if f.endswith(".jsonl")]
        assert files, "No system JSONL files"
        with open(os.path.join(SYS_LOG_DIR, files[0])) as f:
            events = [json.loads(line) for line in f if line.strip()]
        type_names = {
            e["normal"].get("log_type_name", "")
            for e in events if "normal" in e
        }
        assert "fwd_bwd_start" in type_names, f"Missing fwd_bwd span. Types: {type_names}"

    def test_previous_prs_still_work(self):
        """System logs and experiment logs should exist."""
        assert os.path.isdir(OUTPUT_DIR), (
            f"Output dir not found: {OUTPUT_DIR}. "
            "Run: torchrun --nproc_per_node=4 -m torchtitan.experiments.observability.toy_spmd"
        )
        assert os.path.isdir(os.path.join(OUTPUT_DIR, "system_logs")), "system_logs missing"
        assert os.path.isdir(os.path.join(OUTPUT_DIR, "experiment_logs")), "experiment_logs missing"
