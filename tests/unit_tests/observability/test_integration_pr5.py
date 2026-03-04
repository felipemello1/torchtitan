# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration test: validates profiling annotations in toy_spmd.py."""

import json
import os

from torchtitan.observability import profile_annotation

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "torchtitan", "experiments", "observability", "outputs", "toy_spmd",
)


class TestProfilingIntegration:
    def test_profile_annotation_noop_when_not_profiling(self):
        """profile_annotation should be a no-op when no profiler active."""
        with profile_annotation("test_region"):
            x = 1 + 1
        assert x == 2

    def test_system_jsonl_has_fwd_bwd_span(self):
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        assert os.path.exists(sys_dir), f"Missing {sys_dir}"
        files = [f for f in os.listdir(sys_dir) if f.endswith(".jsonl")]
        assert files
        with open(os.path.join(sys_dir, files[0])) as f:
            events = [json.loads(line) for line in f if line.strip()]
        type_names = {e["normal"].get("log_type_name", "") for e in events if "normal" in e}
        assert "fwd_bwd_start" in type_names

    def test_system_jsonl_has_optim_span(self):
        sys_dir = os.path.join(OUTPUT_DIR, "system_logs")
        assert os.path.exists(sys_dir), f"Missing {sys_dir}"
        files = [f for f in os.listdir(sys_dir) if f.endswith(".jsonl")]
        with open(os.path.join(sys_dir, files[0])) as f:
            events = [json.loads(line) for line in f if line.strip()]
        type_names = {e["normal"].get("log_type_name", "") for e in events if "normal" in e}
        assert "optim_start" in type_names
