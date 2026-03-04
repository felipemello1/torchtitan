# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for profiling.py."""

from unittest.mock import MagicMock

import pytest
from torch.profiler import ProfilerAction

from torchtitan.observability.profiling import (
    FileBasedTriggerWatcher,
    is_profiling_step,
    MemSnapshotProfiler,
    NSysProfiler,
    profile_annotation,
    Profiler,
    set_profiling_step,
    TriggerableSchedule,
    TriggerResult,
)


# ---------------------------------------------------------------------------
# profile_annotation / set_profiling_step / is_profiling_step
# ---------------------------------------------------------------------------


class TestProfilingAnnotation:
    def setup_method(self):
        set_profiling_step(False)

    def test_not_profiling_is_zero_overhead(self):
        """When not profiling, profile_annotation is a bare yield."""
        set_profiling_step(False)
        with profile_annotation("test"):
            pass  # Should not raise

    def test_profiling_step_flag(self):
        assert not is_profiling_step()
        set_profiling_step(True)
        assert is_profiling_step()
        set_profiling_step(False)
        assert not is_profiling_step()

    @pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
    def test_profiling_active_uses_record_function(self):
        """When profiling, profile_annotation wraps with record_function."""
        set_profiling_step(True)
        # This should not raise even without an active profiler
        with profile_annotation("test_op"):
            pass
        set_profiling_step(False)


# ---------------------------------------------------------------------------
# TriggerableSchedule
# ---------------------------------------------------------------------------


class TestTriggerableSchedule:
    def test_basic_every_n_steps(self):
        policy = MagicMock(side_effect=lambda step: step > 0 and step % 10 == 0)
        schedule = TriggerableSchedule(warmup=2, active=1, profile_policy=policy)

        # Steps 0-7: NONE
        for s in range(8):
            assert schedule(s) == ProfilerAction.NONE, f"Step {s}"

        # Step 8: warmup for step 10 (8 + warmup=2 → target=10)
        assert schedule(8) == ProfilerAction.WARMUP
        assert schedule(9) == ProfilerAction.WARMUP
        assert schedule(10) == ProfilerAction.RECORD_AND_SAVE

    def test_zero_warmup(self):
        policy = MagicMock(side_effect=lambda step: step == 5)
        schedule = TriggerableSchedule(warmup=0, active=1, profile_policy=policy)
        assert schedule(5) == ProfilerAction.RECORD_AND_SAVE

    def test_triggered_profiling(self):
        policy = MagicMock(return_value=False)
        schedule = TriggerableSchedule(warmup=1, active=1, profile_policy=policy)
        schedule.triggered = True
        assert schedule(0) == ProfilerAction.WARMUP
        assert schedule(1) == ProfilerAction.RECORD_AND_SAVE
        assert schedule(2) == ProfilerAction.NONE

    def test_negative_step_returns_none(self):
        policy = MagicMock(return_value=True)
        schedule = TriggerableSchedule(warmup=0, active=1, profile_policy=policy)
        assert schedule(-1) == ProfilerAction.NONE

    def test_multiple_active_steps(self):
        policy = MagicMock(side_effect=lambda step: step == 5)
        schedule = TriggerableSchedule(warmup=0, active=3, profile_policy=policy)
        assert schedule(5) == ProfilerAction.RECORD
        assert schedule(6) == ProfilerAction.RECORD
        assert schedule(7) == ProfilerAction.RECORD_AND_SAVE
        assert schedule(8) == ProfilerAction.NONE

    def test_trigger_ignored_during_active_cycle(self):
        policy = MagicMock(side_effect=lambda step: step == 5)
        schedule = TriggerableSchedule(warmup=0, active=2, profile_policy=policy)
        assert schedule(5) == ProfilerAction.RECORD
        schedule.triggered = True  # Should be ignored
        assert schedule(6) == ProfilerAction.RECORD_AND_SAVE
        assert schedule.triggered is False

    def test_last_action_tracked(self):
        policy = MagicMock(side_effect=lambda step: step == 5)
        schedule = TriggerableSchedule(warmup=0, active=1, profile_policy=policy)
        schedule(4)
        assert schedule.last_action == ProfilerAction.NONE
        schedule(5)
        assert schedule.last_action == ProfilerAction.RECORD_AND_SAVE


# ---------------------------------------------------------------------------
# FileBasedTriggerWatcher
# ---------------------------------------------------------------------------


class TestFileBasedTriggerWatcher:
    def test_creates_trigger_directory(self, tmp_path):
        watcher = FileBasedTriggerWatcher(
            global_rank=0, role_rank=0, output_dir=str(tmp_path)
        )
        # Give the background thread a moment
        import time
        time.sleep(0.1)
        assert (tmp_path / "triggers").exists()
        watcher.close()

    def test_detects_trigger_file(self, tmp_path):
        watcher = FileBasedTriggerWatcher(
            global_rank=0, role_rank=0, output_dir=str(tmp_path)
        )
        import time
        time.sleep(0.1)

        # Create trigger file
        trigger_dir = tmp_path / "triggers"
        trigger_dir.mkdir(exist_ok=True)
        (trigger_dir / "profiling").write_text("test trigger")

        # First call: returns previous result (empty), submits new detection
        watcher.local_view()
        time.sleep(0.1)  # Wait for background thread to check
        # Second call: returns the detection result
        view = watcher.local_view()
        assert view["profiling"].should_trigger is True
        watcher.close()

    def test_non_rank0_does_not_detect(self, tmp_path):
        watcher = FileBasedTriggerWatcher(
            global_rank=1, role_rank=1, output_dir=str(tmp_path)
        )
        view = watcher.local_view()
        assert view["profiling"].should_trigger is False
        watcher.close()


# ---------------------------------------------------------------------------
# Profiler.Config
# ---------------------------------------------------------------------------


class TestProfilerConfig:
    def test_defaults(self):
        cfg = Profiler.Config()
        assert cfg.enable_profiling is False
        assert cfg.profile_freq == 10
        assert cfg.profiler_warmup == 2
        assert cfg.profiler_active == 1

    def test_custom(self):
        cfg = Profiler.Config(enable_profiling=True, profile_freq=5)
        assert cfg.enable_profiling is True
        assert cfg.profile_freq == 5


# ---------------------------------------------------------------------------
# Profiler orchestrator
# ---------------------------------------------------------------------------


class TestProfiler:
    def test_build_pattern(self):
        """Profiler.Config().build() returns a Profiler instance."""
        cfg = Profiler.Config()
        profiler = cfg.build(output_dir="/tmp/test_profiler")
        assert isinstance(profiler, Profiler)

    def test_context_manager(self):
        cfg = Profiler.Config()
        profiler = Profiler(config=cfg, output_dir="/tmp/test_profiler")
        with profiler:
            pass  # Should not raise

    def test_step_without_profiling(self):
        cfg = Profiler.Config()
        profiler = Profiler(config=cfg, output_dir="/tmp/test_profiler")
        with profiler:
            profiler.step(1)  # Should not raise (profiling disabled)

    def test_cleanup(self):
        cfg = Profiler.Config()
        profiler = Profiler(config=cfg, output_dir="/tmp/test_profiler")
        profiler.cleanup()  # Should not raise

    def test_nsys_and_torch_mutually_exclusive(self):
        cfg = Profiler.Config(enable_profiling=True, enable_nsys=True)
        with pytest.raises(ValueError, match="cannot both"):
            Profiler(config=cfg, output_dir="/tmp/test")

    def test_host_memory_profiler_enabled(self, tmp_path):
        cfg = Profiler.Config(enable_host_memory_profiler=True, host_memory_interval=5)
        profiler = Profiler(config=cfg, output_dir=str(tmp_path))
        assert profiler._host_mem_profiler is not None
        profiler.cleanup()

    def test_memory_snapshot_enabled(self, tmp_path):
        cfg = Profiler.Config(enable_memory_snapshot=True)
        profiler = Profiler(config=cfg, output_dir=str(tmp_path))
        assert profiler._mem_profiler is not None
        profiler.cleanup()
