# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Profiling: GPU kernel tracing, CUDA memory snapshots, host memory, NSys.

Orchestrates multiple sub-profilers (TorchProfiler, MemSnapshot, HostMemory,
NSys) via a single ``Profiler`` context manager. Supports periodic and
on-demand triggering via file sentinels.

Key design:
- ``profile_annotation(name)`` wraps code with record_function + NVTX.
  Always calls record_function (zero overhead when profiler is inactive).
- ``set_profiling_step`` controls NVTX push/pop in profile_annotation.
- ``dist.barrier()`` in cleanup prevents NCCL timeout when profiler stops.
"""



import contextlib
import copy
import logging
import os
import pickle
import time
import tracemalloc
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerAction

from torchtitan.config.configurable import Configurable
from torchtitan.observability.logging_boundary import EveryNSteps

logger = logging.getLogger(__name__)

PROFILING_TRIGGER = "profiling"


# ---------------------------------------------------------------------------
# Profiling annotations
# ---------------------------------------------------------------------------

_is_profiling_step: bool = False


def set_profiling_step(is_profiling: bool) -> None:
    """Set whether the current step is a profiling step.

    Called by the profiler at the beginning of each step.
    When True, profile_annotation will use torch.profiler.record_function.
    When False, profile_annotation will use nullcontext for zero overhead.
    """
    global _is_profiling_step
    _is_profiling_step = is_profiling


def is_profiling_step() -> bool:
    """Check if the current step is a profiling step."""
    return _is_profiling_step


@contextlib.contextmanager
def profile_annotation(name: str):
    """Context manager for profiling annotations.

    Always wraps with record_function — it is zero-overhead when torch.profiler
    is not actively recording, and Dynamo traces through it natively (no graph
    break). NVTX is added only when profiling AND not compiling.

    IMPORTANT: We do NOT branch on _is_profiling_step here. That global causes
    Dynamo guard failures when it flips from False to True, forcing recompilation
    of every compiled block. record_function is safe to always call.

    Args:
        name: The annotation name shown in traces.
    """
    with torch.profiler.record_function(name):
        if _is_profiling_step and not torch._dynamo.is_compiling():
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if _is_profiling_step and not torch._dynamo.is_compiling():
                torch.cuda.nvtx.range_pop()


# ---------------------------------------------------------------------------
# TriggerResult + TriggerView
# ---------------------------------------------------------------------------


@dataclass
class TriggerResult:
    """Result of checking a single trigger (e.g., "profiling" file sentinel)."""

    should_trigger: bool = False
    trigger_info: str = ""


class TriggerView:
    """Synchronize trigger states across distributed ranks.

    Only role_rank 0 checks for file triggers. ``agree_across_ranks`` uses
    all_reduce(MAX) so that if ANY rank detected a trigger, ALL ranks see it.
    """

    @staticmethod
    def agree_across_ranks(
        local_view: dict[str, TriggerResult], group=None
    ) -> dict[str, TriggerResult]:
        """All-reduce local trigger states so all ranks agree."""
        trigger_indexes = {
            i: name for i, name in enumerate(sorted(local_view.keys()))
        }
        trigger_states = [
            1 if local_view[name].should_trigger else 0
            for name in (trigger_indexes[i] for i in range(len(trigger_indexes)))
        ]

        trigger_tensor = torch.tensor(trigger_states, dtype=torch.int32)
        if torch.cuda.is_available():
            trigger_tensor = trigger_tensor.cuda()
        if dist.is_initialized():
            dist.all_reduce(trigger_tensor, op=dist.ReduceOp.MAX, group=group)
        trigger_states = trigger_tensor.tolist()

        global_view = copy.deepcopy(local_view)
        for i, state in enumerate(trigger_states):
            global_view[trigger_indexes[i]].should_trigger = bool(state)
        return global_view


# ---------------------------------------------------------------------------
# FileBasedTriggerWatcher
# ---------------------------------------------------------------------------


class FileBasedTriggerWatcher:
    """Watches for trigger sentinel files in a filesystem directory.

    Only role_rank 0 reads filesystem (avoids N ranks hammering NFS).
    Uses POSIX atomic rename for race-free ownership claim.
    Background ThreadPoolExecutor(1) for non-blocking file detection.

    Usage: touch {output_dir}/triggers/profiling
    """

    def __init__(
        self,
        global_rank: int,
        role_rank: int,
        output_dir: str,
        trigger_names: list[str] | None = None,
    ):
        self.global_rank = global_rank
        self.role_rank = role_rank
        self.trigger_parent_path = Path(output_dir) / "triggers"
        self.trigger_names = trigger_names or [PROFILING_TRIGGER]

        self.detect_future: Future = Future()
        self.detect_future.set_result({})

        if self.can_detect_triggers:
            self.executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"trigger_watcher_{role_rank}"
            )
            self.executor.submit(self._setup_trigger_parent_folder)

    @property
    def can_detect_triggers(self) -> bool:
        return self.role_rank == 0

    def local_view(self) -> dict[str, TriggerResult]:
        """Get the local view of detected triggers."""
        local_view = {name: TriggerResult() for name in self.trigger_names}
        if self.can_detect_triggers:
            for name, info in self.detect_future.result().items():
                local_view[name] = TriggerResult(should_trigger=True, trigger_info=info)
            self.detect_future = self.executor.submit(self._check_dir_for_triggers)
        return local_view

    def close(self):
        if self.can_detect_triggers:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def _setup_trigger_parent_folder(self):
        self.trigger_parent_path.mkdir(exist_ok=True, parents=True)
        try:
            self.trigger_parent_path.chmod(0o777)
        except PermissionError:
            logger.warning(f"Could not set permissions on {self.trigger_parent_path}")

    def _check_dir_for_triggers(self) -> dict[str, str]:
        try:
            all_files = os.listdir(self.trigger_parent_path)
        except FileNotFoundError:
            return {}

        found = {}
        for name in self.trigger_names:
            if name in all_files:
                trigger_path = self.trigger_parent_path / name
                try:
                    renamed = trigger_path.with_suffix(f".{self.global_rank}")
                    trigger_path.replace(renamed)
                    with open(renamed) as f:
                        found[name] = f.read()
                    renamed.unlink()
                    logger.warning(f"Detected trigger {name} ({trigger_path})")
                except (FileNotFoundError, PermissionError, OSError) as e:
                    logger.warning(f"Failed to process trigger {name}: {e}")
        return found


# ---------------------------------------------------------------------------
# TriggerableSchedule
# ---------------------------------------------------------------------------


class TriggerableSchedule:
    """Custom profiler schedule with periodic + on-demand triggering.

    Warmup is needed because CUPTI (CUDA profiling API) needs a few steps to
    stabilize before it produces accurate kernel timing. The schedule
    automatically starts warmup early: if the policy says "profile at step 100"
    and warmup=2, then step 98→WARMUP, 99→WARMUP, 100→RECORD_AND_SAVE.

    On-demand triggering (via ``triggered=True``) starts a new warmup→record
    cycle immediately, regardless of the policy.
    """

    def __init__(self, warmup: int, active: int, profile_policy):
        self.warmup = warmup
        self.active = active
        self.profile_policy = profile_policy
        self.triggered = False
        self._cycle_start_step: int | None = None
        self.last_action: ProfilerAction = ProfilerAction.NONE

        assert warmup >= 0 and active > 0

        additional_steps = getattr(profile_policy, "additional_steps", None)
        if additional_steps is not None:
            invalid = {s for s in additional_steps if s < warmup}
            if invalid:
                raise ValueError(
                    f"Cannot profile at steps {sorted(invalid)} with warmup={warmup}."
                )

    def _start_cycle(self, step: int) -> ProfilerAction:
        self._cycle_start_step = step
        if self.warmup > 0:
            return ProfilerAction.WARMUP
        elif self.active == 1:
            self._cycle_start_step = None
            return ProfilerAction.RECORD_AND_SAVE
        else:
            return ProfilerAction.RECORD

    def __call__(self, step: int) -> ProfilerAction:
        if step < 0:
            return ProfilerAction.NONE

        if self._cycle_start_step is not None:
            self.triggered = False
            relative = step - self._cycle_start_step
            if relative < self.warmup:
                action = ProfilerAction.WARMUP
            elif relative < self.warmup + self.active:
                if relative == self.warmup + self.active - 1:
                    self._cycle_start_step = None
                    action = ProfilerAction.RECORD_AND_SAVE
                else:
                    action = ProfilerAction.RECORD
            else:
                self._cycle_start_step = None
                action = ProfilerAction.NONE
            self.last_action = action
            return action

        if self.triggered:
            self.triggered = False
            action = self._start_cycle(step)
            self.last_action = action
            return action

        target_step = step + self.warmup
        if self.profile_policy(target_step):
            action = self._start_cycle(step)
            self.last_action = action
            return action

        self.last_action = ProfilerAction.NONE
        return ProfilerAction.NONE


# ProfilerConfig is defined as Profiler.Config below (Configurable pattern).


# ---------------------------------------------------------------------------
# MemSnapshotProfiler
# ---------------------------------------------------------------------------


class MemSnapshotProfiler:
    """CUDA memory snapshot profiler.

    Records memory allocation history and dumps pickle files for offline
    visualization via https://pytorch.org/memory_viz. Includes FX trace
    augmentation for torch.compile memory debugging. An OOM observer is
    attached automatically to capture a snapshot when GPU memory is exhausted.
    """

    def __init__(
        self,
        output_dir: str,
        max_entries: int = 1000000,
        start_step: int = 0,
        stop_step: int = 3,
    ):
        self.output_dir = output_dir
        self.max_entries = max_entries
        self.start_step = start_step
        self.stop_step = stop_step
        self._step_num = 0
        self._recording = False

        self.snapshot_dir = os.path.join(output_dir, "memory_snapshot")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Attach OOM observer
        try:
            torch._C._cuda_attach_out_of_memory_observer(self._oom_observer)
        except AttributeError:
            pass

    def _oom_observer(self, device, alloc, device_alloc, device_free):
        """Dump snapshot on OOM for debugging."""
        try:
            self._dump_snapshot(suffix="_oom")
        except Exception as e:
            logger.warning(f"Failed to dump OOM snapshot: {e}")

    def step(self, step_num: int | None = None):
        """Advance profiler by one step."""
        if step_num is not None:
            self._step_num = step_num

        if self._step_num == self.start_step and not self._recording:
            self._start_recording()
        elif self._step_num >= self.stop_step and self._recording:
            self._dump_snapshot()
            self._stop_recording()
        elif self._recording and self._step_num > self.start_step:
            self._dump_snapshot()

    def trigger_profiling(self):
        """On-demand snapshot."""
        if not self._recording:
            self._start_recording()
        self._dump_snapshot(suffix="_triggered")

    def _start_recording(self):
        torch.cuda.memory._record_memory_history(
            max_entries=self.max_entries, stacks="python"
        )
        self._recording = True
        logger.info(f"CUDA memory recording started at step {self._step_num}")

    def _stop_recording(self):
        torch.cuda.memory._record_memory_history(enabled=None)
        self._recording = False
        logger.info(f"CUDA memory recording stopped at step {self._step_num}")

    def _dump_snapshot(self, suffix: str = ""):
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            try:
                snapshot = torch.cuda.memory._snapshot(augment_with_fx_traces=True)
            except TypeError:
                snapshot = torch.cuda.memory._snapshot()
            step_dir = os.path.join(self.snapshot_dir, f"step_{self._step_num}")
            os.makedirs(step_dir, exist_ok=True)
            path = os.path.join(step_dir, f"rank_{rank}{suffix}.pickle")
            with open(path, "wb") as f:
                pickle.dump(snapshot, f)
            logger.info(f"CUDA memory snapshot saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to dump memory snapshot: {e}")


# ---------------------------------------------------------------------------
# HostMemoryProfiler
# ---------------------------------------------------------------------------


class HostMemoryProfiler:
    """Host/CPU memory profiler using Python's tracemalloc.

    Takes periodic snapshots of Python memory allocations for diagnosing
    memory leaks in data loaders, object accumulation, etc.
    """

    def __init__(
        self,
        output_dir: str,
        interval: int = 120,
        frames: int = 10,
        traces: int = 10,
    ):
        self.output_dir = output_dir
        self.interval = interval
        self.frames = frames
        self.traces = traces
        self._active = False
        self.snapshot_dir = os.path.join(output_dir, "host_memory")
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def start(self):
        tracemalloc.start(self.frames)
        self._active = True
        logger.info("Host memory profiler started (tracemalloc)")

    def step(self, step_num: int):
        if not self._active:
            self.start()
        if step_num > 0 and step_num % self.interval == 0:
            self._take_snapshot(step_num)

    def trigger_profiling(self):
        if not self._active:
            self.start()
        self._take_snapshot(suffix="_triggered")

    def stop(self):
        if self._active:
            tracemalloc.stop()
            self._active = False
            logger.info("Host memory profiler stopped")

    def _take_snapshot(self, step_num: int = 0, suffix: str = ""):
        try:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics("lineno")
            rank = dist.get_rank() if dist.is_initialized() else 0
            path = os.path.join(
                self.snapshot_dir, f"step_{step_num}_rank_{rank}{suffix}.txt"
            )
            with open(path, "w") as f:
                f.write(f"Top {self.traces} allocations at step {step_num}:\n")
                for stat in stats[: self.traces]:
                    f.write(str(stat) + "\n")
            logger.info(f"Host memory snapshot saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to take host memory snapshot: {e}")


# ---------------------------------------------------------------------------
# NSysProfiler
# ---------------------------------------------------------------------------


class NSysProfiler:
    """NSys profiler wrapper.

    Controls cudaProfilerStart/Stop + NVTX ranges for nsys traces.
    Mutually exclusive with TorchProfiler. Enable via
    ``ProfilerConfig(enable_nsys=True, nsys_start_step=0, nsys_stop_step=3)``.
    """

    def __init__(self, start_step: int = 0, stop_step: int = 3):
        self.start_step = start_step
        self.stop_step = stop_step
        self._active = False

    def step(self, step_num: int):
        if step_num < self.start_step or step_num > self.stop_step:
            return
        if step_num == self.start_step:
            logger.info(f"Starting NSys profiling at step {step_num}")
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f"Step {step_num}")
            self._active = True
        elif step_num == self.stop_step:
            logger.info(f"Stopping NSys profiling before step {step_num}")
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            self._active = False
        else:
            # Between start and stop: pop previous step's range, push new one
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(f"Step {step_num}")


# ---------------------------------------------------------------------------
# Profiler orchestrator
# ---------------------------------------------------------------------------


class Profiler(Configurable):
    """Orchestrator that composes all sub-profilers.

    Context manager for lifecycle. step() fans out to all sub-profilers.
    trigger_profiling() triggers on-demand profiling for all.

    Usage::

        cfg = Profiler.Config(enable_profiling=True, profile_freq=100)
        with cfg.build(output_dir="output", global_rank=rank) as prof:
            for step in range(num_steps):
                prof.step(step)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Configuration for the Profiler orchestrator.

        Adapt to your use case. Example for periodic GPU kernel profiling::

            Profiler.Config(enable_profiling=True, profile_freq=100, profiler_warmup=2)
        """

        enable_profiling: bool = False
        profile_freq: int = 10  # Profile every N steps
        profiler_warmup: int = 2  # CUPTI warmup steps before recording
        profiler_active: int = 1  # Steps to record per profiling cycle
        with_stack: bool = False  # Capture Python call stacks (slower)
        enable_memory_snapshot: bool = False
        memory_snapshot_max_entries: int = 1000000
        memory_snapshot_start_step: int = 3
        memory_snapshot_stop_step: int = 5
        enable_host_memory_profiler: bool = False
        host_memory_interval: int = 120
        enable_nsys: bool = False
        nsys_start_step: int = 0
        nsys_stop_step: int = 3

    def __init__(
        self,
        config: Config,
        *,
        output_dir: str,
        global_rank: int = 0,
        role_rank: int = 0,
    ):
        self.config = config
        self.output_dir = output_dir
        self._within_context = False
        self._torch_profiler = None
        self._torch_profiler_schedule: TriggerableSchedule | None = None

        # File-based trigger watcher
        self._trigger_watcher = FileBasedTriggerWatcher(
            global_rank=global_rank,
            role_rank=role_rank,
            output_dir=output_dir,
        )

        # Sub-profilers (created lazily or at init based on config)
        self._mem_profiler: MemSnapshotProfiler | None = None
        if config.enable_memory_snapshot:
            self._mem_profiler = MemSnapshotProfiler(
                output_dir=output_dir,
                max_entries=config.memory_snapshot_max_entries,
                start_step=config.memory_snapshot_start_step,
                stop_step=config.memory_snapshot_stop_step,
            )

        self._host_mem_profiler: HostMemoryProfiler | None = None
        if config.enable_host_memory_profiler:
            self._host_mem_profiler = HostMemoryProfiler(
                output_dir=output_dir,
                interval=config.host_memory_interval,
            )

        self._nsys_profiler: NSysProfiler | None = None
        if config.enable_nsys:
            if config.enable_profiling:
                raise ValueError("NSys and torch profiling cannot both be enabled")
            self._nsys_profiler = NSysProfiler(
                start_step=config.nsys_start_step,
                stop_step=config.nsys_stop_step,
            )

    @property
    def torch_profiler(self):
        """Lazy creation of torch profiler."""
        if self._torch_profiler is None and self.config.enable_profiling:
            self._torch_profiler = self._create_torch_profiler()
        return self._torch_profiler

    def _create_torch_profiler(self) -> torch.profiler.profile | None:
        cfg = self.config
        trace_dir = os.path.join(self.output_dir, "profiling", "traces")
        os.makedirs(trace_dir, exist_ok=True)

        rank = dist.get_rank() if dist.is_initialized() else 0

        def trace_handler(prof):
            actual_step = prof.step_num - 1
            curr_dir = os.path.join(trace_dir, f"iteration_{actual_step}")
            os.makedirs(curr_dir, exist_ok=True)
            output_file = os.path.join(curr_dir, f"rank{rank:06d}_trace.json.gz")
            logger.info(f"Dumping trace at step {actual_step}")
            t0 = time.monotonic()
            prof.export_chrome_trace(output_file)
            logger.info(f"Trace dump took {time.monotonic() - t0:.2f}s")

        policy = EveryNSteps(every_n=cfg.profile_freq)
        self._torch_profiler_schedule = TriggerableSchedule(
            warmup=cfg.profiler_warmup,
            active=cfg.profiler_active,
            profile_policy=policy,
        )

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        prof = torch.profiler.profile(
            activities=activities,
            schedule=self._torch_profiler_schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=cfg.with_stack,
        )
        prof.start()
        logger.info(f"Torch profiler started. Traces: {trace_dir}")
        return prof

    def __enter__(self):
        if self._within_context:
            raise RuntimeError("Already in profiler context.")
        self._within_context = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.cleanup()
        self._within_context = False

    def step(self, step: int) -> None:
        """Advance all profilers by one step."""
        # Check file-based triggers
        if dist.is_initialized():
            local_view = self._trigger_watcher.local_view()
            trigger_results = TriggerView.agree_across_ranks(local_view)
            if trigger_results.get(PROFILING_TRIGGER, TriggerResult()).should_trigger:
                self.trigger_profiling()

        # Fan out to sub-profilers
        if self.torch_profiler:
            self.torch_profiler.step_num = step
            self.torch_profiler.step()
            # FIX: set_profiling_step AFTER step(), only during RECORD/RECORD_AND_SAVE
            if self._torch_profiler_schedule:
                action = self._torch_profiler_schedule.last_action
                set_profiling_step(action in (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE))

        if self._mem_profiler:
            self._mem_profiler.step(step)
        if self._host_mem_profiler:
            self._host_mem_profiler.step(step)
        if self._nsys_profiler:
            self._nsys_profiler.step(step)

    def trigger_profiling(self) -> None:
        """Trigger on-demand profiling for all sub-profilers."""
        logger.info("Profiling triggered on-demand")
        if self._mem_profiler:
            self._mem_profiler.trigger_profiling()
        if self._host_mem_profiler:
            self._host_mem_profiler.trigger_profiling()
        if self._torch_profiler_schedule:
            self._torch_profiler_schedule.triggered = True

    def cleanup(self) -> None:
        """Stop all profilers and release resources."""
        if self._torch_profiler is not None:
            try:
                # FIX: barrier before stop prevents NCCL timeout
                if dist.is_initialized():
                    dist.barrier()
                self._torch_profiler.stop()
                logger.info("Torch profiler stopped")
            except Exception as e:
                logger.warning(f"Error stopping torch profiler: {e}")
            finally:
                self._torch_profiler = None
                set_profiling_step(False)

        if self._host_mem_profiler:
            try:
                self._host_mem_profiler.stop()
            except Exception as e:
                logger.warning(f"Error stopping host memory profiler: {e}")

        self._trigger_watcher.close()
