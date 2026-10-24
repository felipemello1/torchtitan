# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a host-side span's wall time goes: CPU work, waiting for a CPU, or blocked.

Example::

    with sl.log_trace_span("preprocess_inputs") as span:
        probe = ThreadProbe(device)
        input_dict = microbatch.to_input_dict(device, non_blocking=True)
        probe.mark("to_device_ms")
        ...
        span.context = probe.finish()

``finish()`` returns the marked sub-span walls plus:

- ``gpu_idle_at_start``: the current CUDA stream had no pending work at the start (None off
  CUDA or during graph capture). ``torch.cuda.Stream.query`` does not block.
- ``thread_cpu_ms``: CPU time of this thread over the span. CUDA's default sync spins while
  waiting for the GPU, so it can include a GPU wait.
- ``sched_wait_ms``: time this thread was runnable but waiting for a CPU
  (``/proc/thread-self/schedstat``; None where it is not available).

The rest of the wall time the thread was blocked: a sleeping CUDA sync, the GIL, page faults.
"""

import time

import torch

_SCHEDSTAT = "/proc/thread-self/schedstat"


def _sched_wait_ns() -> int | None:
    try:
        with open(_SCHEDSTAT, "rb") as f:
            return int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _gpu_idle(device: torch.device) -> bool | None:
    if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return None
    return torch.cuda.current_stream(device).query()


class ThreadProbe:
    """Measures one span on the calling thread; a few microseconds, no device sync."""

    def __init__(self, device: torch.device) -> None:
        self._enabled = not torch.compiler.is_dynamo_compiling()
        if not self._enabled:
            return
        self._fields: dict[str, float | bool | None] = {
            "gpu_idle_at_start": _gpu_idle(device)
        }
        self._device = device
        self._sched_wait_start = _sched_wait_ns()
        self._thread_start = time.thread_time()
        self._start = self._last = time.perf_counter()

    def mark(self, name: str) -> None:
        """Record the wall time since the previous mark (or the start) as ``name``."""
        if not self._enabled:
            return
        now = time.perf_counter()
        self._fields[name] = round((now - self._last) * 1000, 3)
        self._last = now

    def gpu_idle(self, name: str) -> None:
        """Record whether the current CUDA stream has no pending work now, as ``name``."""
        if self._enabled:
            self._fields[name] = _gpu_idle(self._device)

    def finish(self) -> dict[str, float | bool | None] | None:
        """The fields, for the span's ``_end`` record (``log_trace_span.context``)."""
        if not self._enabled:
            return None
        self._fields["thread_cpu_ms"] = round(
            (time.thread_time() - self._thread_start) * 1000, 3
        )
        end = _sched_wait_ns()
        self._fields["sched_wait_ms"] = (
            None
            if end is None or self._sched_wait_start is None
            else round((end - self._sched_wait_start) / 1e6, 3)
        )
        return self._fields
