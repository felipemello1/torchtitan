# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

from .types import Max, Metric

_P = ParamSpec("_P")
_T = TypeVar("_T")


class QueueWaitThreadPoolExecutor(ThreadPoolExecutor):
    """A `ThreadPoolExecutor` that measures how long jobs wait for a free thread, and how many jobs
    are running or waiting at once.

    The longest wait in a flush:
        up to tens of ms         a thread was free
        about a GC pause         a garbage collection delayed the start (no job starts during one)
        seconds beyond that      too few threads, e.g. all held by blocking calls

    The most jobs at once counts the waiting jobs too. So when jobs waited, this many threads would
    have run them all without waiting, and fewer may have been enough.

    Example, a pool of 1 thread:

        executor = QueueWaitThreadPoolExecutor("rollout_worker", max_workers=1)
        executor.submit(time.sleep, 2)
        executor.submit(print, "hi")  # starts when the sleep ends: it waited 2 s
        ...
        executor.flush()
        # -> [Metric("rollout_worker/executor_queue_wait_ms", Max(2000.0)),
        #     Metric("rollout_worker/num_executor_jobs_in_flight", Max(2.0))]
        executor.flush()
        # -> Max(0.0) and Max(0.0): each flush starts over

    A job still waiting at a flush counts with its wait so far, so a pool whose threads are all
    stuck still shows the wait.

    Args:
        role: The kind of process, used as the keys' first part: "rollout_worker" ->
            `rollout_worker/executor_queue_wait_ms`.
        max_workers: Threads in the pool.
    """

    def __init__(self, role: str, *, max_workers: int) -> None:
        super().__init__(max_workers=max_workers)
        self._wait_key = f"{role}/executor_queue_wait_ms"
        self._jobs_key = f"{role}/num_executor_jobs_in_flight"
        self._max_wait_s = 0.0
        # The submit time of each job that hasn't started, oldest first.
        self._submit_time_by_waiting_job: dict[object, float] = {}
        # Jobs submitted and not done yet (running or waiting), and the most at once since the last flush.
        self._num_jobs_in_flight = 0
        self._max_jobs_in_flight = 0
        # Jobs record their wait on the pool's threads, and `flush` reads it on the caller's.
        self._lock = threading.Lock()

    def submit(
        self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        # Identifies this job in `_submit_time_by_waiting_job`.
        job_key = object()

        def record_wait_then_run() -> _T:
            with self._lock:
                submit_time = self._submit_time_by_waiting_job.pop(job_key)
                self._max_wait_s = max(self._max_wait_s, time.monotonic() - submit_time)
            return fn(*args, **kwargs)

        # Store the submit time before the job is queued, so the job finds it when it starts. The base
        # `submit` runs outside our lock: `shutdown(cancel_futures=True)` holds the base class's lock
        # while the cancel runs `_on_job_done`, which takes ours. So holding ours here while the base
        # `submit` waits for its lock could deadlock.
        with self._lock:
            self._submit_time_by_waiting_job[job_key] = time.monotonic()
            self._num_jobs_in_flight += 1
            self._max_jobs_in_flight = max(
                self._max_jobs_in_flight, self._num_jobs_in_flight
            )
        try:
            future = super().submit(record_wait_then_run)
        except RuntimeError:  # the pool is shut down
            with self._lock:
                del self._submit_time_by_waiting_job[job_key]
                self._num_jobs_in_flight -= 1
            raise
        # The done callback runs even for a job cancelled before it started (its function never
        # runs), so every job is uncounted.
        future.add_done_callback(functools.partial(self._on_job_done, job_key))
        return future

    def _on_job_done(self, job_key: object, future: Future) -> None:
        with self._lock:
            self._num_jobs_in_flight -= 1
            # The key is still here only for a job cancelled before it started.
            self._submit_time_by_waiting_job.pop(job_key, None)

    def flush(self) -> list[Metric]:
        """Return the longest wait and the most jobs at once since the last flush, then reset."""
        with self._lock:
            now = time.monotonic()
            oldest_submit_time = next(
                iter(self._submit_time_by_waiting_job.values()), now
            )
            # A job still waiting counts with its wait so far.
            max_wait_s = max(self._max_wait_s, now - oldest_submit_time)
            self._max_wait_s = 0.0
            max_jobs_in_flight = self._max_jobs_in_flight
            # The jobs still in flight count toward the next flush.
            self._max_jobs_in_flight = self._num_jobs_in_flight
        return [
            Metric(self._wait_key, Max(1000 * max_wait_s)),
            Metric(self._jobs_key, Max(float(max_jobs_in_flight))),
        ]
