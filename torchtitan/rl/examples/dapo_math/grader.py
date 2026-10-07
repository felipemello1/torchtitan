# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Math-Verify scoring, run in worker processes with a hard timeout.

`MathVerifyPool` runs this file as a script in each worker process, so it must not
import torchtitan: importing `torchtitan.rl` loads vLLM (~9 s and ~1 GB per process).
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import connection

from math_verify import parse, verify

logger = logging.getLogger(__name__)

_BOXED_START = r"\boxed{"
# From <sys/prctl.h>: signal this process when the thread that started it exits.
_PR_SET_PDEATHSIG = 1


class MathVerifyPool:
    """Score answers in worker processes, so an answer that hangs Math-Verify cannot
    stall the caller.

    Each worker process scores one answer at a time. If a worker gives no score
    within `timeout_seconds`, the answer scores 0 and the worker is killed and
    replaced. Pool threads, not the caller's event loop, wait on the workers, so
    other scores keep flowing while one worker is stuck.

    Why processes: `\\boxed{2000^{2000^{2000}}}` makes sympy compute 2000**(2000**2000),
    one C call that does not finish and holds the GIL throughout. In-process, that means:
    1. Every other thread stops, including the event loop.
    2. A thread timeout (`PyThreadState_SetAsyncExc`) never fires: it is checked
       only between bytecodes.
    3. Math-Verify's SIGALRM timeout needs the main thread, and rollout workers score
       on a monarch actor thread.
    Killing the process is the only way out.

    Example:
        pool = MathVerifyPool(num_workers=4, timeout_seconds=5.0)
        await pool.score(r"Answer: \\boxed{34}", "34")  # 1.0
        await pool.score(r"Answer: \\boxed{2000^{2000^{2000}}}", "34")  # 0.0 after 5 s
    """

    def __init__(self, *, num_workers: int, timeout_seconds: float) -> None:
        self._timeout_seconds = timeout_seconds
        # Each thread drives at most one worker process, started on its first score.
        self._threads = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="math_verify"
        )
        self._thread_local = threading.local()

    async def score(self, response: str, ground_truth: str) -> float:
        """Return `score_math_response(response, ground_truth)`, or 0.0 on timeout."""
        return await asyncio.get_running_loop().run_in_executor(
            self._threads, self._score_in_worker, response, ground_truth
        )

    def _score_in_worker(self, response: str, ground_truth: str) -> float:
        worker = getattr(self._thread_local, "worker", None)
        if worker is None or worker.poll() is not None:
            worker = self._thread_local.worker = _start_worker()

        worker.stdin.write(json.dumps([response, ground_truth]) + "\n")
        worker.stdin.flush()
        # Unlike select(), `connection.wait` uses poll(), which handles fds above 1024.
        ready = connection.wait([worker.stdout], timeout=self._timeout_seconds)
        # An empty line means the worker died mid-score; treat it like a timeout.
        score_line = worker.stdout.readline() if ready else ""
        if score_line:
            return float(score_line)

        worker.kill()
        worker.wait()
        self._thread_local.worker = None
        logger.warning(
            "Math-Verify gave no score within %s seconds; killed its worker and "
            "assigned zero reward",
            self._timeout_seconds,
        )
        return 0.0


def score_math_response(response: str, ground_truth: str) -> float:
    """Score the final `\\boxed{}` expression with Math-Verify, in this process.

    No timeout: on an event loop, use `MathVerifyPool.score` instead.

    Args:
        response: Model response containing a boxed final answer.
        ground_truth: Expected answer from the dataset.

    Example:
        score_math_response(r"work\nAnswer: \boxed{34}", "34")  # 1.0
    """
    prediction = _last_boxed_expression(response)
    if prediction is None:
        return 0.0

    try:
        # Box the gold like the prediction: a bare `2\sqrt{3}` parses as 2, and a
        # bare `(1,2)` or `\pi/4` parses to nothing.
        gold = parse(_BOXED_START + ground_truth + "}", parsing_timeout=None)
        prediction = parse(prediction, parsing_timeout=None)
        return float(bool(gold) and verify(gold, prediction, timeout_seconds=None))
    except Exception:
        # Model output is untrusted; malformed LaTeX produces a zero reward
        # rather than failing the training loop.
        return 0.0


def _last_boxed_expression(text: str) -> str | None:
    """Return the last complete `\\boxed{...}` expression."""
    start = text.rfind(_BOXED_START)
    if start == -1:
        return None

    answer_start = start + len(_BOXED_START)
    depth = 1
    for index, char in enumerate(text[answer_start:], start=answer_start):
        depth += (char == "{") - (char == "}")
        if depth == 0:
            return text[start : index + 1]
    return None


def _start_worker() -> subprocess.Popen[str]:
    """Start a process running `_serve`; return once it is ready to score."""
    worker = subprocess.Popen(
        [sys.executable, __file__],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    # Wait out the interpreter start, so it does not count against the first timeout.
    if worker.stdout.readline() != "ready\n":
        raise RuntimeError(
            f"Math-Verify worker exited during startup with code {worker.wait()}"
        )
    return worker


def _serve() -> None:
    """Worker loop: read `[response, ground_truth]` JSON lines, write one score per line."""
    # SIGKILL this worker if its parent dies: a stuck score never sees stdin close.
    ctypes.CDLL(None).prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    # Library prints go to stderr, so stdout carries only scores. (ANTLR prints a
    # warning on every parse when its runtime and generated parser versions differ.)
    scores_out, sys.stdout = sys.stdout, sys.stderr
    print("ready", file=scores_out, flush=True)
    for line in sys.stdin:
        response, ground_truth = json.loads(line)
        print(score_math_response(response, ground_truth), file=scores_out, flush=True)


if __name__ == "__main__":
    _serve()
