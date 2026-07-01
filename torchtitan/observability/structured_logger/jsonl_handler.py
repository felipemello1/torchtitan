# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Default JSONL backend: formatter, file handler, and factory.

``TraceJsonlFormatter`` is also the base class for backend-specific
formatters (e.g. the Scuba formatter under ``fb/``).
"""

import datetime as dt
import itertools
import json
import logging
import os
import random
import socket
import string
import threading
from typing import Any

from torchtitan.observability.structured_logger.step_state import (
    get_relative_step,
    get_step,
    get_step_tags,
)
from torchtitan.observability.structured_logger.structured_logging import (
    ExtraFields,
    LogType,
    TraceEventsOnlyFilter,
)

console_logger: logging.Logger = logging.getLogger(__name__)

MAX_MESSAGE_SIZE: int = 1000


class TraceJsonlFormatter(logging.Formatter):
    """Format trace records as one JSON line per record.

    Per-process fields (rank, source, hostname, local_rank) are captured
    in ``__init__``; per-step fields (step, relative_step, step_tags)
    are pulled from :mod:`.step_state` at emit time.

    Subclass to enrich records with backend-specific fields.

    Example output (wrapped for readability)::

        {"global_rank": 0, "local_rank": 0, "source": "rl_trainer",
         "host_name": "devgpu001", "pid": 4242, "tid": 4242,
         "step": 5, "relative_step": 5, "step_tags": ["gc"],
         "time_us": 1709500000123456, "log_type": "event",
         "log_type_name": "fwd_bwd_end", "event_name": null, "value": 12.5,
         "task_name": "trainer_loop", "caller": "trainer.py:796:train_step",
         "seq_id": 42}
    """

    def __init__(self, rank: int, source: str):
        super().__init__()
        self.rank = rank
        self.source = source
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._host_name = socket.gethostname()
        self._seq_counter = itertools.count()

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self._log_dict(record))

    def _log_dict(self, record: logging.LogRecord) -> dict[str, Any]:
        """Build the flat dict emitted as one JSONL line.

        `host_name`/`pid` stay on every row (not just a startup record): `global_rank`
        is mesh-local (trainer and generator both have rank 0), so they are the only
        physical-process identity a windowed or merged read can rely on.
        """
        log_dict: dict[str, Any] = {
            "global_rank": self.rank,
            "local_rank": self._local_rank,
            "source": self.source,
            "host_name": self._host_name,
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
        }

        # Step context: per-record override (from event_extra) wins over step_state.
        step = getattr(record, str(ExtraFields.STEP), None)
        step = step if step is not None else get_step()
        if step is not None:
            log_dict["step"] = step
        relative_step = getattr(record, str(ExtraFields.RELATIVE_STEP), None)
        relative_step = (
            relative_step if relative_step is not None else get_relative_step()
        )
        if relative_step is not None:
            log_dict["relative_step"] = relative_step
        step_tags = get_step_tags()
        if step_tags:
            log_dict["step_tags"] = list(step_tags)

        log_dict["time_us"] = int(record.created * 1_000_000)
        log_type = getattr(record, str(ExtraFields.LOG_TYPE), str(LogType.TEXT))
        log_dict["log_type"] = log_type
        log_dict["log_type_name"] = getattr(
            record, str(ExtraFields.LOG_TYPE_NAME), None
        )
        log_dict["event_name"] = getattr(record, str(ExtraFields.EVENT_NAME), None)

        value = getattr(record, str(ExtraFields.VALUE), None)
        if isinstance(value, (float, int)):
            log_dict["value"] = float(value)

        # task_name pairs start/end records
        task_name = getattr(record, str(ExtraFields.TASK_NAME), None)
        if task_name is not None:
            log_dict["task_name"] = task_name

        # Caller field for source traceability (file:line:function)
        log_dict[
            "caller"
        ] = f"{os.path.relpath(record.pathname)}:{record.lineno}:{record.funcName}"

        log_dict["seq_id"] = next(self._seq_counter)

        # EVENT/INSTANT rows are fully described by their structured fields; the
        # human message would duplicate them on every span (measured ~10% of file
        # size). TEXT rows keep it -- it's all they have. Custom handlers still see
        # the original LogRecord, so this is a JSONL-format decision only.
        if log_type == str(LogType.TEXT):
            message = record.getMessage()
            if message:
                if len(message) > MAX_MESSAGE_SIZE:
                    half = MAX_MESSAGE_SIZE // 2
                    message = message[:half] + "..." + message[-half:]
                log_dict["message"] = message

        return log_dict


class TraceJsonlHandler(logging.FileHandler):
    """Per-rank JSONL file handler.

    File path::

        {output_dir}/structured_logs/{source}.global_rank_{rank}.{timestamp}-{random}.jsonl
    """

    def __init__(self, rank: int, source: str, output_dir: str):
        timestamp_str = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        random_str = "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
        )
        filename = f"{source}.global_rank_{rank}.{timestamp_str}-{random_str}.jsonl"
        filepath = os.path.join(output_dir, "structured_logs", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filename=filepath)
        self.setFormatter(TraceJsonlFormatter(rank=rank, source=source))
        self.addFilter(TraceEventsOnlyFilter())


def register_jsonl_handler(
    *,
    structured_logger: logging.Logger,
    rank: int,
    source: str,
    output_dir: str,
    **kw: Any,
) -> None:
    """Default factory: attach a ``TraceJsonlHandler`` to the structured logger."""
    handler = TraceJsonlHandler(rank=rank, source=source, output_dir=output_dir)
    structured_logger.addHandler(handler)
    console_logger.info("Structured logging -> JSONL: %s", handler.baseFilename)
