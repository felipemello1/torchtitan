# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for torchtitan.observability.structured_logging.

Coverage map:
- set_step / add_step_tag / clear_step_tags: adapted from msl_tools context_test.py
- to_structured_json: new (no reference test)
- StructuredJSONFormatter: new (no reference test)
- EventsOnlyFilter: new
- init_observability: new
- record_event: new
- record_span: new
- InflightEventTrackingHandler: new
"""

import json
import logging
import os
import time

import pytest

from torchtitan.observability.common import (
    _STEP,
    _STEP_TAGS,
    add_step_tag,
    clear_step_tags,
    set_step,
    SYSTEM_LOGGER_NAME,
)
from torchtitan.observability.structured_logging import (
    dict_to_str_list,
    event_extra,
    EventsOnlyFilter,
    EventType,
    ExtraFields,
    init_observability,
    InflightEventTrackingHandler,
    LogType,
    MAX_MESSAGE_SIZE,
    record_event,
    record_span,
    StructuredJSONFormatter,
    StructuredLoggingHandler,
    to_structured_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_context():
    """Reset ContextVars before each test."""
    _STEP.set(None)
    _STEP_TAGS.set(())
    yield
    _STEP.set(None)
    _STEP_TAGS.set(())


@pytest.fixture
def system_logger():
    """Provide a clean system logger for testing."""
    logger = logging.getLogger(SYSTEM_LOGGER_NAME)
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    yield logger
    # Restore
    logger.handlers = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate


# ---------------------------------------------------------------------------
# Step context tests (adapted from msl_tools context_test.py TestStepTags)
# ---------------------------------------------------------------------------


class TestSetStep:
    def test_set_step_stores_value(self):
        set_step(42)
        assert _STEP.get() == 42

    def test_set_step_overwrites(self):
        set_step(1)
        set_step(2)
        assert _STEP.get() == 2

    def test_step_default_is_none(self):
        assert _STEP.get() is None


class TestStepTags:
    def test_add_step_tag_adds_tag(self):
        set_step(1)
        add_step_tag("gc")
        assert _STEP_TAGS.get() == ("gc",)

    def test_add_step_tag_deduplicates(self):
        add_step_tag("gc")
        add_step_tag("gc")
        assert _STEP_TAGS.get() == ("gc",)

    def test_add_step_tag_multiple_tags(self):
        add_step_tag("gc")
        add_step_tag("profiling")
        add_step_tag("checkpoint")
        assert _STEP_TAGS.get() == ("gc", "profiling", "checkpoint")

    def test_clear_step_tags_clears_all(self):
        add_step_tag("gc")
        add_step_tag("profiling")
        clear_step_tags()
        assert _STEP_TAGS.get() == ()

    def test_clear_step_tags_on_empty(self):
        clear_step_tags()
        assert _STEP_TAGS.get() == ()

    def test_step_tags_are_tuples(self):
        """ContextVar safety: tuples are immutable, lists would leak across tasks."""
        add_step_tag("a")
        tags = _STEP_TAGS.get()
        assert isinstance(tags, tuple)


# ---------------------------------------------------------------------------
# dict_to_str_list
# ---------------------------------------------------------------------------


class TestDictToListSafe:
    def test_none_returns_none(self):
        assert dict_to_str_list(None) is None

    def test_empty_dict_returns_empty_list(self):
        assert dict_to_str_list({}) == []

    def test_normal_dict(self):
        result = dict_to_str_list({"a": "1", "b": "2"})
        assert set(result) == {"a:1", "b:2"}


# ---------------------------------------------------------------------------
# event_extra
# ---------------------------------------------------------------------------


class TestEventExtra:
    def test_basic_event(self):
        extra = event_extra(EventType.FWD_BWD)
        assert extra[str(ExtraFields.LOG_TYPE)] == str(LogType.EVENT)
        assert extra[str(ExtraFields.LOG_TYPE_NAME)] == str(EventType.FWD_BWD)

    def test_with_step_and_value(self):
        extra = event_extra(EventType.STEP, step=10, value=42.0)
        assert extra[str(ExtraFields.STEP)] == 10
        assert extra[str(ExtraFields.VALUE)] == 42.0

    def test_with_context(self):
        extra = event_extra(EventType.STEP, context={"key": "val"})
        assert extra[str(ExtraFields.CONTEXT)] == ["key:val"]


# ---------------------------------------------------------------------------
# to_structured_json
# ---------------------------------------------------------------------------


class TestToStructuredJson:
    def test_int_field(self):
        result = json.loads(to_structured_json({"rank": 0}))
        assert result["int"]["rank"] == 0

    def test_float_field(self):
        result = json.loads(to_structured_json({"value": 3.14}))
        assert result["double"]["value"] == pytest.approx(3.14)

    def test_string_field(self):
        result = json.loads(to_structured_json({"source": "trainer"}))
        assert result["normal"]["source"] == "trainer"

    def test_list_field(self):
        result = json.loads(to_structured_json({"tags": ["a", "b"]}))
        assert result["normvector"]["tags"] == ["a", "b"]

    def test_none_values_skipped(self):
        result = json.loads(to_structured_json({"a": None, "b": 1}))
        assert "a" not in result["int"]
        assert "a" not in result["normal"]
        assert result["int"]["b"] == 1

    def test_bool_becomes_int(self):
        result = json.loads(to_structured_json({"flag": True}))
        assert result["int"]["flag"] == 1

    def test_empty_dict(self):
        result = json.loads(to_structured_json({}))
        assert result == {"int": {}, "normal": {}, "double": {}, "normvector": {}}


# ---------------------------------------------------------------------------
# StructuredJSONFormatter
# ---------------------------------------------------------------------------


class TestStructuredJSONFormatter:
    def test_format_produces_valid_json(self):
        fmt = StructuredJSONFormatter(rank=0, source="trainer")
        set_step(5)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="hello", args=None, exc_info=None,
        )
        # Add event extra fields
        for k, v in event_extra(EventType.FWD_BWD_START, step=5).items():
            setattr(record, k, v)

        output = fmt.format(record)
        parsed = json.loads(output)
        assert "int" in parsed
        assert "normal" in parsed
        assert parsed["int"]["rank"] == 0
        assert parsed["normal"]["source"] == "trainer"
        assert parsed["int"]["step"] == 5

    def test_rank_and_source_from_self(self):
        fmt = StructuredJSONFormatter(rank=3, source="generator")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test", args=None, exc_info=None,
        )
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["int"]["rank"] == 3
        assert parsed["normal"]["source"] == "generator"

    def test_step_from_contextvar(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
        set_step(42)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test", args=None, exc_info=None,
        )
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["int"]["step"] == 42

    def test_message_truncation(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
        long_msg = "x" * (MAX_MESSAGE_SIZE + 100)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg=long_msg, args=None, exc_info=None,
        )
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "..." in parsed["normal"]["message"]
        assert len(parsed["normal"]["message"]) < len(long_msg)

    def test_step_tags_in_output(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
        set_step(1)
        add_step_tag("gc")
        add_step_tag("profiling")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test", args=None, exc_info=None,
        )
        for k, v in event_extra(EventType.STEP).items():
            setattr(record, k, v)
        output = fmt.format(record)
        parsed = json.loads(output)
        assert "step_tags" in parsed["normvector"]
        assert set(parsed["normvector"]["step_tags"]) == {"gc", "profiling"}

    def test_seq_id_increments(self):
        fmt = StructuredJSONFormatter(rank=0, source="test")
        records = []
        for i in range(3):
            r = logging.LogRecord(
                name="test", level=logging.INFO, pathname="test.py",
                lineno=1, msg=f"msg{i}", args=None, exc_info=None,
            )
            for k, v in event_extra(EventType.STEP).items():
                setattr(r, k, v)
            records.append(r)

        seq_ids = []
        for r in records:
            parsed = json.loads(fmt.format(r))
            seq_ids.append(parsed["int"]["seq_id"])
        assert seq_ids == [0, 1, 2]


# ---------------------------------------------------------------------------
# EventsOnlyFilter
# ---------------------------------------------------------------------------


class TestEventsOnlyFilter:
    def test_passes_event_records(self):
        f = EventsOnlyFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test", args=None, exc_info=None,
        )
        setattr(record, str(ExtraFields.LOG_TYPE_NAME), str(EventType.STEP))
        assert f.filter(record) is True

    def test_blocks_non_event_records(self):
        f = EventsOnlyFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="just text", args=None, exc_info=None,
        )
        assert f.filter(record) is False


# ---------------------------------------------------------------------------
# InflightEventTrackingHandler
# ---------------------------------------------------------------------------


class TestInflightEventTrackingHandler:
    def test_tracks_last_event(self):
        handler = InflightEventTrackingHandler()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test", args=None, exc_info=None,
        )
        setattr(record, str(ExtraFields.LOG_TYPE_NAME), str(EventType.FWD_BWD_START))
        handler.emit(record)
        assert handler.last_event == str(EventType.FWD_BWD_START)
        assert handler.last_event_time is not None

    def test_ignores_non_event_records(self):
        handler = InflightEventTrackingHandler()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="plain text", args=None, exc_info=None,
        )
        handler.emit(record)
        assert handler.last_event is None


# ---------------------------------------------------------------------------
# init_observability
# ---------------------------------------------------------------------------


class TestInitObservability:
    def test_creates_system_jsonl(self, tmp_path, system_logger):
        output_dir = str(tmp_path)
        init_observability(rank=0, source="trainer", output_dir=output_dir)

        # Logger should have handlers
        assert any(isinstance(h, StructuredLoggingHandler) for h in system_logger.handlers)

        # File should exist after we log something
        set_step(1)
        system_logger.info(
            "test event",
            extra=event_extra(EventType.STEP, step=1),
        )

        expected_path = os.path.join(output_dir, "system_logs", "trainer_rank_0_system.jsonl")
        assert os.path.exists(expected_path)

        with open(expected_path) as f:
            line = f.readline().strip()
        parsed = json.loads(line)
        assert parsed["int"]["rank"] == 0
        assert parsed["normal"]["source"] == "trainer"

    def test_idempotent(self, tmp_path, system_logger):
        output_dir = str(tmp_path)
        init_observability(rank=0, source="trainer", output_dir=output_dir)
        handler_count = len(system_logger.handlers)
        init_observability(rank=0, source="trainer", output_dir=output_dir)
        assert len(system_logger.handlers) == handler_count

    def test_creates_inflight_handler(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        assert any(isinstance(h, InflightEventTrackingHandler) for h in system_logger.handlers)


# ---------------------------------------------------------------------------
# record_event
# ---------------------------------------------------------------------------


class TestRecordEvent:
    def test_writes_metric_events(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(10)
        record_event({"train.loss": 2.5, "train.lr": 1e-4})

        jsonl_path = os.path.join(str(tmp_path), "system_logs", "trainer_rank_0_system.jsonl")
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        # Check that event names are present
        event_names = {line["normal"].get("event_name") for line in lines}
        assert "train.loss" in event_names
        assert "train.lr" in event_names
        # Verify step appears in output
        assert all(line["int"].get("step") == 10 for line in lines)


# ---------------------------------------------------------------------------
# record_span
# ---------------------------------------------------------------------------


class TestRecordSpan:
    def test_logs_start_and_end(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(5)

        with record_span("Forward/Backward", EventType.FWD_BWD):
            time.sleep(0.01)

        jsonl_path = os.path.join(str(tmp_path), "system_logs", "trainer_rank_0_system.jsonl")
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["normal"]["log_type_name"] == str(EventType.FWD_BWD_START)
        assert lines[1]["normal"]["log_type_name"] == str(EventType.FWD_BWD_END)
        # End event should have duration in value
        assert lines[1]["double"]["value"] > 0
        # Both events should have step=5
        assert lines[0]["int"]["step"] == 5
        assert lines[1]["int"]["step"] == 5

    def test_works_as_decorator(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        @record_span("Optimizer", EventType.OPTIM)
        def optimizer_step():
            pass

        optimizer_step()

        jsonl_path = os.path.join(str(tmp_path), "system_logs", "trainer_rank_0_system.jsonl")
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["normal"]["log_type_name"] == str(EventType.OPTIM_START)
        assert lines[1]["normal"]["log_type_name"] == str(EventType.OPTIM_END)

    def test_does_not_suppress_exceptions(self, tmp_path, system_logger):
        init_observability(rank=0, source="trainer", output_dir=str(tmp_path))
        set_step(1)

        with pytest.raises(ValueError, match="test error"):
            with record_span("Test", EventType.STEP):
                raise ValueError("test error")

    def test_rejects_start_end_event_types(self):
        with pytest.raises(ValueError, match="not a START/END variant"):
            record_span("Bad", EventType.FWD_BWD_START)
        with pytest.raises(ValueError, match="not a START/END variant"):
            record_span("Bad", EventType.FWD_BWD_END)

    def test_rl_scoring_event_type(self, tmp_path, system_logger):
        init_observability(rank=0, source="reward", output_dir=str(tmp_path))
        set_step(1)
        with record_span("Scoring", EventType.RL_SCORING):
            pass
        jsonl_path = os.path.join(str(tmp_path), "system_logs", "reward_rank_0_system.jsonl")
        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert lines[0]["normal"]["log_type_name"] == str(EventType.RL_SCORING_START)
        assert lines[1]["normal"]["log_type_name"] == str(EventType.RL_SCORING_END)
