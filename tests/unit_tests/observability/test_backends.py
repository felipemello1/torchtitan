# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for backends.py (summary writers)."""

import torch
import pytest

from torchtitan.observability.backends import (
    _to_scalar,
    CompositeSummaryWriter,
    InMemorySummaryWriter,
    LoggingSummaryWriter,
    SummaryWriter,
    TensorBoardSummaryWriter,
)
from torchtitan.observability.tensor_metrics import MeanTMetric, MaxTMetric


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestToScalar:
    def test_python_int(self):
        assert _to_scalar(42) == 42

    def test_python_float(self):
        assert _to_scalar(3.14) == pytest.approx(3.14)

    def test_bool_becomes_int(self):
        assert _to_scalar(True) == 1
        assert _to_scalar(False) == 0

    def test_tensor_raises(self):
        with pytest.raises(ValueError, match="Tensor"):
            _to_scalar(torch.tensor(1.0))

    def test_metric_with_scalar_value(self):
        m = MeanTMetric(sum=6.0, weight=3.0)
        assert _to_scalar(m) == pytest.approx(2.0)

    def test_metric_with_tensor_raises(self):
        m = MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0))
        with pytest.raises(ValueError, match="Tensor"):
            _to_scalar(m)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="unsupported type"):
            _to_scalar("string")


# ---------------------------------------------------------------------------
# InMemorySummaryWriter
# ---------------------------------------------------------------------------


class TestInMemorySummaryWriter:
    def test_basic_logging(self):
        writer = InMemorySummaryWriter()
        writer.open()
        writer(step=1, values={"loss": 2.5, "lr": 1e-4})
        writer(step=2, values={"loss": 2.3, "lr": 1e-4})
        writer.close()

        assert writer.summaries[1]["loss"] == pytest.approx(2.5)
        assert writer.summaries[2]["loss"] == pytest.approx(2.3)

    def test_context_manager(self):
        writer = InMemorySummaryWriter()
        with writer:
            writer(step=1, values={"loss": 1.0})
        assert 1 in writer.summaries

    def test_not_open_raises(self):
        writer = InMemorySummaryWriter()
        with pytest.raises(RuntimeError, match="not open"):
            writer(step=1, values={"loss": 1.0})


# ---------------------------------------------------------------------------
# CompositeSummaryWriter
# ---------------------------------------------------------------------------


class TestCompositeSummaryWriter:
    def test_fans_out_to_all(self):
        w1 = InMemorySummaryWriter()
        w2 = InMemorySummaryWriter()
        composite = CompositeSummaryWriter(writers={"a": w1, "b": w2})
        composite.open()
        composite(step=1, values={"loss": 2.0})
        composite.close()

        assert w1.summaries[1]["loss"] == pytest.approx(2.0)
        assert w2.summaries[1]["loss"] == pytest.approx(2.0)

    def test_empty_writers(self):
        composite = CompositeSummaryWriter(writers={})
        composite.open()
        composite(step=1, values={"loss": 1.0})
        composite.close()

    def test_isolates_writer_failures(self, caplog):
        """If one writer fails, others continue."""

        class FailingWriter(SummaryWriter):
            def _log(self, step, values):
                raise RuntimeError("boom")

        good = InMemorySummaryWriter()
        bad = FailingWriter()
        composite = CompositeSummaryWriter(writers={"good": good, "bad": bad})
        composite.open()
        composite(step=1, values={"loss": 1.0})
        composite.close()

        # Good writer should still have received the values
        assert good.summaries[1]["loss"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LoggingSummaryWriter
# ---------------------------------------------------------------------------


class TestLoggingSummaryWriter:
    def test_logs_to_logger(self, caplog):
        writer = LoggingSummaryWriter()
        with writer:
            with caplog.at_level("INFO"):
                writer(step=5, values={"loss": 1.5})
        assert "step 5" in caplog.text
        assert "1.5" in caplog.text
