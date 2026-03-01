# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for invocation_context.py (PR2).

Adapted from sixlib/invocation_context_test.py (non-distributed subset).
"""

import torch
import pytest

from torchtitan.observability.invocation_context import (
    child_context,
    current_context,
    InvocationContext,
    record_tensor_metric,
)
from torchtitan.observability.tensor_metrics import (
    MaxTMetric,
    MeanTMetric,
    MinTMetric,
    replicate_to_host,
    SumTMetric,
)


class TestInvocationContext:
    def test_add_summary(self):
        with InvocationContext() as ctx:
            ctx.add_summary("loss", MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0)))
        summaries = ctx.summaries()
        assert "loss" in summaries
        assert summaries["loss"].value() == pytest.approx(2.0, abs=1e-6)

    def test_merge_on_duplicate_key(self):
        with InvocationContext() as ctx:
            ctx.add_summary("loss", MeanTMetric(sum=torch.tensor(4.0), weight=torch.tensor(2.0)))
            ctx.add_summary("loss", MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0)))
        summaries = ctx.summaries()
        # Merged: sum=10, weight=5, mean=2.0
        assert summaries["loss"].value() == pytest.approx(2.0, abs=1e-6)

    def test_rejects_non_metric_value(self):
        with InvocationContext() as ctx:
            with pytest.raises(TypeError, match="TMetricValue"):
                ctx.add_summary("bad", 42)

    def test_summaries_returns_copy(self):
        with InvocationContext() as ctx:
            ctx.add_summary("a", MaxTMetric(torch.tensor(1.0)))
        s1 = ctx.summaries()
        s2 = ctx.summaries()
        assert s1 is not s2
        assert set(s1.keys()) == set(s2.keys())

    def test_current_context(self):
        assert current_context() is None
        with InvocationContext() as ctx:
            assert current_context() is ctx
        assert current_context() is None

    def test_nested_context(self):
        with InvocationContext() as outer:
            assert current_context() is outer
            with InvocationContext() as inner:
                assert current_context() is inner
            assert current_context() is outer

    def test_child_context(self):
        with InvocationContext() as ctx:
            with child_context("encoder"):
                record_tensor_metric("loss", MaxTMetric(torch.tensor(5.0)))
        summaries = ctx.summaries()
        assert "encoder/loss" in summaries

    def test_global_step(self):
        with InvocationContext() as ctx:
            ctx.set_global_step(42)
            assert ctx.global_step() == 42

    def test_global_step_from_ancestor(self):
        with InvocationContext() as outer:
            outer.set_global_step(10)
            with InvocationContext() as inner:
                assert inner.global_step() == 10

    def test_step_state(self):
        with InvocationContext() as ctx:
            ctx.add_step_state("lr", 1e-3)
            assert ctx.get_step_state("lr") == 1e-3
            assert ctx.get_step_state("missing") is None

    def test_step_state_duplicate_raises(self):
        with InvocationContext() as ctx:
            ctx.add_step_state("lr", 1e-3)
            with pytest.raises(KeyError, match="already exists"):
                ctx.add_step_state("lr", 2e-3)

    def test_set_global_step_clears_step_state(self):
        with InvocationContext() as ctx:
            ctx.add_step_state("lr", 1e-3)
            ctx.set_global_step(2)
            assert ctx.get_step_state("lr") is None

    def test_update(self):
        with InvocationContext() as parent:
            with InvocationContext() as child:
                child.add_summary("loss", MaxTMetric(torch.tensor(5.0)))
                child.add_step_state("lr", 1e-3)
            parent.update(child, "encoder")
        assert "encoder/loss" in parent.summaries()
        assert parent.get_step_state("lr") == 1e-3

    def test_detaches_gradients(self):
        x = torch.tensor(3.0, requires_grad=True)
        with InvocationContext() as ctx:
            ctx.add_summary("val", MaxTMetric(x))
        # Value should be detached
        assert not ctx.summaries()["val"]._value.requires_grad


class TestRecordTensorMetric:
    def test_no_context_is_noop(self):
        # Should not raise
        record_tensor_metric("loss", MeanTMetric(mean=1.0, weight=1.0))

    def test_records_when_context_active(self):
        with InvocationContext() as ctx:
            record_tensor_metric("loss", MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0)))
            record_tensor_metric("max", MaxTMetric(torch.tensor(10.0)))
        summaries = ctx.summaries()
        assert "loss" in summaries
        assert "max" in summaries

    def test_merges_on_same_key(self):
        with InvocationContext() as ctx:
            record_tensor_metric("loss", MeanTMetric(sum=torch.tensor(4.0), weight=torch.tensor(2.0)))
            record_tensor_metric("loss", MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0)))
        result = replicate_to_host(ctx.summaries())
        assert result["loss"] == pytest.approx(2.0, abs=1e-5)


class TestEndToEnd:
    """Integration: InvocationContext + replicate_to_host."""

    def test_full_flow(self):
        with InvocationContext() as ctx:
            record_tensor_metric("loss", MeanTMetric(sum=torch.tensor(10.0), weight=torch.tensor(5.0)))
            record_tensor_metric("grad_norm", MaxTMetric(torch.tensor(1.5)))
            record_tensor_metric("min_reward", MinTMetric(torch.tensor(-0.5)))
            record_tensor_metric("total_tokens", SumTMetric(torch.tensor(128.0)))

        scalars = replicate_to_host(ctx.summaries())
        assert scalars["loss"] == pytest.approx(2.0, abs=1e-5)
        assert scalars["grad_norm"] == pytest.approx(1.5)
        assert scalars["min_reward"] == pytest.approx(-0.5)
        assert scalars["total_tokens"] == pytest.approx(128.0)
