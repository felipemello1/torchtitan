# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for tensor_metrics.py (PR2).

Adapted from sixlib/metrics_test.py (non-distributed subset).
"""

import torch
import torch.utils._pytree as pytree

import pytest

from torchtitan.observability.tensor_metrics import (
    _check_merge_compatibility,
    DerivedTMetric,
    MaxTMetric,
    MeanTMetric,
    MinTMetric,
    replicate_to_host,
    SumTMetric,
    TMetricValue,
)


# ---------------------------------------------------------------------------
# MeanTMetric (was WeightedScalar)
# ---------------------------------------------------------------------------


class TestMeanTMetric:
    def test_from_mean_and_weight(self):
        m = MeanTMetric(mean=torch.tensor(2.0), weight=torch.tensor(3.0))
        assert m.sum.item() == pytest.approx(6.0)
        assert m.weight.item() == pytest.approx(3.0)

    def test_from_sum_and_weight(self):
        m = MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0))
        assert m.value() == pytest.approx(2.0, abs=1e-6)

    def test_python_scalars(self):
        m = MeanTMetric(mean=2.0, weight=3.0)
        assert m.value() == pytest.approx(2.0)

    def test_neither_mean_nor_sum_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            MeanTMetric(weight=1.0)

    def test_both_mean_and_sum_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            MeanTMetric(mean=1.0, sum=2.0, weight=1.0)

    def test_merge(self):
        a = MeanTMetric(sum=torch.tensor(4.0), weight=torch.tensor(2.0))
        b = MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0))
        a.merge_(b)
        assert a.sum.item() == pytest.approx(10.0)
        assert a.weight.item() == pytest.approx(5.0)
        assert a.value() == pytest.approx(2.0, abs=1e-6)

    def test_merge_python_scalars(self):
        a = MeanTMetric(sum=4.0, weight=2.0)
        b = MeanTMetric(sum=6.0, weight=3.0)
        a.merge_(b)
        assert a.value() == pytest.approx(2.0)

    def test_zero_weight(self):
        m = MeanTMetric(sum=0.0, weight=0.0)
        assert m.value() == 0.0

    def test_weight_1(self):
        m = MeanTMetric(mean=5.0, weight=1.0)
        assert m.weight == 1.0
        assert m.value() == pytest.approx(5.0)

    def test_pytree_roundtrip(self):
        m = MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0))
        leaves, spec = pytree.tree_flatten(m)
        assert len(leaves) == 2
        restored = spec.unflatten(leaves)
        assert isinstance(restored, MeanTMetric)
        assert restored.value() == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _ReducedTensorMetric subclasses
# ---------------------------------------------------------------------------


class TestReducedMetrics:
    @pytest.mark.parametrize("cls,val,expected", [
        (SumTMetric, 5.0, 5.0),
        (MaxTMetric, 5.0, 5.0),
        (MinTMetric, 5.0, 5.0),
    ])
    def test_basic_value(self, cls, val, expected):
        m = cls(val)
        assert m.value() == expected

    @pytest.mark.parametrize("cls,vals,expected", [
        (SumTMetric, [1.0, 2.0, 3.0], 6.0),
        (MaxTMetric, [1.0, 3.0, 2.0], 3.0),
        (MinTMetric, [3.0, 1.0, 2.0], 1.0),
    ])
    def test_merge(self, cls, vals, expected):
        metrics = [cls(v) for v in vals]
        result = metrics[0]
        for m in metrics[1:]:
            result.merge_(m)
        assert result.value() == pytest.approx(expected)

    @pytest.mark.parametrize("cls", [SumTMetric, MaxTMetric, MinTMetric])
    def test_tensor_value(self, cls):
        m = cls(torch.tensor(3.14))
        assert isinstance(m.value(), torch.Tensor)
        assert m.value().item() == pytest.approx(3.14)

    @pytest.mark.parametrize("cls", [SumTMetric, MaxTMetric, MinTMetric])
    def test_pytree_roundtrip(self, cls):
        m = cls(torch.tensor(7.0))
        leaves, spec = pytree.tree_flatten(m)
        restored = spec.unflatten(leaves)
        assert isinstance(restored, cls)
        assert restored.value().item() == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# merge compatibility
# ---------------------------------------------------------------------------


class TestMergeCompatibility:
    def test_tensor_scalar_mismatch(self):
        with pytest.raises(TypeError, match="mixed types"):
            _check_merge_compatibility(torch.tensor(1.0), 1.0, "test")

    def test_float_into_int_rejected(self):
        with pytest.raises(TypeError, match="cannot merge float"):
            _check_merge_compatibility(
                torch.tensor(1, dtype=torch.int64),
                torch.tensor(1.0),
                "test",
            )


# ---------------------------------------------------------------------------
# DerivedTMetric
# ---------------------------------------------------------------------------


class TestDerivedTMetric:
    def test_basic(self):
        a = MeanTMetric(sum=6.0, weight=3.0)
        b = MaxTMetric(10.0)
        d = DerivedTMetric(
            compute_fn=lambda vals: vals[0] / max(vals[1], 1),
            deps=[a, b],
        )
        assert d.value() == pytest.approx(0.2)

    def test_merge_raises(self):
        d = DerivedTMetric(compute_fn=lambda v: v[0], deps=[MaxTMetric(1.0)])
        with pytest.raises(ValueError, match="does not support"):
            d.merge_(d)


# ---------------------------------------------------------------------------
# replicate_to_host (non-distributed — CPU tensors and Python scalars)
# ---------------------------------------------------------------------------


class TestReplicateToHost:
    def test_cpu_tensors(self):
        metrics = {
            "loss": MeanTMetric(sum=torch.tensor(6.0), weight=torch.tensor(3.0)),
            "max": MaxTMetric(torch.tensor(10.0)),
        }
        result = replicate_to_host(metrics)
        assert result["loss"] == pytest.approx(2.0, abs=1e-5)
        assert result["max"] == pytest.approx(10.0)

    def test_python_scalars(self):
        metrics = {
            "avg": MeanTMetric(sum=10.0, weight=5.0),
        }
        result = replicate_to_host(metrics)
        assert result["avg"] == pytest.approx(2.0)

    def test_empty_dict(self):
        assert replicate_to_host({}) == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_tensors(self):
        metrics = {
            "loss": MeanTMetric(
                sum=torch.tensor(6.0, device="cuda"),
                weight=torch.tensor(3.0, device="cuda"),
            ),
        }
        result = replicate_to_host(metrics)
        assert result["loss"] == pytest.approx(2.0, abs=1e-5)
