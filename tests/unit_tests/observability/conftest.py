"""
Shared conftest.py for observability tests.

Provides:
- requires_distributed / requires_gpu markers
- maybe_to_local: DTensor → local tensor for assertions
- assert_nested_equal: deep comparison of nested dicts/lists/tensors
- tmp_output_dir fixture
"""

import os
import pytest
import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Distributed test support
# ---------------------------------------------------------------------------
# For tests that NEED multi-GPU, mark them and run with torchrun.

requires_distributed = pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# DTensor helpers
# ---------------------------------------------------------------------------

def maybe_to_local(tensor):
    """Convert DTensor to local tensor for assertions. No-op for plain tensors."""
    if hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


# ---------------------------------------------------------------------------
# Nested comparison
# ---------------------------------------------------------------------------

def assert_nested_equal(a, b, rtol=1e-5, atol=1e-8):
    """Deep comparison of nested dicts/lists/tensors."""
    if isinstance(a, dict):
        assert isinstance(b, dict), f"Type mismatch: {type(a)} vs {type(b)}"
        assert set(a.keys()) == set(b.keys()), f"Key mismatch: {a.keys()} vs {b.keys()}"
        for key in a:
            assert_nested_equal(a[key], b[key], rtol=rtol, atol=atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
        for x, y in zip(a, b):
            assert_nested_equal(x, y, rtol=rtol, atol=atol)
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(
            maybe_to_local(a), maybe_to_local(b), rtol=rtol, atol=atol
        )
    elif isinstance(a, float):
        assert abs(a - b) <= atol + rtol * abs(b), f"Float mismatch: {a} vs {b}"
    else:
        assert a == b, f"Value mismatch: {a} vs {b}"


# ---------------------------------------------------------------------------
# Temp directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provides a temp directory for JSONL output, cleaned up after test."""
    output_dir = tmp_path / "observability_test_output"
    output_dir.mkdir()
    return str(output_dir)
