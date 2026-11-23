# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import re
import stat
from unittest.mock import patch

import pytest

from torchtitan.rl._env import apply_env_defaults, NODE_CACHE_ROOT_ENV, RL_ENV_DEFAULTS


def test_apply_env_defaults() -> None:
    with patch.dict(os.environ, {}, clear=True):
        apply_env_defaults()
        apply_env_defaults()

        assert dict(os.environ) == {
            **RL_ENV_DEFAULTS,
            "PYTORCH_CUDA_ALLOC_CONF": RL_ENV_DEFAULTS["PYTORCH_ALLOC_CONF"],
        }


def test_apply_env_defaults_preserves_existing_values() -> None:
    with patch.dict(
        os.environ,
        {"NCCL_DEBUG": "INFO", "UNRELATED": "value"},
        clear=True,
    ):
        apply_env_defaults()

        assert os.environ["NCCL_DEBUG"] == "INFO"
        assert os.environ["UNRELATED"] == "value"


@pytest.mark.parametrize("value", ["", "expandable_segments:False"])
def test_apply_env_defaults_mirrors_explicit_allocator(value: str) -> None:
    with patch.dict(os.environ, {"PYTORCH_ALLOC_CONF": value}, clear=True):
        apply_env_defaults()
        apply_env_defaults()

        assert os.environ["PYTORCH_ALLOC_CONF"] == value
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == value


@pytest.mark.parametrize("canonical", [None, "", "expandable_segments:False"])
def test_apply_env_defaults_rejects_legacy_or_conflicting_allocator(canonical) -> None:
    explicit = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if canonical is not None:
        explicit["PYTORCH_ALLOC_CONF"] = canonical
    with patch.dict(os.environ, explicit, clear=True):
        with pytest.raises(ValueError, match="Use PYTORCH_ALLOC_CONF"):
            apply_env_defaults()
        assert dict(os.environ) == explicit


def test_apply_env_defaults_creates_private_node_cache_root(tmp_path) -> None:
    root = tmp_path / "cache"
    with patch.dict(os.environ, {NODE_CACHE_ROOT_ENV: str(root)}, clear=True):
        apply_env_defaults()
        # A second process on the node finds the root already there.
        apply_env_defaults()

    assert stat.S_IMODE(root.stat().st_mode) == 0o700


@pytest.mark.parametrize("unsafe", ["group_writable", "symlink", "other_owner"])
def test_apply_env_defaults_rejects_node_cache_root_others_can_write(
    tmp_path, monkeypatch, unsafe
) -> None:
    root = tmp_path / "cache"
    if unsafe == "symlink":
        private = tmp_path / "private"
        private.mkdir(mode=0o700)
        root.symlink_to(private)
    else:
        root.mkdir(mode=0o700)

    if unsafe == "group_writable":
        root.chmod(0o775)
    if unsafe == "other_owner":
        # A test can't chown; pretend the dir belongs to another user.
        monkeypatch.setattr(os, "geteuid", lambda: root.stat().st_uid + 1)

    with patch.dict(os.environ, {NODE_CACHE_ROOT_ENV: str(root)}, clear=True):
        with pytest.raises(RuntimeError, match=re.escape(str(root))):
            apply_env_defaults()
