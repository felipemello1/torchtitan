# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from torchtitan.components.tokenizer import MultiModalTokenizer


class MuseSparkTokenizer(MultiModalTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(MultiModalTokenizer.Config):
        audio_token: str
        audio_start_token: str
        audio_end_token: str

    TOKEN_FIELDS: ClassVar[tuple[str, ...]] = (
        *MultiModalTokenizer.TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )
    LOSS_MASK_TOKEN_FIELDS: ClassVar[tuple[str, ...]] = (
        *MultiModalTokenizer.LOSS_MASK_TOKEN_FIELDS,
        "audio",
        "audio_start",
        "audio_end",
    )


__all__ = ["MuseSparkTokenizer"]
