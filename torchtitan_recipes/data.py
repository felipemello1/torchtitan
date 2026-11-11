# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset-specific defaults shared by training recipes."""

import logging

from torchtitan.observability.logging import warn_once


logger = logging.getLogger(__name__)


def c4_document_capacity(num_tokens_per_microbatch: int) -> int:
    """Estimate C4 document capacity from the per-rank token budget."""
    warn_once(
        logger,
        "C4 document capacity is an estimate. Tune dataloader.max_num_documents "
        "for your workload to balance attention overhead and packing padding.",
    )
    # The packer pads the rest of a row once it holds this many documents. C4 averages
    # one document per ~460 tokens (Qwen3 tokenizer). The busiest rows measured, from 8k
    # to 164k tokens, held up to 25 documents more than one per 384 tokens. Empty slots
    # cost a little attention time, so the margin stays small: one slot per 384 tokens,
    # plus 32.
    return (num_tokens_per_microbatch + 383) // 384 + 32
