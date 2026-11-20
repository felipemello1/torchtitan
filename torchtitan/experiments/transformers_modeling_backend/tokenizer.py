# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import HuggingFaceTokenizer


class HFBackendTokenizer(HuggingFaceTokenizer):
    """Experiment tokenizer repair that the base class does not do.

    ``HuggingFaceTokenizer`` already passes ``bos_token`` and ``eos_token``
    into chat templates. This subclass sets ``eos_token`` in ``__init__``
    when the base left ``eos_id`` unset.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTokenizer.Config):
        pass

    def __init__(self, config=None, *, tokenizer_path):
        super().__init__(config=config, tokenizer_path=tokenizer_path)
        # Fix: when bos_token and eos_token are the same string, the base
        # tokenizer's elif logic only sets bos_token. Ensure eos is set too.
        if self.eos_id is None and self._hf_config:
            eos_str = self._get_token_from_config(self._hf_config, "eos_token")
            if eos_str is not None:
                self.eos_token = eos_str
                self.eos_id = self.tokenizer.token_to_id(eos_str)
