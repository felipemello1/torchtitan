# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download Qwen checkpoints from Hugging Face and save the real tensors the other scripts read.

Each model runs on the same fixed text, the c4 sample in tests/assets/c4_test/data.json, cut into
1024-token sequences. Saved in --cache-dir; existing files are kept:

    qwen3_8b_head.pt       2 sequences: hidden states [2048, 4096], lm_head weight [151936, 4096]
    qwen3_1p7b_hidden.pt   16 sequences: hidden states [16384, 2048], the router inputs
    qwen3_5_27b_head.pt    2 sequences: hidden states [2048, 5120], lm_head weight [248320, 5120]
                           (only with --with-27b: a 54 GB download, ~60 GB of GPU memory)

Hidden states are the decoder's output after its final norm (the lm_head's input), in bf16.

Example:
    python scripts/fp32_output_linear/prepare_data.py --cache-dir ~/.cache/fp32_output_linear
    python scripts/fp32_output_linear/prepare_data.py --with-27b
"""

import argparse
import json
import os

import torch
from common import CACHE_FILES, DEFAULT_CACHE_DIR

SEQ_LEN = 1024
C4_SAMPLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../tests/assets/c4_test/data.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--with-27b", action="store_true")
    args = parser.parse_args()

    cache_dir = os.path.expanduser(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    for name, (repo, num_sequences, save_weight) in CACHE_FILES.items():
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            print(f"{path}: exists, kept")
            continue
        if repo == "Qwen/Qwen3.5-27B" and not args.with_27b:
            print(f"{path}: skipped (needs --with-27b)")
            continue
        token_ids = fixed_token_ids(repo, num_sequences)
        hidden_TD, weight_VD = run_model(repo, token_ids)
        torch.save(
            {
                "model": repo,
                "token_ids": token_ids,
                "hidden": hidden_TD.cpu(),
                "weight": weight_VD.cpu() if save_weight else None,
            },
            path,
        )
        print(
            f"{path}: {repo}, hidden {tuple(hidden_TD.shape)} {hidden_TD.dtype}", end=""
        )
        print(f", lm_head weight {tuple(weight_VD.shape)}" if save_weight else "")
        del hidden_TD, weight_VD
        torch.cuda.empty_cache()


def fixed_token_ids(repo: str, num_sequences: int) -> torch.Tensor:
    """The c4 sample's documents, joined and tokenized, cut into [num_sequences, 1024] token ids."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo)
    documents = []
    with open(C4_SAMPLE) as f:
        # ~4 characters per token: stop reading once there is surely enough text.
        while sum(len(d) for d in documents) < 8 * num_sequences * SEQ_LEN:
            documents.append(json.loads(f.readline())["text"])
    token_ids = tokenizer("\n\n".join(documents))["input_ids"]
    return torch.tensor(token_ids[: num_sequences * SEQ_LEN]).view(-1, SEQ_LEN)


def run_model(repo: str, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(final-norm hidden states [num_sequences * 1024, D], lm_head weight [V, D]), both bf16."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # Qwen3.5 checkpoints are vision-language models; the text decoder is model.model.
    model_class = (
        AutoModelForImageTextToText if "Qwen3.5" in repo else AutoModelForCausalLM
    )
    model = model_class.from_pretrained(repo, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        hidden_BSD = torch.cat(
            [
                model.model(input_ids=batch).last_hidden_state
                for batch in token_ids.cuda().split(4)
            ]
        )
    return hidden_BSD.flatten(0, 1).contiguous(), model.lm_head.weight.detach().clone()


if __name__ == "__main__":
    main()
