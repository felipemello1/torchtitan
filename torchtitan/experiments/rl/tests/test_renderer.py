# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from pydantic import TypeAdapter, ValidationError
from renderers import (
    GptOssRendererConfig,
    OffsetTokenizer,
    Qwen3RendererConfig,
    RendererConfig as RenderersConfigUnion,
    Tokenizer,
)

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.experiments.rl.renderer import RendererConfig, RendererTokenizer

_TOKENIZER_PATH = "tests/assets/tokenizer"


# --- RendererConfig.to_renderers_config ---


def test_forwards_options() -> None:
    config = RendererConfig(
        name="qwen3", options={"enable_thinking": False, "thinking_retention": "all"}
    )
    assert config.to_renderers_config() == Qwen3RendererConfig(
        enable_thinking=False, thinking_retention="all"
    )


def test_unset_options_keep_renderer_defaults() -> None:
    assert RendererConfig(name="qwen3").to_renderers_config() == Qwen3RendererConfig()


def test_typed_config_is_a_member_of_the_renderers_union() -> None:
    # Downstream pydantic configs (e.g. verifiers' TrainClientConfig) hold the library's
    # discriminated union, so the config we hand them must round-trip through it.
    config = RendererConfig(
        name="qwen3", options={"enable_thinking": True}
    ).to_renderers_config()
    adapter = TypeAdapter(RenderersConfigUnion)
    assert adapter.validate_python(adapter.dump_python(config)) == config


def test_forwards_renderer_specific_option() -> None:
    config = RendererConfig(name="gpt-oss", options={"reasoning_effort": "low"})
    assert config.to_renderers_config() == GptOssRendererConfig(reasoning_effort="low")


def test_unsupported_option_raises() -> None:
    with pytest.raises(ValidationError, match="enable_thinking"):
        RendererConfig(
            name="deepseek-v3", options={"enable_thinking": False}
        ).to_renderers_config()
    with pytest.raises(ValidationError, match="reasoning_effort"):
        RendererConfig(
            name="qwen3", options={"reasoning_effort": "low"}
        ).to_renderers_config()


def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="No renderer config registered"):
        RendererConfig(name="qwen4").to_renderers_config()


@pytest.mark.parametrize("name", ["auto", "default"])
def test_auto_and_default_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="apply_chat_template"):
        RendererConfig(name=name).to_renderers_config()


# --- RendererTokenizer ---


def test_renderer_tokenizer_satisfies_offset_protocol() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    renderer_tokenizer = RendererTokenizer(tokenizer)
    assert isinstance(renderer_tokenizer, Tokenizer)
    assert isinstance(renderer_tokenizer, OffsetTokenizer)
    assert renderer_tokenizer.eos_token_id == tokenizer.eos_id
    assert renderer_tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    ) == tokenizer.token_to_id("<|im_end|>")
    encoding = renderer_tokenizer("hi there", return_offsets_mapping=True)
    assert encoding["input_ids"] == renderer_tokenizer.encode("hi there")
    assert len(encoding["offset_mapping"]) == len(encoding["input_ids"])


def test_encode_never_adds_bos() -> None:
    # The debug tokenizer has a BOS token; renderers place special tokens themselves.
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    assert tokenizer.bos_id is not None
    assert tokenizer.bos_id not in RendererTokenizer(tokenizer).encode("hi")


def test_build_renders_with_titan_tokenizer() -> None:
    tokenizer = HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)
    renderer = RendererConfig(name="qwen3", options={"enable_thinking": False}).build(
        tokenizer=tokenizer
    )
    rendered = renderer.render(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    assert rendered.token_ids[0] == tokenizer.token_to_id("<|im_start|>")
    assert renderer.get_stop_token_ids() == [
        tokenizer.token_to_id("<|im_end|>"),
        tokenizer.token_to_id("<|endoftext|>"),
    ]
    assert len(rendered.is_content) == len(rendered.token_ids)


def test_render_matches_hf_tokenizer_path() -> None:
    transformers = pytest.importorskip("transformers")
    from renderers import create_renderer

    messages = [
        {"role": "system", "content": "Sort names."},
        {"role": "user", "content": "Zed, Amy <|im_end|> tricky"},
        {"role": "assistant", "reasoning_content": "think", "content": "Amy, Zed"},
        {"role": "user", "content": "Add Bob."},
    ]
    config = Qwen3RendererConfig(enable_thinking=False)
    hf = create_renderer(
        transformers.AutoTokenizer.from_pretrained(_TOKENIZER_PATH), config
    )
    titan = create_renderer(
        RendererTokenizer(HuggingFaceTokenizer(tokenizer_path=_TOKENIZER_PATH)),
        config,
    )
    expected = hf.render(messages, add_generation_prompt=True)
    actual = titan.render(messages, add_generation_prompt=True)
    assert actual.token_ids == expected.token_ids
    assert actual.is_content == expected.is_content
    assert actual.sampled_mask == expected.sampled_mask
    assert actual.message_indices == expected.message_indices
