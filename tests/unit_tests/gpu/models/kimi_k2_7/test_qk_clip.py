# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every CITRINE

import importlib.util
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

import torch
import torch.nn as nn
import torch_remat as remat
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.distributed import ParallelismContext
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention

from torchtitan.models.kimi_k2_7.qk_clip import (
    qk_clip,
    QKClipFlexInnerAttention,
    QKClipVarlenInnerAttention,
    register_qk_clip_hook,
)


def _qk_clip_config() -> QKClipFlexInnerAttention.Config:
    return QKClipFlexInnerAttention.Config()


def _qk_clip_mla_attention() -> MLAAttention:
    return MLAAttention.Config(
        wkv_b=Linear.Config(in_features=1, out_features=1),
        packed_kv_head_dim=1,
        inner_attention=_qk_clip_config(),
    ).build()


class QKClipTest(unittest.TestCase):
    def test_attention_records_training_maxima_only(self) -> None:
        attention = _qk_clip_config().build()
        q_THK = torch.randn(2, 2, 4)
        max_scores_1HT = torch.tensor([[[1.0, 3.0], [4.0, 2.0]]])
        block_mask = create_block_mask(
            lambda _b, _h, q_idx, kv_idx: q_idx >= kv_idx,
            1,
            2,
            2,
            2,
            device="cpu",
            _compile=False,
        )
        aux = SimpleNamespace(lse=None, max_scores=max_scores_1HT)

        attention.train()
        with patch(
            "torchtitan.models.common.attention.FlexInnerAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            FlexInnerAttention.forward(
                attention,
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=block_mask,
            )

        self.assertEqual(len(attention.max_attention_logits_H), 1)
        torch.testing.assert_close(
            attention.max_attention_logits_H[0],
            torch.tensor([3.0, 4.0]),
        )

        attention.max_attention_logits_H.clear()
        attention.eval()
        with patch(
            "torchtitan.models.common.attention.FlexInnerAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            FlexInnerAttention.forward(
                attention,
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=block_mask,
            )

        self.assertFalse(attention.max_attention_logits_H)

    def test_remat_replay_does_not_record_duplicate_maxima(self) -> None:
        attention = _qk_clip_config().build()
        attention.train()
        q_THK = torch.randn(2, 2, 4, requires_grad=True)
        max_scores_1HT = torch.tensor([[[1.0, 3.0], [4.0, 2.0]]])
        block_mask = create_block_mask(
            lambda _b, _h, q_idx, kv_idx: q_idx >= kv_idx,
            1,
            2,
            2,
            2,
            device="cpu",
            _compile=False,
        )
        aux = SimpleNamespace(lse=None, max_scores=max_scores_1HT)

        def forward(q_THK: torch.Tensor) -> torch.Tensor:
            return FlexInnerAttention.forward(
                attention,
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=block_mask,
            ).sum()

        with patch(
            "torchtitan.models.common.attention.FlexInnerAttention.compiled_flex_attn",
            return_value=(q_THK.transpose(0, 1).unsqueeze(0), aux),
        ):
            remat.checkpoint(region_name="attention")(forward)(q_THK).backward()

        self.assertEqual(len(attention.max_attention_logits_H), 1)
        torch.testing.assert_close(
            attention.max_attention_logits_H[0],
            torch.tensor([3.0, 4.0]),
        )

    def test_optimizer_hook_runs_qk_clip(self) -> None:
        model = nn.Linear(2, 2, bias=False)
        optimizers = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*",
                    lr=0.0,
                    weight_decay=0.0,
                    fused=False,
                )
            ],
        ).build(model_parts=[model])
        reduction_mesh = cast(
            DeviceMesh,
            SimpleNamespace(size=lambda: 1),
        )

        def get_mesh(mesh_name: str) -> DeviceMesh:
            self.assertEqual(mesh_name, "loss")
            return reduction_mesh

        parallelism_context = cast(
            ParallelismContext,
            SimpleNamespace(get_mesh=get_mesh),
        )
        with patch("torchtitan.models.kimi_k2_7.qk_clip.qk_clip") as mock_qk_clip:
            register_qk_clip_hook(optimizers, [model], parallelism_context)
            model.weight.grad = torch.zeros_like(model.weight)
            optimizers.step()

        mock_qk_clip.assert_called_once_with(
            [model],
            reduction_mesh=reduction_mesh,
        )


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class QKClipDistributedTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def device_type(self) -> str:
        return "cuda"

    @with_comms
    def test_reduces_head_maxima_and_clips_local_dtensor_weights(self) -> None:
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        num_heads = self.world_size
        qk_nope_head_dim = 2
        qk_rope_head_dim = 1
        v_head_dim = 2
        in_features = 2

        def weight_module(num_rows: int) -> nn.Module:
            module = nn.Module()
            module.register_parameter(
                "weight",
                nn.Parameter(
                    distribute_tensor(
                        torch.ones(num_rows, in_features, device=device),
                        mesh,
                        (Shard(0),),
                    )
                ),
            )
            return module

        attention = DeepSeekMLAAttention.__new__(DeepSeekMLAAttention)
        nn.Module.__init__(attention)
        attention.q_lora_rank = 1
        attention.qk_nope_head_dim = qk_nope_head_dim
        attention.qk_rope_head_dim = qk_rope_head_dim
        attention.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        attention.v_head_dim = v_head_dim
        attention.wq_b = weight_module(num_heads * attention.qk_head_dim)
        attention.inner_attention = _qk_clip_mla_attention()
        attention.inner_attention.wkv_b = weight_module(
            num_heads * (qk_nope_head_dim + v_head_dim)
        )
        rank_maxima = (
            torch.tensor([50.0, 400.0], device=device)
            if self.rank == 0
            else torch.tensor([200.0, 50.0], device=device)
        )
        attention.inner_attention.inner_attention.max_attention_logits_H.append(
            rank_maxima
        )
        model = nn.Module()
        model.add_module("attention", attention)

        qk_clip(
            [model],
            reduction_mesh=mesh,
        )

        local_scale = 0.5 if self.rank == 0 else 0.25
        q_weight_HDI = attention.wq_b.weight.to_local().view(
            1,
            attention.qk_head_dim,
            in_features,
        )
        kv_weight_HDI = attention.inner_attention.wkv_b.weight.to_local().view(
            1,
            qk_nope_head_dim + v_head_dim,
            in_features,
        )
        torch.testing.assert_close(
            q_weight_HDI[:, :qk_nope_head_dim],
            torch.full(
                (1, qk_nope_head_dim, in_features),
                local_scale**0.5,
                device=device,
            ),
        )
        torch.testing.assert_close(
            q_weight_HDI[:, qk_nope_head_dim:],
            torch.full(
                (1, qk_rope_head_dim, in_features),
                local_scale,
                device=device,
            ),
        )
        torch.testing.assert_close(
            kv_weight_HDI[:, :qk_nope_head_dim],
            torch.full(
                (1, qk_nope_head_dim, in_features),
                local_scale**0.5,
                device=device,
            ),
        )
        torch.testing.assert_close(
            kv_weight_HDI[:, qk_nope_head_dim:],
            torch.ones(1, v_head_dim, in_features, device=device),
        )
        self.assertFalse(
            attention.inner_attention.inner_attention.max_attention_logits_H
        )

    @with_comms
    def test_weight_scaling_is_communication_free(self) -> None:
        """Per-head scaling must stay local for every clipped weight.

        Building the replicated scales with ``distribute_tensor`` instead of
        ``from_local`` would broadcast once per weight per step, so this asserts
        the packed maxima all-reduce is the only collective ``qk_clip`` issues.
        """
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        num_heads = self.world_size
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim, in_features = 2, 1, 2, 2
        num_layers = 4

        def weight_module(num_rows: int) -> nn.Module:
            module = nn.Module()
            module.register_parameter(
                "weight",
                nn.Parameter(
                    distribute_tensor(
                        torch.ones(num_rows, in_features, device=device),
                        mesh,
                        (Shard(0),),
                    )
                ),
            )
            return module

        model = nn.Module()
        for layer_id in range(num_layers):
            attention = DeepSeekMLAAttention.__new__(DeepSeekMLAAttention)
            nn.Module.__init__(attention)
            attention.q_lora_rank = 1
            attention.qk_nope_head_dim = qk_nope_head_dim
            attention.qk_rope_head_dim = qk_rope_head_dim
            attention.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
            attention.v_head_dim = v_head_dim
            attention.wq_b = weight_module(num_heads * attention.qk_head_dim)
            attention.inner_attention = _qk_clip_mla_attention()
            attention.inner_attention.wkv_b = weight_module(
                num_heads * (qk_nope_head_dim + v_head_dim)
            )
            attention.inner_attention.inner_attention.max_attention_logits_H.append(
                torch.full((num_heads,), 400.0, device=device)
            )
            model.add_module(f"layer_{layer_id}", attention)

        with CommDebugMode() as comm_mode:
            qk_clip([model], reduction_mesh=mesh)

        collectives = {
            str(op): count for op, count in comm_mode.get_comm_counts().items() if count
        }
        total = sum(collectives.values())
        # One packed MAX all-reduce covers every layer and head; the 8 weights
        # this model clips must add nothing on top of it.
        self.assertEqual(total, 1, f"expected one collective, got {collectives}")


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = actual.float() - expected.float()
    return (diff.norm() / expected.float().norm()).item()


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability() >= (10, 0)
    and importlib.util.find_spec("cudnn.flex_attention") is not None,
    "cuDNN interval-mask attention needs SM100+ and cudnn-frontend >= 1.31",
)
class QKClipVarlenInnerAttentionTest(unittest.TestCase):
    """QKClipVarlenInnerAttention against QKClipFlexInnerAttention at MLA head dims."""

    def setUp(self) -> None:
        QKClipVarlenInnerAttention._mask_plan_cache = None
        torch.manual_seed(0)
        # Two packed documents; MLA head dims (192 query/key, 128 value).
        self.positions = torch.cat([torch.arange(384), torch.arange(640)]).cuda()
        num_tokens, num_heads = self.positions.numel(), 4
        self.scale = 192**-0.5
        self.q = torch.randn(
            num_tokens, num_heads, 192, device="cuda", dtype=torch.bfloat16
        )
        self.k = torch.randn(
            num_tokens, num_heads, 192, device="cuda", dtype=torch.bfloat16
        )
        # A strided value, as materialize_mla_kv returns it.
        self.v = torch.randn(
            num_tokens, num_heads, 256, device="cuda", dtype=torch.bfloat16
        )[..., 128:]
        self.grad_out = torch.randn(
            num_tokens, num_heads, 128, device="cuda", dtype=torch.bfloat16
        )

    def _run(self, attention: nn.Module, metadata) -> tuple[torch.Tensor, ...]:
        q, k, v = (t.detach().requires_grad_() for t in (self.q, self.k, self.v))
        out = attention(q, k, v, attention_metadata=metadata, scale=self.scale)
        out.backward(self.grad_out)
        assert q.grad is not None and k.grad is not None and v.grad is not None
        return out, q.grad, k.grad, v.grad

    def test_matches_flex_qk_clip(self) -> None:
        flex = QKClipFlexInnerAttention.Config().build().train()
        varlen = QKClipVarlenInnerAttention.Config().build().train()
        expected = self._run(
            flex, FlexInnerAttention.Config().build_attention_metadata(self.positions)
        )
        actual = self._run(
            varlen,
            QKClipVarlenInnerAttention.Config().build_attention_metadata(
                self.positions
            ),
        )
        # Both are bf16 kernels; each is ~2e-3 from fp32.
        for name, a, e in zip(("out", "dq", "dk", "dv"), actual, expected, strict=True):
            self.assertLess(_rel_l2(a, e), 1e-2, name)
        torch.testing.assert_close(
            varlen.max_attention_logits_H[0],
            flex.max_attention_logits_H[0],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_layers_share_one_mask_plan(self) -> None:
        from cudnn import flex_attention

        metadata = QKClipVarlenInnerAttention.Config().build_attention_metadata(
            self.positions
        )
        layers = [QKClipVarlenInnerAttention.Config().build().train() for _ in range(3)]
        with patch.object(
            flex_attention,
            "create_mask_plan",
            wraps=flex_attention.create_mask_plan,
        ) as create_mask_plan:
            for layer in layers:
                self._run(layer, metadata)
        create_mask_plan.assert_called_once()

    def test_records_training_maxima_once(self) -> None:
        attention = QKClipVarlenInnerAttention.Config().build()
        metadata = QKClipVarlenInnerAttention.Config().build_attention_metadata(
            self.positions
        )
        attention.eval()
        with torch.no_grad():
            attention(
                self.q, self.k, self.v, attention_metadata=metadata, scale=self.scale
            )
        self.assertFalse(attention.max_attention_logits_H)

        attention.train()
        q = self.q.detach().requires_grad_()

        def forward(q: torch.Tensor) -> torch.Tensor:
            # A dense output gradient: cuDNN's backward rejects the expanded one of .sum().
            out = attention(
                q, self.k, self.v, attention_metadata=metadata, scale=self.scale
            )
            return (out * self.grad_out).sum()

        remat.checkpoint(region_name="attention")(forward)(q).backward()
        self.assertEqual(len(attention.max_attention_logits_H), 1)

    def test_qk_clip_collects_varlen_maxima(self) -> None:
        attention = DeepSeekMLAAttention.__new__(DeepSeekMLAAttention)
        nn.Module.__init__(attention)
        attention.inner_attention = MLAAttention.Config(
            wkv_b=Linear.Config(in_features=1, out_features=1),
            packed_kv_head_dim=1,
            inner_attention=QKClipVarlenInnerAttention.Config(),
        ).build()
        inner = attention.inner_attention.inner_attention
        inner.max_attention_logits_H.append(torch.tensor([50.0, 400.0]))
        model = nn.Module()
        model.add_module("attention", attention)
        reduction_mesh = cast(DeviceMesh, SimpleNamespace(size=lambda: 1))

        with patch("torchtitan.models.kimi_k2_7.qk_clip._clip_mla_weights") as clip:
            qk_clip([model], reduction_mesh=reduction_mesh)

        clip.assert_called_once()
        torch.testing.assert_close(clip.call_args.args[1], torch.tensor([1.0, 0.25]))
        self.assertFalse(inner.max_attention_logits_H)
