# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
import unittest.mock

import spmd_types as spmd
import torch
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.qwen3 import build_model_config
from torchtitan.trainer import Trainer


class TestTokenDispatcherModule(unittest.TestCase):
    def test_dispatcher_has_no_checkpoint_state(self):
        dispatcher = AllToAllTokenDispatcher.Config(
            num_experts=2,
            top_k=1,
        ).build()

        self.assertIsInstance(dispatcher, torch.nn.Module)
        self.assertEqual(list(dispatcher.state_dict()), [])


class TestExpertParallelConfigValidation(unittest.TestCase):
    @staticmethod
    def _config(ep: int, tp: int = 1):
        model_config = build_model_config("debugmodel_moe")
        runtime_config = Trainer.Config(
            model=model_config,
            training=TrainingConfig(
                max_context_length=model_config.max_context_length,
                disable_cuda_graphs=True,
            ),
            parallelism=ParallelismConfig(
                expert_parallel_degree=ep,
                tensor_parallel_degree=tp,
            ),
        )
        return model_config, runtime_config

    def test_all_moe_layers_divisible(self):
        self._config(ep=8)

    def test_later_moe_layer_not_divisible(self):
        model_config = build_model_config("debugmodel_moe")
        moe = model_config.layers[1].moe
        assert moe is not None
        moe.num_experts = 63

        with self.assertRaisesRegex(
            ValueError,
            r"layers\.1\.moe\.num_experts \(63\).*expert_parallel_degree \(8\)",
        ):
            Trainer.Config(
                model=model_config,
                training=TrainingConfig(
                    max_context_length=model_config.max_context_length,
                    disable_cuda_graphs=True,
                ),
                parallelism=ParallelismConfig(expert_parallel_degree=8),
            )

    def test_tensor_parallel_requires_expert_parallel(self):
        with self.assertRaisesRegex(
            ValueError,
            r"expert_parallel_degree \(1\).*tensor_parallel_degree \(2\)",
        ):
            self._config(ep=1, tp=2)

    def test_tensor_parallel_preserves_explicit_expert_parallel(self):
        _, runtime_config = self._config(ep=4, tp=2)

        self.assertEqual(runtime_config.parallelism.expert_parallel_degree, 4)

    def test_moe_without_tensor_parallel_preserves_sequence_parallel_config(self):
        _, runtime_config = self._config(ep=1, tp=1)

        self.assertTrue(runtime_config.parallelism.enable_sequence_parallel)

    def test_dense_tensor_parallel_does_not_require_expert_parallel(self):
        model_config = build_model_config("debugmodel")
        runtime_config = Trainer.Config(
            model=model_config,
            training=TrainingConfig(
                max_context_length=model_config.max_context_length,
                disable_cuda_graphs=True,
            ),
            parallelism=ParallelismConfig(tensor_parallel_degree=2),
        )
        self.assertEqual(runtime_config.parallelism.expert_parallel_degree, 1)
        self.assertTrue(runtime_config.parallelism.enable_sequence_parallel)

    def test_model_sharding_is_resolved_from_parallelism(self):
        shared_config = build_model_config("debugmodel_moe")
        trainer_config = copy.deepcopy(shared_config)
        generator_config = copy.deepcopy(shared_config)

        trainer_config.set_sharding_(
            ParallelismConfig(
                expert_parallel_degree=2,
                enable_sequence_parallel=False,
            )
        )
        generator_config.set_sharding_(
            ParallelismConfig(
                expert_parallel_degree=1,
                enable_sequence_parallel=True,
            )
        )

        shared_moe = shared_config.layers[0].moe
        trainer_moe = trainer_config.layers[0].moe
        generator_moe = generator_config.layers[0].moe
        assert shared_moe is not None
        assert trainer_moe is not None
        assert generator_moe is not None
        self.assertIsNone(shared_config.tok_embeddings.sharding_config)
        self.assertIsNotNone(trainer_moe.routed_experts.w13.sharding_config)
        self.assertIsNone(generator_moe.routed_experts.w13.sharding_config)
        self.assertNotEqual(
            trainer_config.tok_embeddings.sharding_config,
            generator_config.tok_embeddings.sharding_config,
        )

    def test_resolved_model_sharding_must_match_parallelism(self):
        model_config = build_model_config("debugmodel_moe")
        model_config.set_sharding_(ParallelismConfig(expert_parallel_degree=2))

        with self.assertRaisesRegex(
            ValueError, "sharding does not match expert_parallel_degree"
        ):
            Trainer.Config(
                model=model_config,
                training=TrainingConfig(
                    max_context_length=model_config.max_context_length,
                    disable_cuda_graphs=True,
                ),
                parallelism=ParallelismConfig(expert_parallel_degree=1),
            )


class TestPermute(unittest.TestCase):
    """Test AllToAllTokenDispatcher._permute which reorders tokens from rank-major to expert-major layout.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """

    def _make_dispatcher(self) -> AllToAllTokenDispatcher:
        """Create a minimal AllToAllTokenDispatcher for testing _permute."""
        cfg = AllToAllTokenDispatcher.Config(num_experts=1, top_k=1)
        return AllToAllTokenDispatcher(cfg)

    def _permute(self, tokens_per_expert_group, experts_per_rank, num_ranks):
        """Helper that calls _permute and returns (permuted_indices, num_tokens_per_expert)."""
        dispatcher = self._make_dispatcher()
        mock_mesh = unittest.mock.MagicMock()
        mock_mesh.size.return_value = num_ranks
        total = tokens_per_expert_group.sum().item()
        dummy_input = torch.zeros(total, 1)
        with unittest.mock.patch.object(
            AllToAllTokenDispatcher,
            "ep_mesh",
            new_callable=unittest.mock.PropertyMock,
            return_value=mock_mesh,
        ):
            _, _, permuted_indices, num_tokens_per_expert = dispatcher._permute(
                dummy_input, tokens_per_expert_group
            )
        return permuted_indices, num_tokens_per_expert

    def test_basic_2ranks_2experts(self):
        # 2 ranks, 2 experts per rank
        # tokens_per_expert_group: [r0e0, r0e1, r1e0, r1e1] = [2, 3, 1, 4]
        tokens_per_expert_group = torch.tensor([2, 3, 1, 4])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major layout: e0r0(2), e0r1(1), e1r0(3), e1r1(4)
        # Input positions:
        #   r0e0: [0, 1], r0e1: [2, 3, 4], r1e0: [5], r1e1: [6, 7, 8, 9]
        # Output order:
        #   e0r0: [0, 1], e0r1: [5], e1r0: [2, 3, 4], e1r1: [6, 7, 8, 9]
        expected_indices = torch.tensor([0, 1, 5, 2, 3, 4, 6, 7, 8, 9])
        torch.testing.assert_close(permuted_indices, expected_indices)

        # num_tokens_per_expert: sum across ranks for each expert
        # e0: r0e0 + r1e0 = 2 + 1 = 3, e1: r0e1 + r1e1 = 3 + 4 = 7
        expected_num_tokens = torch.tensor([3, 7])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_single_rank(self):
        # 1 rank, 3 experts: no reordering needed
        tokens_per_expert_group = torch.tensor([4, 2, 5])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=1
        )

        expected_indices = torch.arange(11)
        torch.testing.assert_close(permuted_indices, expected_indices)
        torch.testing.assert_close(num_tokens_per_expert, tokens_per_expert_group)

    def test_single_expert(self):
        # 3 ranks, 1 expert per rank: no reordering needed
        tokens_per_expert_group = torch.tensor([3, 5, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=1, num_ranks=3
        )

        expected_indices = torch.arange(10)
        torch.testing.assert_close(permuted_indices, expected_indices)

        # Single expert gets all tokens
        expected_num_tokens = torch.tensor([10])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_zero_tokens_for_some_experts(self):
        # 2 ranks, 2 experts, some with zero tokens
        # [r0e0, r0e1, r1e0, r1e1] = [0, 3, 2, 0]
        tokens_per_expert_group = torch.tensor([0, 3, 2, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        # Expert-major: e0r0(0), e0r1(2), e1r0(3), e1r1(0)
        # Input positions: r0e0: [], r0e1: [0, 1, 2], r1e0: [3, 4], r1e1: []
        # Output order: e0r0: [], e0r1: [3, 4], e1r0: [0, 1, 2], e1r1: []
        expected_indices = torch.tensor([3, 4, 0, 1, 2])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([2, 3])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_all_zero_tokens(self):
        tokens_per_expert_group = torch.tensor([0, 0, 0, 0])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=2
        )

        self.assertEqual(permuted_indices.numel(), 0)
        expected_num_tokens = torch.tensor([0, 0])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_uniform_distribution(self):
        # 3 ranks, 2 experts, uniform token counts
        # [r0e0, r0e1, r1e0, r1e1, r2e0, r2e1] = [2, 2, 2, 2, 2, 2]
        tokens_per_expert_group = torch.tensor([2, 2, 2, 2, 2, 2])
        permuted_indices, num_tokens_per_expert = self._permute(
            tokens_per_expert_group, experts_per_rank=2, num_ranks=3
        )

        # Expert-major: e0r0(2), e0r1(2), e0r2(2), e1r0(2), e1r1(2), e1r2(2)
        # Input positions:
        #   r0e0: [0,1], r0e1: [2,3], r1e0: [4,5], r1e1: [6,7], r2e0: [8,9], r2e1: [10,11]
        # Output: e0r0[0,1], e0r1[4,5], e0r2[8,9], e1r0[2,3], e1r1[6,7], e1r2[10,11]
        expected_indices = torch.tensor([0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11])
        torch.testing.assert_close(permuted_indices, expected_indices)

        expected_num_tokens = torch.tensor([6, 6])
        torch.testing.assert_close(num_tokens_per_expert, expected_num_tokens)

    def test_permutation_is_valid(self):
        # The output should be a permutation of [0, total)
        tokens_per_expert_group = torch.tensor([3, 1, 4, 1, 5, 9])
        permuted_indices, _ = self._permute(
            tokens_per_expert_group, experts_per_rank=3, num_ranks=2
        )

        total = tokens_per_expert_group.sum().item()
        self.assertEqual(permuted_indices.numel(), total)
        self.assertEqual(
            set(permuted_indices.tolist()),
            set(range(total)),
        )


class TestAllGatherDispatch(unittest.TestCase):
    """Test AllToAllTokenDispatcher's CUDA graph (all-gather) dispatch and combine.

    The collectives are mocked, so each call sees the tensors of all EP ranks.
    """

    num_experts = 4
    ep_size = 2

    def _dispatch_and_combine(self, ep_rank, x_GD, scores_GK, expert_ids_GK):
        """Run one rank with experts ``y = (global_expert_id + 1) * x``; return its partial sum."""
        dispatcher = AllToAllTokenDispatcher.Config(
            num_experts=self.num_experts, top_k=expert_ids_GK.shape[1]
        ).build()
        mock_mesh = unittest.mock.MagicMock()
        mock_mesh.size.return_value = self.ep_size
        mock_mesh.get_local_rank.return_value = ep_rank
        num_tokens = x_GD.shape[0] // self.ep_size
        local_slice = slice(ep_rank * num_tokens, (ep_rank + 1) * num_tokens)
        partial_outs = []
        with (
            unittest.mock.patch.object(
                AllToAllTokenDispatcher,
                "ep_mesh",
                new_callable=unittest.mock.PropertyMock,
                return_value=mock_mesh,
            ),
            unittest.mock.patch.object(
                spmd, "all_gather", side_effect=[x_GD, scores_GK, expert_ids_GK]
            ),
            unittest.mock.patch.object(
                spmd,
                "reduce_scatter",
                side_effect=lambda x, *args, **kwargs: partial_outs.append(x) or x,
            ),
        ):
            (
                routed_input_ND,
                num_tokens_per_local_expert_e,
                metadata,
            ) = dispatcher._all_gather_dispatch(
                x_GD[local_slice], scores_GK[local_slice], expert_ids_GK[local_slice]
            )
            num_local_experts = self.num_experts // self.ep_size
            # Rows past the local ones get id num_local_experts; combine drops them.
            local_expert_ids_N = torch.searchsorted(
                num_tokens_per_local_expert_e.cumsum(0),
                torch.arange(routed_input_ND.shape[0]),
                right=True,
            )
            routed_output_ND = routed_input_ND * (
                ep_rank * num_local_experts + local_expert_ids_N + 1
            ).unsqueeze(-1)
            dispatcher.combine(routed_output_ND, metadata, x_GD[local_slice])
        return metadata, num_tokens_per_local_expert_e, partial_outs[0]

    def test_docstring_example_on_rank_1(self):
        x_GD = torch.arange(4.0).reshape(4, 1)
        expert_ids_GK = torch.tensor([[3], [0], [2], [1]])
        metadata, num_tokens_per_local_expert_e, _ = self._dispatch_and_combine(
            1, x_GD, torch.ones(4, 1), expert_ids_GK
        )

        torch.testing.assert_close(
            metadata.token_indices_experts_sorted_N, torch.tensor([2, 0, 4, 4])
        )
        torch.testing.assert_close(num_tokens_per_local_expert_e, torch.tensor([1, 1]))

    def test_partial_sums_add_up_to_routed_output(self):
        # 3 tokens per rank, top-2; token 5 routes both picks to rank 0.
        x_GD = torch.arange(1.0, 7.0).reshape(6, 1)
        scores_GK = torch.tensor(
            [[0.5, 0.25], [1.0, 2.0], [0.75, 0.5], [2.0, 1.0], [0.25, 0.5], [1.0, 1.0]]
        )
        expert_ids_GK = torch.tensor([[0, 3], [1, 2], [2, 3], [3, 0], [1, 2], [0, 1]])

        partial_sums = [
            self._dispatch_and_combine(ep_rank, x_GD, scores_GK, expert_ids_GK)[2]
            for ep_rank in range(self.ep_size)
        ]

        expected_GD = (scores_GK * (expert_ids_GK + 1)).sum(-1, keepdim=True) * x_GD
        torch.testing.assert_close(sum(partial_sums), expected_GD)


if __name__ == "__main__":
    unittest.main()
