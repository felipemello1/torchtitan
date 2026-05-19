# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.envs import EnvStep
from torchtitan.experiments.rl.envs.token_env import PromptState, TokenStep
from torchtitan.experiments.rl.grpo import _CompletionBatcher, _rename_metric, RLTrainer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.replay import (
    ReplayGroup,
    RolloutQueue,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.rollouts import do_single_rollout
from torchtitan.experiments.rl.types import (
    Completion,
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
)


def test_rollout_to_replay_sample_masks_multiturn_prefix_continuation():
    rollout = RolloutOutput(
        group_id="g0",
        sample_idx=0,
        status=RolloutStatus.COMPLETED,
        reward=1.0,
        turns=[
            RolloutTurn(
                prompt_token_ids=[10, 11],
                response_token_ids=[20],
                response_logprobs=[-0.2],
            ),
            RolloutTurn(
                prompt_token_ids=[10, 11, 20, 12],
                response_token_ids=[21, 22],
                response_logprobs=[-0.3, -0.4],
            ),
        ],
    )

    [sample] = rollouts_to_replay_samples([rollout])

    assert sample.token_ids == [10, 11, 20, 12, 21, 22]
    assert sample.loss_mask == [0, 0, 1, 0, 1, 1]
    assert sample.behavior_logprobs == [0.0, 0.0, -0.2, 0.0, -0.3, -0.4]


def test_max_turn_truncation_does_not_train_on_nonterminal_reward():
    class TokenEnvStub:
        async def initial_prompt(self) -> PromptState:
            return PromptState(token_ids=[1, 2], messages=[])

        async def step(self, completion: Completion) -> TokenStep:
            return TokenStep(
                env_step=EnvStep(
                    reward=0.75,
                    reward_components={"shaped": 0.75},
                    done=False,
                ),
                response_messages=[],
                next_prompt=PromptState(token_ids=[1, 2, 3], messages=[]),
            )

    async def run() -> None:
        async def completion_fn(
            prompt_token_ids: list[int],
            sampling: SamplingConfig,
            request_id: str,
        ) -> Completion:
            return Completion(
                policy_version=0,
                prompt_idx=0,
                text="partial",
                token_ids=[3],
                token_logprobs=[-0.1],
                finish_reason="stop",
            )

        rollout = await do_single_rollout(
            token_env=TokenEnvStub(),
            completion_fn=completion_fn,
            sampling=SamplingConfig(n=1),
            group_id="g0",
            sample_idx=0,
            max_turns=1,
        )

        assert rollout.status == RolloutStatus.TRUNCATED
        assert rollout.reward == 0.0
        assert rollout.reward_components == {"max_turns": 1.0}

    asyncio.run(run())


def test_validation_metric_rename_handles_reward_summary_keys():
    rollouts = [
        RolloutOutput(
            group_id="g0",
            sample_idx=0,
            status=RolloutStatus.COMPLETED,
            reward=1.0,
            turns=[
                RolloutTurn(
                    prompt_token_ids=[1, 2],
                    response_token_ids=[3],
                    response_logprobs=[-0.1],
                )
            ],
        )
    ]

    metrics = RLTrainer._build_rollout_metrics(
        rollouts,
        generation_metrics=[],
        prefix="rollout",
    )
    renamed = [
        _rename_metric(metric, old_prefix="rollout/", new_prefix="validation/")
        for metric in metrics
    ]
    aggregate = m.MetricsProcessor._aggregate_metrics(renamed)

    assert aggregate["validation/reward/_mean"] == 1.0
    assert aggregate["validation/response_length/mean"] == 1.0


def test_completion_batcher_coalesces_same_tick_requests():
    async def run() -> None:
        calls: list[list[list[int]]] = []

        async def generate_batch(
            prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> list[Completion]:
            calls.append([list(prompt) for prompt in prompts])
            return [
                Completion(
                    policy_version=0,
                    prompt_idx=idx,
                    text=str(idx),
                    token_ids=[idx],
                    token_logprobs=[-0.1],
                    finish_reason="stop",
                )
                for idx, _prompt in enumerate(prompts)
            ]

        batcher = _CompletionBatcher(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)

        completions = await asyncio.gather(
            batcher.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="a",
            ),
            batcher.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="b",
            ),
        )

        assert calls == [[[1], [2]]]
        assert [completion.token_ids for completion in completions] == [[0], [1]]

    asyncio.run(run())


def test_empty_collate_produces_no_loss_dummy_row():
    batch = RLTrainer._collate_samples([])

    assert batch.token_ids.shape == (1, 1)
    assert batch.seq_lens == [1]
    assert not batch.loss_mask.any()


def test_rollout_queue_drains_bounded_fifo_until_close():
    async def run() -> None:
        queue = RolloutQueue(max_groups=1)

        def sample(idx: int) -> ReplaySample:
            return ReplaySample(
                token_ids=[idx, idx + 10],
                loss_mask=[0, 1],
                behavior_logprobs=[0.0, -0.1],
                advantages=[0.0, 1.0],
                group_id=f"g{idx}",
                sample_idx=0,
                behavior_version=0,
                reward=1.0,
            )

        async def producer() -> None:
            await queue.put(
                ReplayGroup(
                    group_id="g0",
                    samples=[sample(0)],
                    behavior_version=0,
                    train_step=1,
                )
            )
            await queue.put(
                ReplayGroup(
                    group_id="g1",
                    samples=[sample(1)],
                    behavior_version=0,
                    train_step=1,
                )
            )
            await queue.close()

        producer_task = asyncio.create_task(producer())
        samples, stats = await queue.get_all(train_version=0)
        await producer_task

        assert [sample.group_id for sample in samples] == ["g0", "g1"]
        assert stats.num_groups == 2
        assert stats.num_loss_tokens == 2

    asyncio.run(run())
