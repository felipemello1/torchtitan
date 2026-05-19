# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from torchtitan.experiments.rl.envs import EnvStep
from torchtitan.experiments.rl.envs.token_env import PromptState, TokenEnv, TokenStep
from torchtitan.experiments.rl.generation_scheduler import GenerationScheduler
from torchtitan.experiments.rl.grpo import Provisioner, RLTrainer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.metrics.rl import (
    build_rollout_metrics,
    rename_metric_prefix,
)
from torchtitan.experiments.rl.replay import (
    has_advantage_signal,
    ReplayBuffer,
    ReplayGroup,
    rollouts_to_replay_samples,
)
from torchtitan.experiments.rl.rollout_logging import RolloutSampleLogger
from torchtitan.experiments.rl.rollouts import do_single_rollout
from torchtitan.experiments.rl.sampling import SamplingConfig
from torchtitan.experiments.rl.types import (
    Completion,
    ReplaySample,
    RolloutOutput,
    RolloutStatus,
    RolloutTurn,
)


def test_sampling_config_default_is_one_completion():
    assert SamplingConfig().n == 1


def test_provisioner_respects_parent_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    provisioner = Provisioner(total_gpus=4)

    provisioner.allocate(2)()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5"
    provisioner.allocate(2)()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6,7"


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
                policy_version=0,
            ),
            RolloutTurn(
                prompt_token_ids=[10, 11, 20, 12],
                response_token_ids=[21, 22],
                response_logprobs=[-0.3, -0.4],
                policy_version=0,
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


def test_token_env_keeps_truncated_response_message_for_logging():
    class EnvStub:
        async def reset(self):
            raise AssertionError("reset is not needed for a length-stop step")

        async def step(self, assistant_message):
            raise AssertionError("length-stop responses should not step the env")

        async def close(self):
            pass

    class RendererStub:
        def parse_response(self, token_ids):
            assert token_ids == [10, 11]
            return SimpleNamespace(
                content="partial answer",
                reasoning_content=None,
                tool_calls=None,
            )

    async def run() -> None:
        token_step = await TokenEnv(EnvStub(), RendererStub()).step(
            Completion(
                policy_version=0,
                token_ids=[10, 11],
                token_logprobs=[-0.1, -0.2],
                finish_reason="length",
            )
        )

        assert token_step.env_step.status == RolloutStatus.TRUNCATED
        assert token_step.response_messages == [
            {"role": "assistant", "content": "partial answer"}
        ]

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
                    policy_version=0,
                )
            ],
        )
    ]

    metrics = build_rollout_metrics(
        rollouts,
        generation_metrics=[],
        prefix="rollout",
    )
    renamed = [
        rename_metric_prefix(metric, old_prefix="rollout/", new_prefix="validation/")
        for metric in metrics
    ]
    aggregate = m.MetricsProcessor._aggregate_metrics(renamed)

    assert aggregate["validation/reward/_mean"] == 1.0
    assert aggregate["validation/response_length/mean"] == 1.0


def test_generation_scheduler_coalesces_same_tick_requests():
    async def run() -> None:
        calls: list[list[list[int]]] = []

        async def generate_batch(
            prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            return (
                [
                    Completion(
                        policy_version=0,
                        token_ids=[idx],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for idx, _prompt in enumerate(prompts)
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)

        completions = await asyncio.gather(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="a",
            ),
            scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="b",
            ),
        )

        assert calls == [[[1], [2]]]
        assert [completion.token_ids for completion in completions] == [[0], [1]]

    asyncio.run(run())


def test_generation_scheduler_pauses_new_admission_until_resume():
    async def run() -> None:
        calls: list[list[list[int]]] = []
        active_started = asyncio.Event()
        finish_active = asyncio.Event()
        version = 0

        async def generate_batch(
            prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            if prompts == [[1]]:
                active_started.set()
                await finish_active.wait()
            return (
                [
                    Completion(
                        policy_version=version,
                        token_ids=[prompt[0]],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for prompt in prompts
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        first = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="first",
            )
        )
        await active_started.wait()

        pause = asyncio.create_task(scheduler.pause_for_weight_sync())
        await asyncio.sleep(0)
        assert not pause.done()

        second = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="second",
            )
        )
        await asyncio.sleep(0)
        assert calls == [[[1]]]
        assert not second.done()

        finish_active.set()
        await pause
        assert not second.done()

        version = 1
        await scheduler.resume_after_weight_sync()

        assert (await first).policy_version == 0
        assert (await second).policy_version == 1
        assert calls == [[[1]], [[2]]]

    asyncio.run(run())


def test_generation_scheduler_close_drops_cancelled_queued_request():
    async def run() -> None:
        calls: list[list[list[int]]] = []

        async def generate_batch(
            prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            return (
                [
                    Completion(
                        policy_version=0,
                        token_ids=[prompt[0]],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for prompt in prompts
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        request = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="queued",
            )
        )
        await asyncio.sleep(0)

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await scheduler.close()

        assert calls == []

    asyncio.run(run())


def test_generation_scheduler_close_waits_for_active_generation():
    async def run() -> None:
        calls: list[list[list[int]]] = []
        active_started = asyncio.Event()
        finish_active = asyncio.Event()

        async def generate_batch(
            prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            calls.append([list(prompt) for prompt in prompts])
            active_started.set()
            await finish_active.wait()
            return (
                [
                    Completion(
                        policy_version=0,
                        token_ids=[prompt[0]],
                        token_logprobs=[-0.1],
                        finish_reason="stop",
                    )
                    for prompt in prompts
                ],
                [],
            )

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        request = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="active",
            )
        )
        await active_started.wait()

        close = asyncio.create_task(scheduler.close())
        await asyncio.sleep(0)
        assert not close.done()
        with pytest.raises(RuntimeError, match="generation scheduler is closed"):
            await scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="closed",
            )

        finish_active.set()
        await close

        assert (await request).token_ids == [1]
        assert calls == [[[1]]]

    asyncio.run(run())


def test_generation_scheduler_propagates_generator_exception_to_pending_batch():
    async def run() -> None:
        async def generate_batch(
            prompts: list[list[int]],
            sampling: SamplingConfig,
        ) -> tuple[list[Completion], list[m.Metric]]:
            raise ValueError(f"bad batch: {prompts}")

        scheduler = GenerationScheduler(generate_batch)
        sampling = SamplingConfig(n=1, temperature=0.0, top_p=1.0, max_tokens=4)
        first = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[1],
                sampling=sampling,
                request_id="first",
            )
        )
        second = asyncio.create_task(
            scheduler.submit(
                prompt_token_ids=[2],
                sampling=sampling,
                request_id="second",
            )
        )

        with pytest.raises(ValueError, match="bad batch"):
            await first
        with pytest.raises(ValueError, match="bad batch"):
            await second
        await scheduler.close()

    asyncio.run(run())


def test_empty_collate_produces_no_loss_dummy_row():
    batch = RLTrainer._collate_samples([])

    assert batch.token_ids.shape == (1, 1)
    assert batch.seq_lens == [1]
    assert not batch.loss_mask.any()


def test_rollout_sample_logger_caps_groups_per_step(tmp_path):
    logger = RolloutSampleLogger(str(tmp_path), max_groups_per_step=1)
    rollouts = [
        RolloutOutput(
            group_id=f"g{idx}",
            sample_idx=0,
            status=RolloutStatus.COMPLETED,
            reward=1.0,
            turns=[
                RolloutTurn(
                    prompt_token_ids=[1, 2],
                    response_token_ids=[3],
                    response_logprobs=[-0.1],
                    policy_version=idx,
                    prompt_messages=[{"role": "user", "content": "sort"}],
                    response_messages=[{"role": "assistant", "content": "done"}],
                )
            ],
        )
        for idx in range(2)
    ]

    logger.write(step=3, phase="train", rollouts=rollouts)
    logger.write(step=3, phase="train", rollouts=rollouts)
    logger.write(step=3, phase="validation", rollouts=rollouts)

    records = [
        json.loads(line)
        for line in (tmp_path / "rollout_samples.jsonl").read_text().splitlines()
    ]

    assert [(record["phase"], record["group_id"]) for record in records] == [
        ("train", "g0"),
        ("validation", "g0"),
    ]
    assert records[0]["turns"][0]["messages"][0] == {
        "role": "user",
        "content": "sort",
    }


def test_has_advantage_signal_detects_nonzero_group_signal():
    def sample(group_id: str, advantage: float) -> ReplaySample:
        return ReplaySample(
            token_ids=[1, 2],
            loss_mask=[0, 1],
            behavior_logprobs=[0.0, -0.1],
            advantage=advantage,
            group_id=group_id,
            sample_idx=0,
            behavior_version=0,
            reward=advantage,
        )

    assert not has_advantage_signal([sample("zero", 0.0)])
    assert has_advantage_signal([sample("signal", 0.0), sample("signal", 0.5)])


def test_replay_buffer_blocks_until_batch_is_ready():
    async def run() -> None:
        buffer = ReplayBuffer(max_groups=1)

        def sample(idx: int) -> ReplaySample:
            return ReplaySample(
                token_ids=[idx, idx + 10],
                loss_mask=[0, 1],
                behavior_logprobs=[0.0, -0.1],
                advantage=1.0,
                group_id=f"g{idx}",
                sample_idx=0,
                behavior_version=0,
                reward=1.0,
            )

        async def producer() -> None:
            await buffer.put(
                ReplayGroup(
                    group_id="g0",
                    samples=[sample(0)],
                    rollouts=[],
                    behavior_version=0,
                    max_behavior_version=0,
                )
            )
            await buffer.put(
                ReplayGroup(
                    group_id="g1",
                    samples=[sample(1)],
                    rollouts=[],
                    behavior_version=0,
                    max_behavior_version=0,
                )
            )
            await buffer.close()

        producer_task = asyncio.create_task(producer())
        batch = await buffer.get_batch(min_samples=2, train_version=0)
        await producer_task

        assert [sample.group_id for sample in batch.samples] == ["g0", "g1"]
        assert batch.stats.num_groups == 2
        assert batch.stats.num_loss_tokens == 2

    asyncio.run(run())
