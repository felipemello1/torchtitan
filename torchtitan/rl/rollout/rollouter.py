# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np

from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.renderer import RendererConfig
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.curriculum import Curriculum
from torchtitan.rl.rollout.data import RLDataset, RolloutDataLoader
from torchtitan.rl.rollout.message_env import MessageEnv
from torchtitan.rl.rollout.rubric import Rubric, RubricOutput
from torchtitan.rl.rollout.token_env import TokenEnv
from torchtitan.rl.rollout.types import (
    error_entry,
    GenerateFn,
    ReleaseSessionFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)

if TYPE_CHECKING:
    from renderers import Renderer

    from torchtitan.rl.actors.system import ActorSystem

    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.rl.generator.types import SamplingConfig


logger = logging.getLogger(__name__)

# How long `Rollouter.close` waits for in-flight `run_group` calls before stopping their workers anyway,
# and `Rollouter.cancel_group` for a cancelled group to end. Drains took under 0.3 s once the generators
# closed; 30 s only covers a group stuck in a slow env step.
_RUN_GROUP_DRAIN_TIMEOUT_S = 30.0


class Rollouter(Configurable):
    """Turns a problem (train/val datasets and a rollout worker config) into scored rollouts, the
    RL training data.

    Like a `Dataloader` turns a `Dataset` into training batches, a `Rollouter`
    turns a problem into rollouts: its worker builds the envs, drives them against the inference engine
    (via a `generate_fn` the controller provides), and scores the results with `score_group`.

    The flow for one prompt group: the controller passes a `generate_fn` callable; each rollout
    drives its own calls, so the generator runs a whole group's calls together in one continuous
    batch.

        index, sample = rollouter.get_training_sample()
        group = await rollouter.run_group_rollouts(     # one worker runs and scores the group
            generate_fn=generate_fn, release_session_fn=release_session_fn,
            sample=sample, group_id=index,
            group_size=N, sampling=sampling)

    Example:
        rollouter = Rollouter.Config(
            training_dataloader=IterableRolloutDataLoader.Config(
                dataset=MyDataset.Config()
            ),
            validation_dataset=MyDataset.Config(),
            worker=MessageEnvRolloutWorker.Config(
                rubric=Rubric.Config(
                    reward_fns=[RewardCorrect.Config(), RewardFormat.Config(weight=0.3)]
                ),
                message_env=MyEnv.Config(),
                token_env=TokenEnv.Config(max_rollout_tokens=32768),
            ),
        ).build()

    Customization:
        Rollouter supports customization at several levels:
          - Sample source: provide source configs for training and validation.
          - Task state that follows training (a difficulty level, a schedule on the
            train step): set `curriculum`; see `Curriculum`.
          - Group execution, coarse: override `_run_group_rollouts` for your own
            orchestration. The worker then becomes optional -- but override
            `setup_async` too, or the worker pool is still spawned unused, and
            `cancel_group`: the base one stops the group on its rollout worker.
          - Group execution, fine: keep the stock orchestration and point `worker`
            at a `MessageEnvRolloutWorker.Config` subclass, overriding only what you need
            (`make_env_group`, `score_group`, `run_group`). A worker whose env isn't a
            `MessageEnv` subclasses `BaseRolloutWorker` and implements `run_group`.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        training_dataloader: RolloutDataLoader.Config
        """Builds a checkpointable iterator of identified training inputs."""

        validation_dataset: RLDataset.Config
        """Builds an iterable validation dataset."""

        worker: BaseRolloutWorker.Config
        """How a rollout group is built, driven, scored and advantaged. Selects the
        `BaseRolloutWorker` subclass by config type; one worker is built per pool process."""

        rollouts_per_worker: int = 128
        """Concurrent training rollouts per rollout worker process.
        It sets the pool size, rounded up: 300 concurrent training rollouts / 128 per worker -> 3 workers.

        Higher starts fewer processes (~0.8 GiB of private memory each after setup), and more
        rollouts share each worker's event loop. Too high shows in
        `rollout_worker/event_loop_lag_ms_per_s` and `rollout_worker/gc_pause_ms`: see "Rollout
        worker metrics" in `torchtitan/rl/observability/metrics/README.md`."""

        num_threads_per_worker: int = 32
        """Threads in each worker process's default thread pool, the pool behind every
        `asyncio.to_thread` call in that process.

        All rollouts in the worker share these threads, and a blocking call holds its thread until it
        returns. Threads busy on average = blocking calls per second x seconds each, e.g. 0.2 calls per
        second x 10 s each = 2 threads. Bursts fill more: leave ~4x headroom when a few threads are busy
        (8 here), ~2x from ~10 busy up. 32 is Python's own default on hosts with 28+ CPUs, and leaves
        room for envs whose calls block threads for seconds. The pool starts a thread only when a call
        finds the others busy, and each started thread costs ~4 MiB. Too few shows in
        `rollout_worker/executor_queue_wait_ms`: see "Rollout worker metrics" in
        `torchtitan/rl/observability/metrics/README.md`."""

        curriculum: Curriculum.Config = field(default_factory=Curriculum.Config)
        """Rewrites training samples from the train step and trained groups' results; the default changes nothing."""

        def __post_init__(self) -> None:
            if self.num_threads_per_worker < 1:
                raise ValueError(
                    "num_threads_per_worker must be at least 1, got "
                    f"{self.num_threads_per_worker}"
                )

    def __init__(self, config: Config) -> None:
        self._config = config
        training_dataloader = config.training_dataloader.build()
        if not isinstance(training_dataloader, RolloutDataLoader):
            raise ValueError("training_dataloader must build a RolloutDataLoader")
        self._training_dataloader = iter(training_dataloader)
        self._curriculum: Curriculum = config.curriculum.build()
        self._step = 0
        # Summaries of finished training groups by group id, until those groups are trained.
        self._summaries: dict[int, object] = {}

        # The worker pool's actor handle and the actor system that spawned it.
        self._worker_actors: Any | None = None
        self._actor_system: ActorSystem | None = None
        # `run_group` calls still running on a worker, including those whose caller was cancelled.
        self._inflight_run_group_calls: set[asyncio.Future[RolloutGroup]] = set()

    def get_training_sample(self) -> tuple[int, object]:
        """Return the next globally identified input from the training loader."""
        return next(self._training_dataloader)

    def commit_training_sample_ids(self, sample_ids: Iterable[int]) -> None:
        """Mark training samples as safe to omit from the next checkpoint, and hand
        their groups' summaries to the curriculum."""
        sample_ids = list(sample_ids)
        self._training_dataloader.commit(sample_ids)
        # A group whose rollout raised has no summary.
        summaries = [
            self._summaries.pop(sample_id)
            for sample_id in sample_ids
            if sample_id in self._summaries
        ]
        self._curriculum.update(step=self._step, summaries=summaries)

    def curriculum_metrics(self) -> list[m.Metric]:
        """The curriculum's metrics after its last update, logged with the train step."""
        return self._curriculum.metrics()

    def get_validation_samples(self, steps: int) -> list[object]:
        """Materialize one fresh validation pass, optionally bounded by steps."""
        validation_data = self._config.validation_dataset.build()
        try:
            iterator = iter(validation_data)
        except TypeError as error:
            raise ValueError("validation_dataset must build an iterable") from error
        if steps == -1:
            return list(iterator)
        return list(itertools.islice(iterator, steps))

    async def setup_async(
        self,
        *,
        actor_system: ActorSystem,
        num_workers: int,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
        # Constructor args of each rollout worker known only at setup, e.g. a server address.
        # TODO: VerifiersRollouter is the only user. Once its env runs in the rollout worker, remove
        # this, `spawn_rollout_workers(**kwargs)` and `RolloutWorkerActor(worker_kwargs=...)`.
        worker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Spawn `num_workers` rollout workers through `actor_system` and initialize them."""
        if self._worker_actors is not None:
            raise RuntimeError("rollout worker pool is already initialized")

        self._actor_system = actor_system
        self._worker_actors = await actor_system.spawn_rollout_workers(
            self._config.worker,
            num_workers=num_workers,
            num_threads=self._config.num_threads_per_worker,
            **(worker_kwargs or {}),
        )
        await self._worker_actors.setup_async.call(
            tokenizer_config=tokenizer_config,
            renderer_config=renderer_config,
            hf_assets_path=hf_assets_path,
            vision_processor_config=vision_processor_config,
        )

    async def close(self) -> None:
        """Wait for in-flight `run_group` calls, then stop the rollout worker pool.

        Close the generators first, so pending `generate_fn` calls fail fast:

            generators closed -> generate_fn raises -> run_group returns -> stop_rollout_workers()
            run_group still running after _RUN_GROUP_DRAIN_TIMEOUT_S       -> stop_rollout_workers()

        Why wait: if a worker stops while a generator (or the generator router) still owes it a reply,
        Monarch fails that actor and the finished job exits 130.
        """
        actor_system = self._actor_system
        self._worker_actors = None
        self._actor_system = None
        calls = tuple(self._inflight_run_group_calls)
        if calls:
            _, pending = await asyncio.wait(calls, timeout=_RUN_GROUP_DRAIN_TIMEOUT_S)
            if pending:
                logger.warning(
                    "Stopping rollout workers with %d run_group calls still running; "
                    "the job may exit 130",
                    len(pending),
                )
                for call in pending:
                    call.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        # TODO: also wait for the generate calls of groups that `cancel_group` stopped. Their worker no longer
        # waits for them, so a generator router that answers after the workers stop may make the job exit 130.
        if actor_system is not None:
            await actor_system.stop_rollout_workers()

    async def sync_log_step(self, step: int) -> None:
        """Record the step for the curriculum and propagate it to every rollout worker."""
        self._step = step
        if self._worker_actors is not None:
            await self._worker_actors.sync_log_step.call(step)

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Send one prompt group to a rollout worker and return its scored group.

        Training groups also go through the curriculum: `prepare` before, `summarize` after.

        Args:
            generate_fn: Async callable that returns a Completion given a prompt.
            release_session_fn: Async callable each rollout calls after its last turn, to
                release resources associated to that session. E.g. generator's kv-cache.
            sample: Dataset sample shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.

        Returns:
            One scored `RolloutGroup`.
        """
        # Validation groups (negative ids) skip the curriculum, so every pass plays the same settings.
        if group_id >= 0:
            sample = self._curriculum.prepare(sample, step=self._step)
        group = await self._run_group_rollouts(
            generate_fn=generate_fn,
            release_session_fn=release_session_fn,
            sample=sample,
            group_id=group_id,
            group_size=group_size,
            sampling=sampling,
        )
        if group_id >= 0:
            self._summaries[group_id] = self._curriculum.summarize(sample, group)
        return group

    async def _run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Run one group on a rollout worker; override for your own orchestration."""
        if self._worker_actors is None:
            raise RuntimeError("rollout worker pool is not initialized")

        # `choose` (see ActorSystem's call convention) sends the call to the `run_group`
        # endpoint of one worker in the pool.
        call = asyncio.ensure_future(
            self._worker_actors.run_group.choose(
                generate_fn=generate_fn,
                release_session_fn=release_session_fn,
                sample=sample,
                group_id=group_id,
                group_size=group_size,
                sampling=sampling,
            )
        )
        self._inflight_run_group_calls.add(call)
        call.add_done_callback(self._on_run_group_call_done)
        # Shield `call` from the controller's end-of-training cancel, so `close` can still wait for it.
        return await asyncio.shield(call)

    async def cancel_group(self, group_id: int) -> None:
        """Stop a training group's `run_group` on its rollout worker; return once it ended there.

            cancel the caller of `run_group_rollouts`   the worker keeps running the group (Monarch
                                                        runs a call to the end)
            then `cancel_group(group_id)`               the worker cancels it; this returns once it ended
            the worker errors, or takes longer than     logged, not raised; the group may keep running
                `_RUN_GROUP_DRAIN_TIMEOUT_S`

        Every rollout worker gets the call; only the one running the group cancels it.
        """
        if self._worker_actors is None:
            raise RuntimeError("rollout worker pool is not initialized")
        try:
            async with asyncio.timeout(_RUN_GROUP_DRAIN_TIMEOUT_S):
                await self._worker_actors.cancel_group.call(group_id)
        except Exception:
            # Logged, not raised: the caller's rollout loop must keep serving groups.
            logger.exception(
                "cancel_group(%d) failed; the group may keep running", group_id
            )

    def _on_run_group_call_done(self, call: asyncio.Future[RolloutGroup]) -> None:
        self._inflight_run_group_calls.discard(call)
        # A cancelled caller never reads the error; read it so asyncio does not log it.
        if not call.cancelled():
            call.exception()

    def state_dict(self) -> dict[str, object]:
        """Return the checkpoint state of the training input iterator, curriculum and step."""
        return {
            "dataloader": self._training_dataloader.state_dict(),
            "curriculum": self._curriculum.state_dict(),
            "step": self._step,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore the training iterator (uncommitted inputs included), curriculum and step."""
        self._training_dataloader.load_state_dict(state_dict["dataloader"])
        self._curriculum.load_state_dict(state_dict["curriculum"])
        self._step = state_dict["step"]


def log_failed_rollout(rollout: Rollout) -> None:
    """Log a rollout whose status is an error, in one line: its status, its last error with the
    end of its traceback, and its other logs. A rollout that didn't fail logs nothing.

    Example:
        rollout group=3/rollout=1 failed: ERROR_TIMEOUT | TimeoutError: MessageEnv.step timed out
        after 1800.0s
    """
    if not rollout.status.is_error():
        return
    parts = [
        f"rollout group={rollout.group_id}/rollout={rollout.rollout_id} failed: "
        f"{rollout.status.name}"
    ]
    errors = rollout.logs.get("errors", [])
    if errors:
        error = errors[-1]
        parts.append(f"{error['type']}: {error['message']}")
        parts.extend((error.get("traceback") or "").strip().splitlines()[-3:])
    other_logs = {key: value for key, value in rollout.logs.items() if key != "errors"}
    if other_logs:
        parts.append(" ".join(f"{key}={value}" for key, value in other_logs.items()))
    logger.warning(" | ".join(parts))


class BaseRolloutWorker(Configurable, ABC):
    """Runs one rollout group in a rollout worker process, then scores it and computes its
    advantages. Subclasses implement `run_group`."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rubric: Rubric.Config
        """Reward functions + weights used by `score_group`."""

        advantage: Configurable.Config = field(
            default_factory=AdvantageEstimator.Config
        )
        """Post-scoring advantage estimator. Default = Dr.GRPO (mean-baseline only);
        set `AdvantageEstimator.Config(should_std_normalize=True)` for standard GRPO."""

    def __init__(self, config: Config) -> None:
        self.rubric: Rubric = config.rubric.build()
        self.advantage_estimator: AdvantageEstimator = config.advantage.build()

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Called once in the worker process after it is spawned."""

    async def score_group(
        self,
        rollouts: list[Rollout],
        env_input: object,
    ) -> list[RubricOutput]:
        """Score one group's rollouts; `make_scored_group` applies the rewards.
        Default impl delegates to `self.rubric.score_group`.

        Args:
            rollouts: Sibling rollouts in one prompt group, already stepped. Only `is_trainable()`
                rollouts are passed (see `make_scored_group`).
            env_input: the env initial input shared by the group.

        Returns:
            One `RubricOutput` per rollout, in input order.
        """
        return await self.rubric.score_group(rollouts, env_input)

    @abstractmethod
    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Roll out and score one prompt group: run its rollouts, then pass them to
        `make_scored_group`. Each rollout returns a `Rollout` even when it fails (ERROR_INFRA on
        an exception), and logs it with `log_failed_rollout`, so one failure doesn't fail the group.

        Args:
            generate_fn: Async callable that returns a Completion given a prompt.
            release_session_fn: Async callable each rollout calls after its last turn, to
                release resources associated to that session. E.g. generator's kv-cache.
            sample: Dataset sample shared by the group.
            group_id: Stable group id; siblings share it for advantage centering.
            group_size: Number of sibling rollouts.
            sampling: Sampling config for every generate call in the group.

        Returns:
            One scored `RolloutGroup`.
        """

    async def make_scored_group(
        self,
        *,
        rollouts: list[Rollout],
        sample: object,
        group_id: int,
    ) -> RolloutGroup:
        """Score one group's trainable rollouts and fill their advantages.

        Example, group 7, rewards from `score_group`:

            rollouts    [Rollout(0, COMPLETED), Rollout(1, ERROR_INFRA), Rollout(2, ERROR)]
            rewards     [1.0,                   None,                    0.0]
            advantages  [0.5,                   None,                    -0.5]

        A rollout that isn't `is_trainable()` gets no reward or advantage, and isn't trained.

        Args:
            rollouts: one per sibling, in rollout-id order.
            sample: the dataset sample the group shares; passed to `score_group`.
            group_id: the group's id.
        """
        # ======== Score the trainable rollouts ========
        # "Trainable" is about the status, not about training: validation groups are scored here too.
        # A group with no trainable rollout has nothing to score or center.
        rollouts_to_score = [
            rollout for rollout in rollouts if rollout.status.is_trainable()
        ]
        if not rollouts_to_score:
            return RolloutGroup(group_id=group_id, rollouts=rollouts)
        # TODO: a reward fn that raises (e.g. a judge that is down) fails the whole group. Catch it
        #   per rollout instead, so its siblings still train.
        rubric_outputs = await self.score_group(rollouts_to_score, sample)
        for rollout, rubric_output in zip(
            rollouts_to_score, rubric_outputs, strict=True
        ):
            rollout.reward = rubric_output.reward
            rollout.reward_breakdown = rubric_output.reward_breakdown

        # ======== Compute advantages ========
        # TODO: run an ordered list of group postprocessors here, e.g. a length penalty, or drop a
        # group whose rewards all match, then compute advantages. The advantage estimator becomes
        # one of them, and so can the group filters in `TrainingSampleBuilder.build_from_group`.
        # Then a user adds a group filter without editing this method or the builder. Not done yet:
        # it moves the builder's filters, which every recipe runs.
        advantages = self.advantage_estimator(
            RolloutGroup(group_id=group_id, rollouts=rollouts_to_score)
        )
        for rollout, advantage in zip(rollouts_to_score, advantages, strict=True):
            rollout.advantage = advantage
        return RolloutGroup(group_id=group_id, rollouts=rollouts)


class MessageEnvRolloutWorker(BaseRolloutWorker):
    """Builds each rollout's env from `message_env` and `token_env`, and drives it against the
    generator."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseRolloutWorker.Config):
        message_env: MessageEnv.Config
        """The env to build per sample; `make_env_group` calls `build(env_input=sample)`."""

        token_env: TokenEnv.Config
        """`TokenEnv` wraps the `MessageEnv` in `make_env_group`."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._message_env_config = config.message_env
        self._token_env_config = config.token_env
        self._renderer: Renderer

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
        vision_processor_config: VisionProcessor.Config | None = None,
    ) -> None:
        """Build the renderer that each rollout's `TokenEnv` uses.

        ``vision_processor_config``: `Controller.Config.vision_processor`, for a renderer that
        reads image files; None for text prompts.
        """
        tokenizer = tokenizer_config.build(tokenizer_path=hf_assets_path)
        vision_processor = (
            vision_processor_config.build()
            if vision_processor_config is not None
            else None
        )
        self._renderer = renderer_config.build(
            tokenizer=tokenizer, vision_processor=vision_processor
        )

    def make_env_group(
        self,
        *,
        sample: object,
        group_size: int,
    ) -> list[TokenEnv]:
        """Construct `group_size` single-use envs from one dataset sample.

        Args:
            sample: The source sample used to build the environment.
            group_size: number of sibling envs for this prompt group.

        Returns:
            `TokenEnv` * `group_size` instances, each ready for one rollout.
        """
        return [
            self._token_env_config.build(
                message_env=self._message_env_config.build(env_input=sample),
                renderer=self._renderer,
            )
            for _ in range(group_size)
        ]

    async def run_group(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        group_id: int,
        group_size: int,
        sampling: SamplingConfig,
    ) -> RolloutGroup:
        """Build `group_size` envs from `sample`, run them concurrently, then score them and fill their advantages."""
        # One prompt becomes [env] * group_size.
        envs = self.make_env_group(
            sample=sample,
            group_size=group_size,
        )

        # TODO(perf): siblings in a group share the first-turn prompt; tokenize it once per group and
        # reuse across the group_size rollouts (truest spot is the worker's first-turn render).
        # Each rollout catches its own failures, so a raise here is a bug. The TaskGroup then
        # cancels the siblings before raising, so none is still generating once the group is freed.
        try:
            async with asyncio.TaskGroup() as task_group:
                tasks = [
                    task_group.create_task(
                        self._run_single_rollout(
                            generate_fn=generate_fn,
                            release_session_fn=release_session_fn,
                            env=env,
                            # Offset the base seed per sample so a group's n=1
                            # requests are diverse yet reproducible run-to-run.
                            sampling=(
                                sampling
                                if sampling.seed is None
                                else replace(sampling, seed=sampling.seed + sample_idx)
                            ),
                            group_id=group_id,
                            rollout_id=sample_idx,
                        )
                    )
                    for sample_idx, env in enumerate(envs)
                ]
        finally:
            # close the envs
            await asyncio.gather(*(env.close() for env in envs), return_exceptions=True)
        return await self.make_scored_group(
            rollouts=[task.result() for task in tasks], sample=sample, group_id=group_id
        )

    async def _run_single_rollout(
        self,
        *,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
        env: TokenEnv,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Produce a single rollout, alternating between env and generator calls,
        until the env is terminal (env `done`, truncation, errors). An exception ends it as
        `ERROR_INFRA`. A failed rollout is logged once, at the end.

        For custom logic, users can override this method.

        Args:
            generate_fn: Async callable that runs one generation; keeps the worker
                decoupled from the generator actor.
            release_session_fn: Async callable this rollout calls once after its last turn,
                with its `routing_session_id`.
            env: The env for this rollout; `run_group` closes it.
            sampling: Sampling config for every generate call.
            group_id: The GRPO group id.
            rollout_id: Sibling index within the group; combined with the turn index into the
                per-turn `RolloutTurnID`, and stored as `Rollout.rollout_id`.

        Returns:
            One unscored `Rollout`; `make_scored_group` fills its reward later.
        """
        # Per-sample sticky key: a sample's turns reuse one generator's prefix cache.
        routing_session_id = RolloutTurnID(
            group_id=group_id, rollout_id=rollout_id, turn_id=0
        ).to_string(include_turn=False)
        turns: list[RolloutTurn] = []
        try:
            env_step = await env.init()
            while not env_step.status.is_terminal():
                turn_rollout_id = RolloutTurnID(
                    group_id=group_id,
                    rollout_id=rollout_id,
                    turn_id=len(turns),
                )

                # generator call
                completion = await generate_fn(
                    prompt=env_step.next_prompt,
                    request_id=turn_rollout_id.to_string(),
                    group_id=group_id,
                    routing_session_id=routing_session_id,
                    sampling_config=sampling,
                )

                # env call
                next_env_step = await env.step(completion)

                # this turn from a token and message perspective
                turns.append(
                    RolloutTurn(
                        rollout_id=turn_rollout_id,
                        prompt_prefix_len=env_step.next_prompt_prefix_len,
                        prompt_delta_token_ids=env_step.next_prompt_delta_token_ids,
                        prompt_delta_images=env_step.next_prompt_delta_images,
                        prompt_delta_image_placeholders=(
                            env_step.next_prompt_delta_image_placeholders
                        ),
                        prompt_messages=env_step.next_prompt_messages or [],
                        completion_token_ids=np.asarray(
                            completion.token_ids, dtype=np.int32
                        ),
                        completion_logprobs=np.asarray(
                            completion.token_logprobs, dtype=np.float32
                        ),
                        completion_message=next_env_step.completion_message,
                        env_messages=next_env_step.env_messages,
                        env_rewards=next_env_step.env_rewards,
                        min_policy_version=completion.min_policy_version,
                        max_policy_version=completion.max_policy_version,
                        metrics=completion.metrics,
                    )
                )

                # holds the input for next generation call
                env_step = next_env_step

            status = env_step.status
            errors = env_step.errors
        except Exception as error:
            # An exception is a failure outside the policy, so the rollout ends ERROR_INFRA.
            # The policy's own failures arrive as statuses instead: e.g. `TokenEnv` returns
            # ERROR_PARSE, and an env replies to bad output (`MessageEnv.step`).
            # TODO: at shutdown, every rollout still generating lands here and logs a failure.
            # Raise a typed error on generator close, and skip the log for it.
            status = RolloutStatus.ERROR_INFRA
            errors = [error_entry(error)]

        try:
            await release_session_fn(routing_session_id=routing_session_id)
        except Exception:
            # Not fatal: `max_sessions` evicts an unreleased session.
            logger.exception("releasing routing session %s failed", routing_session_id)
        rollout = Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
            logs={"errors": errors} if errors else {},
        )
        log_failed_rollout(rollout)
        return rollout
