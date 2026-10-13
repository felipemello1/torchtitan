# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Rollouter backed by a Verifiers environment service."""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import numpy as np
from verifiers.v1.configs.client import TrainClientConfig as VerifiersTrainClientConfig
from verifiers.v1.configs.taskset import TasksetConfig as VerifiersTasksetConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.serve.client import EnvClient as VerifiersEnvClient
from verifiers.v1.types import SamplingConfig as VerifiersSamplingConfig

from torchtitan.components.renderer import RendererConfig, RenderersConfigAdapter

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.rl.components.data import IterableRLDataLoader
from torchtitan.rl.examples.verifiers.data import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.rl.examples.verifiers.env_server import VerifiersEnvServer
from torchtitan.rl.examples.verifiers.generation_server import (
    GenerationServer,
    GROUP_ID_SAMPLING_PARAM,
    MAX_ROLLOUT_TOKENS_SAMPLING_PARAM,
    VerifiersGenerationMetadata,
)
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import RewardFn, Rubric
from torchtitan.rl.rollout.types import (
    GenerateFn,
    ReleaseSessionFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
    split_prompt,
)

if TYPE_CHECKING:
    from torchtitan.rl.generator import SamplingConfig


logger = logging.getLogger(__name__)

VERIFIERS_REWARD_KEY = "verifiers_reward"


class RewardFromVerifiers(RewardFn):
    """Pass through the reward that Verifiers computed during the rollout."""

    @dataclass(kw_only=True, slots=True)
    class Config(RewardFn.Config):
        pass

    async def __call__(self, rollout: Rollout, env_input: object) -> float:
        del env_input
        for turn in reversed(rollout.turns):
            if VERIFIERS_REWARD_KEY in turn.env_rewards:
                return float(turn.env_rewards[VERIFIERS_REWARD_KEY])
        return 0.0


@dataclass(kw_only=True, slots=True)
class VerifiersValidationSet:
    """An extra validation set with its own taskset, hence its own EnvServer (e.g. a benchmark
    whose environment, images or grader differ from the training taskset's)."""

    dataset: VerifiersTaskDataset.Config
    """Verifiers taskset used for this set's validation samples."""

    verifiers_env_server: VerifiersEnvServer.Config
    """EnvServer for this set; its taskset is derived from `dataset`."""

    max_rollout_tokens: int | None = None
    """Context cap (prompt + output tokens) of this set's rollouts; None: the generation server's
    `max_rollout_tokens`. A turn whose prompt reaches it stops the rollout as truncated."""

    def __post_init__(self) -> None:
        if self.max_rollout_tokens is not None and self.max_rollout_tokens <= 0:
            raise ValueError("max_rollout_tokens must be positive")
        configured_taskset = self.verifiers_env_server.environment.taskset
        if configured_taskset not in (
            VerifiersTasksetConfig(),
            self.dataset.verifiers_taskset,
        ):
            raise ValueError(
                "an extra validation set's verifiers_env_server.environment.taskset is derived "
                "from its dataset.verifiers_taskset and must not configure a different taskset"
            )
        self.verifiers_env_server = replace(
            self.verifiers_env_server,
            environment=self.verifiers_env_server.environment.model_copy(
                update={"taskset": self.dataset.verifiers_taskset}
            ),
            local_taskset_module=_local_taskset_module(self.dataset.verifiers_taskset),
        )


class VerifiersRollouter(Rollouter):
    """Adapt Verifiers execution to TitanRL's ``Rollouter`` contract.

    Verifiers owns environment execution, including its process pool and the
    message/tool loop, so this path bypasses TitanRL's ``RolloutWorkerActor``,
    ``RolloutWorker``, ``MessageEnv``, and ``TokenEnv``. Token-in/token-out is
    preserved through ``GenerationServer``, which forwards Verifiers model
    requests to TitanRL's controller-provided ``GenerateFn``.

    TitanRL still owns dataset and group scheduling, generator routing, policy
    metadata, conversion to ``RolloutTurn``, reward and advantage assignment,
    and the downstream training path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        training_dataloader: IterableRLDataLoader.Config
        """Verifiers taskset used for training samples and environment scoring."""

        validation_dataset: VerifiersTaskDataset.Config
        """Verifiers taskset used for validation samples."""

        max_rollout_tokens: int | None = None
        """Context cap (prompt + output tokens) of training rollouts; None: the generation
        server's `max_rollout_tokens`. A turn whose prompt reaches it stops the rollout as
        truncated (graded, with the rubric's truncation penalty) instead of producing a sample the
        batcher would drop as oversized."""

        validation_max_rollout_tokens: int | None = None
        """Context cap of the main validation set's rollouts; None: `max_rollout_tokens`."""

        extra_validation_sets: dict[str, VerifiersValidationSet] = field(
            default_factory=dict
        )
        """Named validation sets on their own EnvServers, run by `AsyncLoopConfig.extra_validations`
        entries of the same name. Unlike `validation_dataset`, their taskset may differ from the
        training one."""

        worker: RolloutWorker.Config | None = None
        """Inherited schema field; hidden because Verifiers replaces that path."""

        worker_pool_size: int = 1
        """Unused because the Verifiers EnvServer owns its worker pool."""

        num_threads_per_worker: int = 1
        """Unused because no TitanRL rollout-worker process is spawned."""

        verifiers_env_server: VerifiersEnvServer.Config
        """Environment server spawned and owned by this rollouter."""

        rubric: Rubric.Config
        """TorchTitan rubric that consumes rewards returned by Verifiers."""

        advantage: AdvantageEstimator.Config = field(
            default_factory=AdvantageEstimator.Config
        )
        """TitanRL estimator applied after Verifiers returns rollout rewards."""

        generation_server: GenerationServer.Config
        """Local HTTP bridge from Verifiers to TitanRL generation."""

        renderer_multiplex: int = 256
        """Maximum concurrent rollouts sharing one Verifiers renderer instance.

        TODO: evaluate sharing this renderer pool with TitanRL's native rollout
        path instead of keeping it specific to the Verifiers client.
        """

        connection_timeout_sec: float = 120.0
        """Maximum time to wait for the Verifiers server to become healthy."""

        def __post_init__(self) -> None:
            Rollouter.Config.__post_init__(self)
            configured_taskset = self.verifiers_env_server.environment.taskset
            if configured_taskset not in (
                VerifiersTasksetConfig(),
                self.training_dataloader.dataset.verifiers_taskset,
            ):
                raise ValueError(
                    "verifiers_env_server.environment.taskset is derived from "
                    "training_dataloader.dataset.verifiers_taskset and must not configure a "
                    "different taskset"
                )
            self.verifiers_env_server = replace(
                self.verifiers_env_server,
                environment=self.verifiers_env_server.environment.model_copy(
                    update={
                        "taskset": self.training_dataloader.dataset.verifiers_taskset
                    }
                ),
                local_taskset_module=_local_taskset_module(
                    self.training_dataloader.dataset.verifiers_taskset
                ),
            )
            for name, cap in (
                ("max_rollout_tokens", self.max_rollout_tokens),
                ("validation_max_rollout_tokens", self.validation_max_rollout_tokens),
            ):
                if cap is not None and cap <= 0:
                    raise ValueError(f"{name} must be positive")
            if self.renderer_multiplex <= 0:
                raise ValueError("renderer_multiplex must be positive")
            if self.connection_timeout_sec <= 0:
                raise ValueError("connection_timeout_sec must be positive")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._verifiers_config = config
        self._rubric: Rubric = config.rubric.build()
        self._advantage_estimator: AdvantageEstimator = config.advantage.build()
        self._verifiers_env_server = config.verifiers_env_server.build()
        self._extra_env_servers = {
            name: validation_set.verifiers_env_server.build()
            for name, validation_set in config.extra_validation_sets.items()
        }
        self._extra_env_clients: dict[str, VerifiersEnvClient] = {}
        self._generation_server: GenerationServer | None = None
        self._verifiers_env_client: VerifiersEnvClient | None = None
        self._verifiers_train_client_config: VerifiersTrainClientConfig | None = None

    async def setup_async(
        self,
        *,
        tokenizer_config: HuggingFaceTokenizer.Config,
        renderer_config: RendererConfig,
        hf_assets_path: str,
    ) -> None:
        """Start the EnvServer and connect it to TorchTitan generation."""
        del tokenizer_config
        if self._verifiers_env_client is not None:
            return
        if not isinstance(renderer_config, RenderersConfigAdapter):
            raise ValueError(
                "Verifiers requires a renderer configured with from_renderers(...) "
                "so its client can construct the same renderer in the "
                "environment-server process"
            )

        # Verifiers generates through an HTTP endpoint, while TorchTitan exposes
        # an in-process GenerateFn. This server bridges those interfaces.
        # TODO: remove this bridge when the controller can provide a generator
        # HTTP endpoint directly.
        generation_server = self._verifiers_config.generation_server.build()
        verifiers_server_address = await self._verifiers_env_server.start()
        verifiers_env_client = None
        try:
            await generation_server.start()
            verifiers_env_client = VerifiersEnvClient(verifiers_server_address)
            await verifiers_env_client.wait_for_server_startup(
                timeout=self._verifiers_config.connection_timeout_sec
            )
            verifiers_train_client_config = VerifiersTrainClientConfig(
                base_url=generation_server.base_url,
                # The local endpoint does not authenticate. An empty environment
                # variable name makes Verifiers send its required "EMPTY" value.
                api_key_var="",
                renderer=renderer_config.renderers_config,
                multiplex=self._verifiers_config.renderer_multiplex,
                renderer_model_name=hf_assets_path,
            )
        except BaseException:
            if verifiers_env_client is not None:
                await verifiers_env_client.close()
            try:
                await generation_server.close()
            finally:
                await self._verifiers_env_server.close()
            raise
        self._generation_server = generation_server
        self._verifiers_env_client = verifiers_env_client
        self._verifiers_train_client_config = verifiers_train_client_config
        # Extra validation sets: one EnvServer each, sharing the generation server.
        for name, env_server in self._extra_env_servers.items():
            address = await env_server.start()
            client = VerifiersEnvClient(address)
            self._extra_env_clients[name] = client
            await client.wait_for_server_startup(
                timeout=self._verifiers_config.connection_timeout_sec
            )

    async def close(self) -> None:
        """Close the Verifiers clients, generation server, and environment servers."""
        extra_clients, self._extra_env_clients = self._extra_env_clients, {}
        for client in extra_clients.values():
            await client.close()
        for env_server in self._extra_env_servers.values():
            await env_server.close()
        try:
            if self._verifiers_env_client is not None:
                await self._verifiers_env_client.close()
        finally:
            self._verifiers_env_client = None
            self._verifiers_train_client_config = None
            try:
                if self._generation_server is not None:
                    await self._generation_server.close()
            finally:
                self._generation_server = None
                await self._verifiers_env_server.close()

    def get_validation_samples(
        self, steps: int, validation_set: str | None = None
    ) -> list[object]:
        """Materialize one validation pass; an extra set's tasks are tagged with its name so they
        run on its EnvServer."""
        if validation_set is None:
            return Rollouter.get_validation_samples(self, steps)
        try:
            dataset = self._verifiers_config.extra_validation_sets[validation_set].dataset
        except KeyError:
            raise ValueError(
                f"unknown validation set {validation_set!r}; extra_validation_sets has "
                f"{sorted(self._verifiers_config.extra_validation_sets)}"
            ) from None
        iterator = iter(dataset.build())
        samples = (
            list(iterator) if steps == -1 else list(itertools.islice(iterator, steps))
        )
        return [replace(sample, validation_set=validation_set) for sample in samples]

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
        """Run sibling rollouts through Verifiers, then compute advantages."""
        if self._generation_server is None:
            raise RuntimeError("Verifiers rollouter is not initialized")
        self._generation_server.set_generate_fn(generate_fn)
        results = await asyncio.gather(
            *(
                self._run_single_rollout(
                    release_session_fn=release_session_fn,
                    sample=sample,
                    sampling=(
                        sampling
                        if sampling.seed is None
                        else replace(sampling, seed=sampling.seed + rollout_id)
                    ),
                    group_id=group_id,
                    rollout_id=rollout_id,
                )
                for rollout_id in range(group_size)
            ),
            return_exceptions=True,
        )
        # Raise only once every sibling finished: the caller releases the group right after.
        for result in results:
            if isinstance(result, BaseException):
                raise result
        rollouts = results

        outputs = await self._rubric.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        advantages = self._advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        return group

    def _max_rollout_tokens_param(
        self, sample: VerifiersTaskSample, group_id: int
    ) -> dict[str, int]:
        """The context cap to send with this rollout's generation requests, if any.

        Example: training cap 65,536, an extra set "deepswe" at 131,072:
            training sample (group 7)             -> {MAX_ROLLOUT_TOKENS_SAMPLING_PARAM: 65536}
            main validation sample (group -3)     -> validation_max_rollout_tokens or 65536
            "deepswe" sample                      -> {MAX_ROLLOUT_TOKENS_SAMPLING_PARAM: 131072}
        """
        config = self._verifiers_config
        if sample.validation_set is not None:
            cap = config.extra_validation_sets[sample.validation_set].max_rollout_tokens
        elif group_id < 0 and config.validation_max_rollout_tokens is not None:
            cap = config.validation_max_rollout_tokens
        else:
            cap = config.max_rollout_tokens
        return {} if cap is None else {MAX_ROLLOUT_TOKENS_SAMPLING_PARAM: cap}

    async def _run_single_rollout(
        self,
        *,
        release_session_fn: ReleaseSessionFn,
        sample: object,
        sampling: SamplingConfig,
        group_id: int,
        rollout_id: int,
    ) -> Rollout:
        """Send one task to Verifiers, release its routing sessions, and convert its trace to a
        rollout."""
        if not isinstance(sample, VerifiersTaskSample):
            raise TypeError("Verifiers requires a VerifiersTaskSample")
        if (
            self._generation_server is None
            or self._verifiers_env_client is None
            or self._verifiers_train_client_config is None
        ):
            raise RuntimeError("Verifiers rollouter is not initialized")

        # One VerifiersEnvClient.run executes a complete rollout. The harness
        # owns the multi-turn model/tool loop and calls the generation server once per
        # generation; run returns only after the rollout stops.
        env_client = (
            self._verifiers_env_client
            if sample.validation_set is None
            else self._extra_env_clients[sample.validation_set]
        )
        verifiers_episode = await env_client.run(
            task_data=sample.verifiers_task_data,
            client=self._verifiers_train_client_config,
            model=self._generation_server.model_id,
            sampling=VerifiersSamplingConfig(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                max_tokens=sampling.max_tokens,
                seed=sampling.seed,
                num_topk_logprobs=sampling.num_topk_logprobs,
                **{GROUP_ID_SAMPLING_PARAM: group_id},
                **self._max_rollout_tokens_param(sample, group_id),
            ),
        )
        # The rollout made its last call. Verifiers sent each trace's id as its X-Session-ID.
        # TODO: sessions of attempts Verifiers retried (not in `traces`) wait for the group's
        # release; the generation server could track each rollout's sessions.
        for trace in verifiers_episode.traces:
            try:
                await release_session_fn(routing_session_id=trace.id)
            except Exception:
                # Not fatal: `max_sessions` evicts an unreleased session.
                logger.exception("releasing routing session %s failed", trace.id)
        traces = [trace for trace in verifiers_episode.traces if trace.agent.trainable]
        if len(traces) != 1:
            raise ValueError(
                "Verifiers expects one trainable trace per rollout; got "
                f"{len(traces)}"
            )

        # Convert Verifiers' graph trace to TitanRL's linear rollout turns.
        trace = traces[0]
        generation_metadata = self._generation_server.pop_generation_metadata(trace.id)
        turns = self.trace_to_rollout_turns(
            trace=trace,
            generation_metadata=generation_metadata,
            group_id=group_id,
            rollout_id=rollout_id,
        )
        status = self.rollout_status(
            verifiers_episode=verifiers_episode,
            trace=trace,
            is_training=sample.validation_set is None and group_id >= 0,
        )
        if not turns:
            # Verifiers <= 0.3.1 stops an overlong prompt as context_length, later ones as
            # max_input_tokens.
            if trace.stop_condition in ("context_length", "max_input_tokens"):
                # The harness builds the first prompt in the env server, so the dataset can't
                # be filtered for it; flag it here as a data problem, not a policy one.
                task = sample.verifiers_task_data.get(
                    "name", sample.verifiers_task_data.get("idx")
                )
                logger.warning(
                    "rollout group=%d/rollout=%d: the first prompt%s does not fit the "
                    "context; marking ERROR",
                    group_id,
                    rollout_id,
                    "" if task is None else f" of task {task}",
                )
            status = RolloutStatus.ERROR
        logs = verifiers_rollout_logs(verifiers_episode, trace)
        if status.is_error():
            log_failed_rollout(logs, group_id=group_id, rollout_id=rollout_id)
        if status != RolloutStatus.ERROR:
            turns[-1].env_rewards[VERIFIERS_REWARD_KEY] = trace.reward
        return Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=status,
            turns=turns,
            logs=logs,
        )

    @staticmethod
    def rollout_status(
        *, verifiers_episode: Any, trace: Any, is_training: bool = True
    ) -> RolloutStatus:
        """Map how Verifiers ended a rollout to its status.

        A stage deadline (setup, agent, finalize, scoring) flags `trace.is_timeout` but can leave
        `ok` True, so a timed-out rollout would still be scored. Training marks it ERROR_TIMEOUT,
        which stays out of the loss (usually slow infra or a stalled agent, not the policy);
        validation keeps it as a scored outcome.

        Example (ok, is_timeout, is_training -> status):
            True,  True,  True  -> ERROR_TIMEOUT
            True,  True,  False -> COMPLETED or TRUNCATED_*
            False, False, any   -> ERROR
        """
        if is_training and trace.is_timeout:
            return RolloutStatus.ERROR_TIMEOUT
        if not verifiers_episode.ok or not trace.ok:
            return RolloutStatus.ERROR
        if not trace.is_truncated:
            return RolloutStatus.COMPLETED
        if trace.stop_condition == "max_turns":
            return RolloutStatus.TRUNCATED_MAX_TURNS
        return RolloutStatus.TRUNCATED_LENGTH

    @staticmethod
    def trace_to_rollout_turns(
        *,
        trace: Any,
        generation_metadata: VerifiersGenerationMetadata | None,
        group_id: int,
        rollout_id: int,
    ) -> list[RolloutTurn]:
        """Flatten a Verifiers trace into TorchTitan's trainable rollout turns.

        For example, a Verifiers branch containing user tokens ``[1, 2]``,
        sampled assistant tokens ``[3, 4]``, tool-result tokens ``[5, 6]``, and
        sampled assistant tokens ``[7, 8]`` becomes two TitanRL turns. Their
        prompts are ``[1, 2]`` and ``[1, 2, 3, 4, 5, 6]``; their completions are
        ``[3, 4]`` and ``[7, 8]``. Shared sampled graph nodes are emitted once.
        Non-sampled nodes after a sampled one, e.g. the tool result ``[5, 6]``,
        become that turn's ``env_messages``, also once per node.

        Verifiers does not return TorchTitan policy metadata, so every emitted
        turn receives the conservative min/max policy-version span accumulated
        by the generation server for the whole rollout. Generator metrics are
        attached once to avoid double counting. A turn's top-k rows come from the
        generation whose prompt length and completion tokens match its node.
        """
        if generation_metadata is None:
            if any(any(node.mask) for node in trace.nodes):
                raise ValueError(
                    "Verifiers trace has trainable tokens but no generation metadata"
                )
            return []

        node_index = {id(node): index for index, node in enumerate(trace.nodes)}
        trained_nodes: set[int] = set()
        replied_nodes: set[int] = set()
        last_turn_by_node: dict[int, RolloutTurn] = {}
        turns: list[RolloutTurn] = []
        # The last appended turn's prompt + completion; not reset between branches
        previous_token_ids: list[int] = []

        for branch in trace.branches:
            token_ids = branch.token_ids
            logprobs = branch.logprobs
            # [len(token_ids), num_layers, top_k] from the generation server; None without replay.
            # TODO: Verifiers gives a continued turn's last token the row of the token before it,
            # so 1 input per turn boundary replays its neighbor's experts (native: next prefill's row).
            routed_expert_ids = branch.routed_experts
            branch_offset = 0
            reply_to: RolloutTurn | None = None
            for node in branch.nodes:
                index = node_index[id(node)]
                mask = list(node.mask)
                if node.sampled and any(mask):
                    if index in trained_nodes:
                        mask = [False] * len(mask)
                    else:
                        trained_nodes.add(index)
                topk = None
                if generation_metadata.topk_by_generation and any(mask):
                    # The prompt this node's completion was sampled from ends at its first
                    # sampled token.
                    key = (
                        branch_offset + node.mask.index(True),
                        tuple(
                            token_id
                            for token_id, sampled in zip(node.token_ids, node.mask)
                            if sampled
                        ),
                    )
                    if key not in generation_metadata.topk_by_generation:
                        raise ValueError(
                            "Verifiers node has no generation with the same prompt length "
                            "and completion tokens, so its top-k logprobs are unknown"
                        )
                    topk = generation_metadata.topk_by_generation[key]
                for start, end in _trainable_token_spans(mask):
                    absolute_start = branch_offset + start
                    absolute_end = branch_offset + end
                    prompt_prefix_len, prompt_delta_token_ids = split_prompt(
                        token_ids[:absolute_start], previous_token_ids
                    )
                    previous_token_ids = token_ids[:absolute_end]
                    # Completion rows of this span: sampled tokens before `start` in the node.
                    row = sum(node.mask[:start])
                    turns.append(
                        RolloutTurn(
                            rollout_id=RolloutTurnID(
                                group_id=group_id,
                                rollout_id=rollout_id,
                                turn_id=len(turns),
                            ),
                            prompt_prefix_len=prompt_prefix_len,
                            prompt_delta_token_ids=prompt_delta_token_ids,
                            completion_token_ids=np.asarray(
                                token_ids[absolute_start:absolute_end], dtype=np.int32
                            ),
                            completion_logprobs=np.asarray(
                                logprobs[absolute_start:absolute_end], dtype=np.float32
                            ),
                            routed_expert_ids=(
                                None
                                if routed_expert_ids is None
                                else routed_expert_ids[
                                    max(prompt_prefix_len - 1, 0) : absolute_end - 1
                                ]
                            ),
                            completion_topk_token_ids=(
                                None
                                if topk is None
                                else topk[0][row : row + end - start]
                            ),
                            completion_topk_logprobs=(
                                None
                                if topk is None
                                else topk[1][row : row + end - start]
                            ),
                            min_policy_version=generation_metadata.min_policy_version,
                            max_policy_version=generation_metadata.max_policy_version,
                            completion_message=message_to_wire(node.message),
                            metrics=(
                                list(generation_metadata.metrics) if not turns else []
                            ),
                        )
                    )
                    last_turn_by_node[index] = turns[-1]
                if node.sampled:
                    reply_to = last_turn_by_node.get(index)
                elif reply_to is not None and index not in replied_nodes:
                    replied_nodes.add(index)
                    reply_to.env_messages.append(message_to_wire(node.message))
                branch_offset += len(node.token_ids)
        return turns


def _trainable_token_spans(mask: list[bool]) -> list[tuple[int, int]]:
    """Return half-open token spans marked trainable by a Verifiers node mask."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, sampled in enumerate([*mask, False]):
        if sampled and start is None:
            start = index
        elif not sampled and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _local_taskset_module(taskset: VerifiersTasksetConfig) -> str | None:
    """Return the module backing a locally registered taskset alias."""
    module = type(taskset).__module__
    alias = module.replace(".", "_").lower()
    return module if taskset.id == alias else None


def verifiers_rollout_logs(verifiers_episode: Any, trace: Any) -> dict[str, Any]:
    """Summarize how a Verifiers rollout ended, for ``Rollout.logs``.

    Verifiers' own "rollout done" line prints only the error class, so an agent
    timeout, a crashed tool call and an unreachable model endpoint all read as
    "HarnessError". This keeps the error messages and tracebacks (a failed
    harness's message ends with its stderr), per-phase wall time and model-call
    latency. For example, ``agent_sec=7200 model_sec=6900
    slowest_model_call_sec=1801`` on an agent timeout points at slow generation,
    not the task.
    """
    timing = trace.timing
    call_seconds = [call.time.duration for call in trace.calls]
    return {
        "verifiers_trace_id": trace.id,
        "task": trace.task.key,
        "stop_condition": trace.stop_condition,
        "errors": [
            error.model_dump(mode="json")
            for error in (*verifiers_episode.errors, *trace.errors)
        ],
        "setup_sec": timing.setup.duration,
        "agent_sec": timing.agent.duration,
        "model_sec": timing.agent.model.duration,
        "harness_sec": timing.agent.harness.duration,
        "scoring_sec": timing.scoring.duration,
        "model_calls": len(call_seconds),
        "failed_model_calls": sum(call.error is not None for call in trace.calls),
        "slowest_model_call_sec": max(call_seconds, default=0.0),
    }


def log_failed_rollout(logs: dict[str, Any], *, group_id: int, rollout_id: int) -> None:
    """Log the last error and timing from ``verifiers_rollout_logs`` in one line."""
    error = logs["errors"][-1] if logs["errors"] else {}
    traceback_tail = " | ".join(
        (error.get("traceback") or "").strip().splitlines()[-3:]
    )
    logger.warning(
        "Verifiers rollout failed: trace=%s group=%d rollout=%d task=%s stop=%s "
        "error=%s: %s | setup=%.0fs agent=%.0fs (model=%.0fs harness=%.0fs) "
        "scoring=%.0fs | model_calls=%d failed_calls=%d slowest_call=%.0fs | %s",
        logs["verifiers_trace_id"],
        group_id,
        rollout_id,
        logs["task"],
        logs["stop_condition"],
        error.get("type"),
        error.get("message"),
        logs["setup_sec"],
        logs["agent_sec"],
        logs["model_sec"],
        logs["harness_sec"],
        logs["scoring_sec"],
        logs["model_calls"],
        logs["failed_model_calls"],
        logs["slowest_model_call_sec"],
        traceback_tail,
    )
