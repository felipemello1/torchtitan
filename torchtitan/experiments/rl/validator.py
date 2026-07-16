# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Held-out validation as a swappable component.

The controller decides WHEN a pass runs (cadence) and WHAT pauses around it (the loop
mode); both live in the controller's `ValidationConfig`. This `Validator` owns the pass
itself: sample held-out prompts, generate greedily through a controller-supplied
`generate_fn`, score, and return metrics. A future benchmark or separate-pool evaluator
replaces this class without touching controller scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace

from renderers import Renderer

from torchtitan.config import Configurable
from torchtitan.experiments.rl.actors.generator import SamplingConfig
from torchtitan.experiments.rl.controller_metrics import compute_rollout_metrics
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.observability import structured_logger as sl

logger = logging.getLogger(__name__)


class Validator(Configurable):
    """Held-out greedy validation: one pass scores `num_samples` prompts (n=1) and returns metrics.

    Example:

        validator = config.async_loop.validation.validator.build(
            rollouter=rollouter, renderer=renderer, rollout_recorder=recorder, sampling=sampling)
        metrics = await validator.evaluate(step=step, generate_fn=generate_fn)
        metrics_processor.log(step=step, metrics=metrics, is_validation=True)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """What one validation pass evaluates (cadence and loop mode live on `ValidationConfig`)."""

        num_samples: int = 20
        """Held-out prompts scored greedily (temp=0, n=1) per pass. 0 disables validation."""

    def __init__(
        self,
        config: Config,
        *,
        rollouter: Rollouter,
        renderer: Renderer,
        rollout_recorder: RolloutSampleRecorder,
        sampling: SamplingConfig,
    ) -> None:
        self.config = config
        self._rollouter = rollouter
        self._renderer = renderer
        self._rollout_recorder = rollout_recorder
        # Greedy, single-sample validation regardless of the training sampling config.
        self._greedy = replace(sampling, temperature=0.0, top_p=1.0)

    # TODO: support running the entire validation dataset, not a fixed num_samples.
    # TODO: investigate using pass@k for validation (group_size=1 / best-of-1 today).
    @sl.log_trace_span("validate")
    async def evaluate(self, *, step: int, generate_fn: GenerateFn) -> list[m.Metric]:
        """Sample held-out prompts, run each greedily (n=1) concurrently via `generate_fn`, score, return metrics.

        The pass must sample exactly one generator policy version; the controller's loop
        mode freezes the generator for the duration, and a pass that still spans more
        than one version raises. Records rollouts; does not log (the controller logs the
        returned metrics).

        Args:
            step: Training step this pass is attributed to (0 for the pre-training pass).
            generate_fn: Controller-supplied generation entrypoint (the shared router
                today; a separate pool in a future validator).

        Returns:
            Rollout metrics (prefix `validation`), `validation/policy_version`,
            `validation/group_failures`, and `timing/validate`.
        """
        if self.config.num_samples == 0:  # validation disabled (e.g. loss-guard CI)
            return []
        started_at = time.perf_counter()

        # TODO(naming): reserve "sample" for TrainingSample; rename the rollouter's raw-prompt "sample" -> "prompt"/"data_input".
        validation_samples = [
            self._rollouter.get_validation_sample()
            for _ in range(self.config.num_samples)
        ]
        group_results = await asyncio.gather(
            *(
                self._rollouter.run_group_rollouts(
                    generate_fn=generate_fn,
                    sample=validation_sample,
                    # Negative ids keep validation request_ids disjoint from training in the shared engine.
                    group_id=-(group_index + 1),
                    group_size=1,
                    sampling=self._greedy,
                    renderer=self._renderer,
                )
                for group_index, validation_sample in enumerate(validation_samples)
            ),
            return_exceptions=True,
        )

        # Keep the groups that succeeded; log + count the ones that raised.
        rollout_groups: list[RolloutGroup] = []
        num_failed_groups = 0
        for group_index, result in enumerate(group_results):
            if isinstance(result, BaseException):
                logger.error(
                    f"validation group {-(group_index + 1)} (step={step}) failed; dropping",
                    exc_info=(type(result), result, result.__traceback__),
                )
                num_failed_groups += 1
                continue
            rollout_groups.append(result)

        self._rollout_recorder.record(is_validation=True, rollout_groups=rollout_groups)

        metrics = compute_rollout_metrics(
            prefix="validation",
            rollouts=[
                rollout for group in rollout_groups for rollout in group.rollouts
            ],
        )
        metrics.extend(_single_policy_metrics(rollout_groups))
        metrics.append(
            m.Metric("validation/group_failures", m.Sum(float(num_failed_groups)))
        )
        metrics.append(
            m.Metric("timing/validate", m.NoReduce(time.perf_counter() - started_at))
        )
        return metrics


def _single_policy_metrics(rollout_groups: list[RolloutGroup]) -> list[m.Metric]:
    """The one policy version this pass sampled, as a metric; raises if the pass spanned versions.

    Every shipped loop mode promises the generator is frozen during a pass, so a mixed
    pass is a coordinator bug, not a valid eval point. Turns with no generation
    (`min/max_policy_version is None`) are skipped.

    Example:

        # one group, two turns both sampled at version 7
        _single_policy_metrics([group])
        # -> [Metric("validation/policy_version", NoReduce(7.0))]
    """
    min_versions = [
        rollout_turn.min_policy_version
        for group in rollout_groups
        for rollout in group.rollouts
        for rollout_turn in rollout.turns
        if rollout_turn.min_policy_version is not None
    ]
    max_versions = [
        rollout_turn.max_policy_version
        for group in rollout_groups
        for rollout in group.rollouts
        for rollout_turn in rollout.turns
        if rollout_turn.max_policy_version is not None
    ]
    if not min_versions:  # no generation happened (e.g. every group failed)
        return []

    min_version = min(min_versions)
    max_version = max(max_versions)
    if min_version != max_version:
        raise RuntimeError(
            "single-policy validation sampled generator versions "
            f"{min_version}..{max_version}"
        )
    return [m.Metric("validation/policy_version", m.NoReduce(float(min_version)))]
