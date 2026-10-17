# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metric helpers for the async RL loop to keep the loop functions free of metric computations."""

# TODO(async-rl): revisit this module's path/name — not observability/metrics (that is public API),
# but components/ may not be the right home either.

import contextlib
import gc
import itertools
import json
import logging
import os
import random
import time
import types
from collections import Counter, defaultdict

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Rollout

logger = logging.getLogger(__name__)


class MetricsTimer:
    """Times named code spans; flush() drains them into Mean metrics, then resets.

    Example:
        metric_timer = MetricsTimer()
        with metric_timer.record("timing/step/total"):
            for _ in range(num_microbatches):
                with metric_timer.record("timing/step/forward_backward"):
                    ...
        time_metrics = metric_timer.flush()
        # -> [Metric("timing/step/total", Mean(...)),
        #     Metric("timing/step/forward_backward", Mean(...))]   # mean over num_microbatches
    """

    def __init__(self) -> None:
        self.durations: dict[str, list[float]] = defaultdict(list)

    @contextlib.contextmanager
    def record(self, key: str):
        # TODO(async-rl): consider asynccontextmanager if span entry/exit ever needs async work; this
        # sync manager still measures awaited blocks correctly.
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[key].append(time.perf_counter() - start)

    def flush(self) -> list[m.Metric]:
        """Return one Mean metric per recorded span, then reset so the timer can be reused."""
        durations = self.durations
        self.durations = defaultdict(list)
        return [
            m.Metric(key, m.Mean.from_list(values)) for key, values in durations.items()
        ]


class GCTimer:
    """Time this process spends in Python's garbage collector, per generation; flush() drains it per step.

    Example:
        gc_timer = GCTimer()  # registers a gc callback until close()
        ...                   # collections run during the step
        gc_timer.flush()
        # -> [Metric("perf/controller/gc_seconds", Sum(1.9)),    # every collection this step
        #     Metric("perf/controller/gc_seconds", Max(1.2)),    # the longest pause
        #     ...,                                               # gen0, gen1: the same 4 keys as below
        #     Metric("controller/gc/gen2/collections", Sum(2)),  # full collections
        #     Metric("controller/gc/gen2/seconds", Sum(1.8)),    # time in them
        #     Metric("controller/gc/gen2/seconds", Max(1.2)),    # the longest one
        #     Metric("controller/gc/gen2/collected", Sum(310))]  # objects they freed
        gc_timer.close()
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._reset()
        gc.callbacks.append(self._on_gc)

    def _reset(self) -> None:
        # One entry per GC generation: 0, 1, 2.
        self._collections = [0, 0, 0]
        self._total_s = [0.0, 0.0, 0.0]
        self._max_s = [0.0, 0.0, 0.0]
        self._collected = [0, 0, 0]

    def _on_gc(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._start = time.perf_counter()
            return
        # Wall time: if another thread takes the GIL during these callbacks, its run time counts too.
        seconds = time.perf_counter() - self._start
        generation = info["generation"]
        self._collections[generation] += 1
        self._total_s[generation] += seconds
        self._max_s[generation] = max(self._max_s[generation], seconds)
        self._collected[generation] += info["collected"]
        if seconds >= 0.1:
            # Timestamped, to line pauses up with event-loop lag and request latency.
            sl.log_trace_scalar({f"gc_gen{generation}_pause_ms": seconds * 1000})

    def flush(self) -> list[m.Metric]:
        """Return the GC metrics since the last flush, then reset."""
        collections, total_s, max_s, collected = (
            self._collections,
            self._total_s,
            self._max_s,
            self._collected,
        )
        self._reset()
        metrics = [
            m.Metric("perf/controller/gc_seconds", m.Sum(sum(total_s))),
            m.Metric("perf/controller/gc_seconds", m.Max(max(max_s))),
        ]
        # Not under perf/: the console prints every perf/ key.
        for generation in range(3):
            prefix = f"controller/gc/gen{generation}"
            metrics += [
                m.Metric(f"{prefix}/collections", m.Sum(collections[generation])),
                m.Metric(f"{prefix}/seconds", m.Sum(total_s[generation])),
                m.Metric(f"{prefix}/seconds", m.Max(max_s[generation])),
                m.Metric(f"{prefix}/collected", m.Sum(collected[generation])),
            ]
        return metrics

    def close(self) -> None:
        gc.callbacks.remove(self._on_gc)


def _type_name(obj: object) -> str:
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def append_heap_census(
    *, path: str, step: int, num_referrer_types: int = 0, num_samples: int = 200
) -> None:
    """Append one JSON line to `path` counting, by type, the objects a full GC scans: containers
    and class instances, not str, bytes or numbers. The top types show what makes a full GC slow.

    With `num_referrer_types` > 0, it also samples `num_samples` objects of each of the top
    types and counts their referrers' types, two levels up, in two passes over the heap
    (~1 s per pass per 1M objects).

    Example:
        append_heap_census(path="outputs/rl/heap_census.jsonl", step=20, num_referrer_types=3)
        # appends:
        # {"step": 20, "rss_gib": 31.2,
        #  "objects_per_generation": [1520, 3100, 1371000],  # a full collection scans all 3
        #  "frozen_objects": 1021000,                        # after gc.freeze(): never scanned
        #  "top_types": [["builtins.dict", 412000], ["builtins.list", 160000], ...],  # top 40
        #  "referrers": {"builtins.dict": [["builtins.list", 180], ...], ...},
        #  "referrers_2": {"builtins.dict": [["mod.Rollout", 150], ...], ...},
        #  "census_seconds": 0.25}
    """
    start = time.perf_counter()
    gc_was_enabled = gc.isenabled()
    gc.disable()  # a collection between the calls would list gen-0 survivors twice
    generations = [gc.get_objects(generation=g) for g in range(3)]
    type_counts = Counter(map(type, itertools.chain.from_iterable(generations)))
    record: dict = {
        "step": step,
        "objects_per_generation": [len(objects) for objects in generations],
        "frozen_objects": gc.get_freeze_count(),
        "top_types": [
            [f"{obj_type.__module__}.{obj_type.__qualname__}", count]
            for obj_type, count in type_counts.most_common(40)
        ],
    }
    # A full GC visits every reference of every tracked object, untracked ints and strs too,
    # so its cost follows references, not objects: a list of 7k ints is 1 object, 7k visits.
    referents_by_type: Counter = Counter()
    referents_per_generation = []
    for objects in generations:
        total = 0
        for obj in objects:
            num_referents = len(gc.get_referents(obj))
            referents_by_type[type(obj)] += num_referents
            total += num_referents
        referents_per_generation.append(total)
    # A gen1 collect walks generations 0 and 1 only; a full collect walks all three.
    record["referents_per_generation"] = referents_per_generation
    record["total_referents"] = sum(referents_by_type.values())
    record["top_types_by_referents"] = [
        [f"{obj_type.__module__}.{obj_type.__qualname__}", count]
        for obj_type, count in referents_by_type.most_common(30)
    ]
    if num_referrer_types > 0:
        everything = list(itertools.chain.from_iterable(generations))
        rng = random.Random(step)
        # id -> sampled type name; level 1 = sampled objects, level 2 = their referrers.
        sampled: dict[int, str] = {}
        for obj_type, _ in type_counts.most_common(num_referrer_types):
            candidates = [obj for obj in everything if type(obj) is obj_type]
            for obj in rng.sample(candidates, min(num_samples, len(candidates))):
                sampled[id(obj)] = _type_name(obj)
            del candidates
        level_2: dict[int, str] = {}
        referrers: dict[str, Counter] = defaultdict(Counter)
        for obj in everything:
            for referent in gc.get_referents(obj):
                sampled_type = sampled.get(id(referent))
                if sampled_type is not None:
                    referrers[sampled_type][_type_name(obj)] += 1
                    level_2.setdefault(id(obj), sampled_type)
        referrers_2: dict[str, Counter] = defaultdict(Counter)
        for obj in everything:
            for referent in gc.get_referents(obj):
                sampled_type = level_2.get(id(referent))
                if sampled_type is not None:
                    referrers_2[sampled_type][_type_name(obj)] += 1
        del everything
        record["referrers"] = {
            key: counts.most_common(10) for key, counts in referrers.items()
        }
        record["referrers_2"] = {
            key: counts.most_common(10) for key, counts in referrers_2.items()
        }
    del generations
    if gc_was_enabled:
        gc.enable()
    # statm's 2nd field is the resident set size, in pages.
    with open("/proc/self/statm") as file:
        rss_bytes = int(file.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    record["rss_gib"] = round(rss_bytes / 2**30, 2)
    record["census_seconds"] = round(time.perf_counter() - start, 3)
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")
    logger.info(
        f"Heap census at step {step}: {sum(record['objects_per_generation'])} tracked objects, "
        f"{record['census_seconds']} s -> {path}"
    )


def combine_microbatch_metrics(
    microbatch_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Combine per-microbatch loss metrics over the grad-accumulation: mean/frac keys are pre-normalized
    by num_global_valid_tokens so summing them reconstructs the global value. For keys ending in "max",
    the max value is taken.

    Example:
        # already normalized by num_global_valid_tokens
        combine_microbatch_metrics([{"loss/ratio_clipped_frac": 0.1, "x/max": 2.0},
                                    {"loss/ratio_clipped_frac": 0.2, "x/max": 5.0}])
        # output
        # -> {"loss/ratio_clipped_frac": 0.3, "x/max": 5.0}
    """
    combined: dict[str, float] = {}
    for microbatch in microbatch_metrics:
        for key, value in microbatch.items():
            if key not in combined:
                combined[key] = value
            elif key.endswith("/max"):
                combined[key] = max(combined[key], value)
            elif key.endswith(("/mean", "_mean", "/frac", "_frac")):
                combined[key] += value
    return combined


def compute_perf_ratio_metrics(
    *, num_global_tokens: int, time_metrics: list[m.Metric]
) -> list[m.Metric]:
    """Return trainer throughput and each loop phase's share of `timing/step/total`.

    Example: num_global_tokens=100, timing/step/total = 10 s:

        timing/step/<phase>                        seconds   perf/trainer/step_time_ratio/<phase>
        wait_for_training_batch                    2         0.2
        forward_backward                           4         0.4
        blocking_trainer_push_model_state_dict     1         0.1
        optimizer                                  1         0.1
        blocking_generator_pull_model_state_dict   1         0.1
        (in no phase)                              1         unaccounted: 0.1

        tokens_per_second_full_step         = 100 / 10 = 10
        tokens_per_second_forward_backward  = 100 / 4  = 25

    A phase not recorded this step gets no ratio, and `unaccounted` is skipped.
    `num_global_tokens` counts every non-padding token in the step, prompts included.
    """
    # Each span is recorded once/step; Mean.from_list stores the summed seconds in `.value`.
    seconds = {
        metric.key: metric.value.value
        for metric in time_metrics
        if isinstance(metric.value, m.Mean)
    }
    step_s = seconds.get("timing/step/total")
    if not step_s:  # no step wall-clock -> no denominator to derive ratios from
        return []

    out: list[m.Metric] = []

    def _add_metric(key: str, value: float) -> None:
        out.append(m.Metric(key, m.NoReduce(value)))

    # Throughput over the whole step (includes the idle wait for the next batch).
    _add_metric("perf/trainer/tokens_per_second_full_step", num_global_tokens / step_s)

    # Throughput over forward/backward only (no optimizer step or checkpoint save).
    forward_backward_s = seconds.get("timing/step/forward_backward")
    if forward_backward_s:
        _add_metric(
            "perf/trainer/tokens_per_second_forward_backward",
            num_global_tokens / forward_backward_s,
        )

    # Each phase's share of the step (skip a phase that was not recorded).
    # blocking_*: how long the trainer waited for the overlapped push/pull, not how long it took.
    phase_seconds = {
        phase: seconds.get(f"timing/step/{phase}")
        for phase in (
            "wait_for_training_batch",
            "forward_backward",
            "blocking_trainer_push_model_state_dict",
            "optimizer",
            "blocking_generator_pull_model_state_dict",
        )
    }
    for phase, phase_s in phase_seconds.items():
        if phase_s is not None:
            _add_metric(f"perf/trainer/step_time_ratio/{phase}", phase_s / step_s)

    # Step time the phases don't cover -- only when every phase is present, else it misleads.
    if None not in phase_seconds.values():
        accounted_s = sum(phase_seconds.values())
        _add_metric(
            "perf/trainer/step_time_ratio/unaccounted", (step_s - accounted_s) / step_s
        )

    return out


def compute_policy_age_metrics(
    *,
    trainer_policy_version: int,
    min_policy_versions: list[int],
    target_offpolicy_steps: int,
    max_offpolicy_steps: int | None,
) -> list[m.Metric]:
    """Age of each packed training sample at the moment the trainer consumes the batch.

    Computed in the trainer loop (not at pack time) so the logged age is faithful to the version the
    batch actually trains against. Without a bound (`max_offpolicy_steps=None`) samples older than the
    target are counted and warned about, never rejected. With a bound, exceeding it is an invariant
    failure: the window makes it impossible.

    Args:
        trainer_policy_version: Policy version that will consume this batch.
        min_policy_versions: Oldest sampled policy version for each packed training sample.
        target_offpolicy_steps: Target mean offpolicy steps used to size the active buffer.
        max_offpolicy_steps: Hard consume-time offpolicy step limit, `target + windowed_fifo_batches`;
            None when there is no window.

    Example:
        # trainer at v=10; training samples' oldest versions [8, 9, 5] -> ages [2, 1, 5]
        compute_policy_age_metrics(
            trainer_policy_version=10,
            min_policy_versions=[8, 9, 5],
            target_offpolicy_steps=3,
            max_offpolicy_steps=None,
        )
        # -> train_batch/policy_age mean 2.67, train_batch/policy_age_max 5,
        #    train_batch/pct_samples_over_target_age 33.3, one logger.warning (uncapped, 5 > 3)
        # with max_offpolicy_steps=4 the same batch raises RuntimeError (5 > 4)
    """
    policy_ages = [
        trainer_policy_version - min_policy_version
        for min_policy_version in min_policy_versions
    ]
    max_policy_age = max(policy_ages, default=0)
    if max_offpolicy_steps is not None and max_policy_age > max_offpolicy_steps:
        raise RuntimeError(
            "rollout backpressure admitted stale training data: "
            f"max_policy_age={max_policy_age}, "
            f"target_offpolicy_steps={target_offpolicy_steps}, "
            f"max_offpolicy_steps={max_offpolicy_steps}, "
            f"trainer_policy_version={trainer_policy_version}"
        )
    num_samples_over_target_age = sum(
        policy_age > target_offpolicy_steps for policy_age in policy_ages
    )
    pct_samples_over_target_age = 100.0 * num_samples_over_target_age / len(policy_ages)
    if max_offpolicy_steps is None and num_samples_over_target_age:
        logger.warning(
            f"Training batch contains {num_samples_over_target_age} samples "
            f"({pct_samples_over_target_age:.1f}%) older than target_offpolicy_steps={target_offpolicy_steps} "
            f"(max_policy_age={max_policy_age}, trainer_policy_version={trainer_policy_version}). "
            "Oldest-ready consumption with no cap trains these instead of dropping them; "
            "frequent hits indicate a heavy generation tail."
        )
    return [
        m.Metric("train_batch/policy_age", m.Mean.from_list(policy_ages)),
        m.Metric("train_batch/policy_age_max", m.NoReduce(float(max_policy_age))),
        m.Metric(
            "train_batch/pct_samples_over_target_age",
            m.NoReduce(pct_samples_over_target_age),
        ),
    ]


def compute_rollout_metrics(prefix: str, rollouts: list[Rollout]) -> list[m.Metric]:
    """Build rollout-derived metrics: lengths, truncation, reward breakdown, and each turn's
    per-generation metrics (the latter keep their own generator-side keys, unprefixed).

    Args:
        prefix: Metric namespace (e.g. `"rollout"` or `"validation"`).
        rollouts: Rollouts to compute metrics for.
    """
    # Lengths, truncation, reward
    completion_lens = [
        len(rollout_turn.completion_token_ids)
        for rollout in rollouts
        for rollout_turn in rollout.turns
    ]
    prompt_lens = [
        rollout.turns[0].num_prompt_tokens for rollout in rollouts if rollout.turns
    ]
    total_lens = [
        rollout.turns[-1].num_prompt_tokens
        + len(rollout.turns[-1].completion_token_ids)
        for rollout in rollouts
        if rollout.turns
    ]

    truncated = [float(rollout.status.is_truncated()) for rollout in rollouts]
    rewards = [rollout.reward for rollout in rollouts if rollout.reward is not None]
    num_turns = [float(len(rollout.turns)) for rollout in rollouts]

    out: list[m.Metric] = [
        m.Metric(f"{prefix}/output_tokens", m.Mean.from_list(completion_lens)),
        m.Metric(f"{prefix}/output_tokens", m.Std.from_list(completion_lens)),
        m.Metric(f"{prefix}/output_tokens", m.Max.from_list(completion_lens)),
        m.Metric(f"{prefix}/response_length", m.Mean.from_list(completion_lens)),
        m.Metric(f"{prefix}/response_length", m.Max.from_list(completion_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Mean.from_list(prompt_lens)),
        m.Metric(f"{prefix}/prompt_length", m.Max.from_list(prompt_lens)),
        m.Metric(f"{prefix}/total_length", m.Mean.from_list(total_lens)),
        m.Metric(f"{prefix}/total_length", m.Max.from_list(total_lens)),
        m.Metric(f"{prefix}/num_turns", m.Mean.from_list(num_turns)),
        m.Metric(f"{prefix}/num_turns", m.Max.from_list(num_turns)),
        m.Metric(f"{prefix}/truncation_rate", m.Mean.from_list(truncated)),
        m.Metric(f"{prefix}_reward", m.SummaryStats.from_list(rewards)),
    ]

    # Per-component reward breakdown
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for name, value in rollout.reward_breakdown.items():
            values_by_name[name].append(float(value))
    out.extend(
        m.Metric(f"{prefix}_reward/component/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    )

    # Per-generation turn metrics (latencies, output tokens) measured by the generator.
    # They carry their own keys (e.g. "generator/..."), so they ride through unprefixed.
    # Fold across turns: group metrics outlive the rollouts (until the step is logged),
    # and every full GC walks them.
    out.extend(
        m.fold(
            metric
            for rollout in rollouts
            for rollout_turn in rollout.turns
            for metric in rollout_turn.metrics
        )
    )
    return out


def _describe(obj: object) -> str:
    """A short label for one object in a garbage chain: its type, plus where code objects live."""
    name = _type_name(obj)
    if isinstance(obj, types.FrameType):
        code = obj.f_code
        return f"{name} {code.co_name} ({code.co_filename}:{obj.f_lineno})"
    if isinstance(obj, types.FunctionType):
        code = obj.__code__
        return f"{name} {obj.__qualname__} ({code.co_filename}:{code.co_firstlineno})"
    if isinstance(obj, types.CoroutineType):
        return f"{name} {obj.__qualname__}"
    if isinstance(obj, types.MethodType):
        return f"{name} {obj.__func__.__qualname__}"
    if isinstance(obj, dict):
        return f"{name} keys={list(obj)[:6]}"
    if isinstance(obj, BaseException):
        return f"{name} {str(obj)[:120]!r}"
    return name


def collect_and_explain_garbage(
    *, path: str, step: int, sample_types: tuple[str, ...], num_samples: int = 5
) -> int:
    """Run a full collection that keeps its garbage (`gc.DEBUG_SAVEALL`), and append one JSON line to
    `path` that names the reference cycles: garbage counted by type, and for a few garbage objects of
    each `sample_types` type, the chain of garbage objects that refer to it, walked back until it
    closes a cycle. Clears `gc.garbage` afterwards. Returns the number of objects collected.

    Example:
        collect_and_explain_garbage(path="garbage.jsonl", step=8, sample_types=("RolloutTurn",))
        # {"step": 8, "collected": 1096819, "top_types": [["builtins.dict", 412000], ...],
        #  "chains": {"RolloutTurn": [["...RolloutTurn", "builtins.list", "...Rollout", ...,
        #                              "builtins.frame _run (events.py:88)", "cycle -> 3"]]}}
    """
    start = time.perf_counter()
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        collected = gc.collect()
    finally:
        gc.set_debug(0)
    garbage = gc.garbage[:]
    gc.garbage.clear()
    garbage_ids = {id(obj): obj for obj in garbage}
    referrers: dict[int, list[object]] = defaultdict(list)
    for obj in garbage:
        for referent in gc.get_referents(obj):
            if id(referent) in garbage_ids:
                referrers[id(referent)].append(obj)
    chains: dict[str, list[list[str]]] = {}
    for wanted in sample_types:
        samples = [obj for obj in garbage if type(obj).__qualname__ == wanted][:num_samples]
        chains[wanted] = []
        for obj in samples:
            chain, seen, current = [], {}, obj
            for _ in range(40):
                if id(current) in seen:
                    chain.append(f"cycle -> {seen[id(current)]}")
                    break
                seen[id(current)] = len(chain)
                chain.append(_describe(current))
                holders = referrers.get(id(current))
                if not holders:
                    chain.append("(no garbage referrer)")
                    break
                current = holders[0]
            chains[wanted].append(chain)
    record = {
        "step": step,
        "collected": collected,
        "top_types": [
            [name, count]
            for name, count in Counter(map(_type_name, garbage)).most_common(40)
        ],
        "chains": chains,
        "seconds": round(time.perf_counter() - start, 2),
    }
    del garbage, garbage_ids, referrers
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")
    logger.info(f"Garbage at step {step}: {collected} objects; explained in {path}")
    return collected
