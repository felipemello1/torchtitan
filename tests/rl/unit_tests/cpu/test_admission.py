# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Admission policies. `KVUsageAdmission` is checked against prime-rl's controller: the expected
caps come from prime-rl's constants (prime-rl@0bcb868e9,
`src/prime_rl/orchestrator/concurrency.py:42-89`), written out as numbers."""

from types import SimpleNamespace

import pytest

import torchtitan.rl.distributed.routing.admission as admission_module
from torchtitan.rl.distributed.routing.admission import (
    KVGrowthEstimateAdmission,
    KVUsageAdmission,
)
from torchtitan.rl.distributed.routing.types import EngineLoad, KVCacheBudget

# 1,000 blocks of 10 tokens and 100-token contexts: each generator holds 100 max-length contexts.
_BUDGET = KVCacheBudget(
    num_blocks=1000,
    block_size=10,
    num_growing_groups=1,
    fixed_blocks_per_session=0,
    max_model_len=100,
)


class _Clock:
    """Stands in for `time.monotonic` in the admission module only."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(
            admission_module, "time", SimpleNamespace(monotonic=lambda: self.now)
        )


def _policy(**config) -> KVUsageAdmission:
    config.setdefault("initial_inflight", 100)
    return KVUsageAdmission.Config(**config).build(
        budgets=[_BUDGET, _BUDGET], group_size=1
    )


def _loads(
    kv_usage: float = 0.5,
    *,
    num_waiting: int = 0,
    num_preemptions: int = 0,
) -> dict[int, EngineLoad]:
    """Two generators: the first carries the given load, the second an idle one."""
    return {
        0: EngineLoad(
            kv_usage=kv_usage,
            num_running=10,
            num_waiting=num_waiting,
            num_waiting_for_capacity=num_waiting,
            num_preemptions=num_preemptions,
        ),
        1: EngineLoad(
            kv_usage=0.0,
            num_running=10,
            num_waiting=0,
            num_waiting_for_capacity=0,
            num_preemptions=0,
        ),
    }


def _with_groups_in_flight(policy: KVUsageAdmission, num_groups: int):
    policy.admitted_groups = set(range(num_groups))
    return policy


def _admit(policy, group_id: int, *, num_prompt_tokens=10, skip_queue=False):
    return policy.admit(
        group_id,
        num_prompt_tokens=num_prompt_tokens,
        serving=[0, 1],
        skip_queue=skip_queue,
    )


@pytest.mark.parametrize(
    "config, expected_cap",
    [
        # 2 generators x 100 contexts, one per sample; 4 samples per group.
        ({"initial_inflight": None}, 200 // 4),
        ({"initial_inflight": 7}, 7),
        ({"initial_inflight": 2000}, 1024),  # max_inflight
        ({"initial_inflight": None, "max_inflight": None, "min_inflight": 60}, 60),
    ],
)
def test_initial_cap(config, expected_cap):
    policy = KVUsageAdmission.Config(**config).build(
        budgets=[_BUDGET, _BUDGET], group_size=4
    )
    assert int(policy.cap) == expected_cap


@pytest.mark.parametrize(
    "kv_usage, multiplier",
    [(0.0, 1.2), (0.4, 1.1), (0.6, 1.05), (0.8, 1.0)],
)
def test_each_finished_group_grows_the_cap_by_the_tapered_multiplier(
    kv_usage, multiplier
):
    policy = _with_groups_in_flight(_policy(), 100)
    policy.observe(_loads(kv_usage))
    assert policy.growth_multiplier == pytest.approx(multiplier)

    # 1/100 of a turnover of the 100 groups in flight.
    policy.release(0, num_completion_tokens=50)
    assert policy.cap == pytest.approx(100 * multiplier ** (1 / 100))


@pytest.mark.parametrize(
    "case",
    ["cap not binding", "vLLM queue", "no tokens", "stale poll", "high usage"],
)
def test_cap_does_not_grow(case, monkeypatch):
    clock = _Clock(monkeypatch)
    in_flight = 89 if case == "cap not binding" else 100  # binding at 90
    policy = _with_groups_in_flight(_policy(), in_flight)
    policy.observe(
        _loads(
            0.85 if case == "high usage" else 0.5,
            num_waiting=5 if case == "vLLM queue" else 0,
        )
    )
    if case == "stale poll":
        clock.now += 15.0

    policy.release(0, num_completion_tokens=0 if case == "no tokens" else 50)
    if case == "high usage":
        assert policy.cap == 82  # trimmed, never grown
    else:
        assert policy.cap == 100


@pytest.mark.parametrize(
    "kv_usage, expected_cap, trim",
    [
        (0.80, 100, "none"),
        (0.85, 82, "soft"),  # 100 x 0.7 / 0.85
        (0.95, 73, "hard"),  # 100 x 0.7 / 0.95, no group cancelled
    ],
)
def test_usage_trims_the_cap_to_the_target(kv_usage, expected_cap, trim):
    policy = _with_groups_in_flight(_policy(), 100)
    policy.observe(_loads(kv_usage))
    assert policy.cap == expected_cap
    assert len(policy.admitted_groups) == 100
    assert policy.last_change.startswith(trim)


def test_trims_wait_out_the_cooldown():
    policy = _with_groups_in_flight(_policy(), 100)
    policy.observe(_loads(0.85))
    assert policy.cap == 82
    for group_id in range(10):
        policy.release(group_id, num_completion_tokens=50)

    for _ in range(5):
        policy.observe(_loads(0.85))
        assert policy.cap == 82
    policy.observe(_loads(0.85))
    assert policy.cap == 74  # 90 x 0.7 / 0.85, six polls after the first trim


def test_preemption_cuts_then_drains_then_escalates():
    policy = _with_groups_in_flight(_policy(), 100)
    # A generator's first answer is the baseline, not a new preemption.
    policy.observe(_loads(num_preemptions=3))
    assert policy.cap == 100

    policy.observe(_loads(num_preemptions=4))
    assert (policy.cap, policy.draining) == (80, True)  # 0.8 x 100
    # Preemptions while draining are the cut groups' backlog: no second cut.
    policy.observe(_loads(num_preemptions=9))
    assert policy.cap == 80

    for group_id in range(20):
        policy.release(group_id, num_completion_tokens=50)
    policy.observe(_loads(num_preemptions=9))
    assert not policy.draining
    # A repeat within the grace window cuts to half.
    policy.observe(_loads(num_preemptions=10))
    assert policy.cap == 40  # 0.5 x 80


@pytest.mark.parametrize(
    "in_flight, num_polls, expected_cap",
    [
        (100, 5, 100),
        (100, 6, 90),  # 0.9 x 100
        (60, 6, 100),  # the cap is not full yet, e.g. during the startup ramp
    ],
)
def test_persistent_vllm_queue_cuts_a_full_cap(in_flight, num_polls, expected_cap):
    policy = _with_groups_in_flight(_policy(), in_flight)
    # 11 waiting > 0.5 x 20 running, summed over both generators.
    for _ in range(num_polls):
        policy.observe(_loads(num_waiting=11))
    assert policy.cap == expected_cap


@pytest.mark.parametrize("initial_inflight, burst", [(100, 10), (5, 1)])
def test_admissions_per_window_are_smoothed(initial_inflight, burst, monkeypatch):
    clock = _Clock(monkeypatch)
    policy = _policy(initial_inflight=initial_inflight)
    policy.observe(_loads())
    admitted = [_admit(policy, group_id) for group_id in range(burst + 1)]
    assert admitted.count(None) == 1 and admitted[-1] is None

    # A finished group gives its admission back; a new 5 s window gives a new burst, poll or not.
    policy.release(0, num_completion_tokens=50)
    assert _admit(policy, 100) is not None
    assert _admit(policy, 101) is None
    clock.now += 5.0
    assert _admit(policy, 101) is not None


def test_new_groups_spread_from_the_lowest_usage_generator():
    policy = _policy()
    policy.observe(_loads(0.5))  # generator 1 is idle
    assert [_admit(policy, group_id) for group_id in range(3)] == [1, 0, 1]
    policy.observe(_loads(0.5))
    assert _admit(policy, 3) == 1


def test_validation_groups_take_no_permit():
    policy = _with_groups_in_flight(_policy(), 100)
    policy.observe(_loads())
    assert _admit(policy, 100) is None
    assert _admit(policy, -1, skip_queue=True) == 1

    policy.release(-1, num_completion_tokens=50)
    assert len(policy.admitted_groups) == 100


# KVGrowthEstimateAdmission. 10-token blocks, no fixed blocks: an N-token session holds N / 10.
# Its reserve R starts at half a max-length context: 5 blocks.


def _growth_policy(limit=1.0, group_size=1) -> KVGrowthEstimateAdmission:
    return KVGrowthEstimateAdmission.Config(limit=limit).build(
        budgets=[_BUDGET, _BUDGET], group_size=group_size
    )


def _play(policy, group_id: int, session: str, *contexts: int | None) -> None:
    for num_tokens in contexts:
        policy.set_session_tokens(group_id, session_id=session, num_tokens=num_tokens)


def test_growth_reserve_is_the_growth_per_ended_session(monkeypatch):
    clock = _Clock(monkeypatch)
    policy = _growth_policy()
    assert _admit(policy, 0) == 0
    # 49 sessions grow from 1 to 7 blocks, then end: the reserve is still the initial 5.
    for i in range(49):
        _play(policy, 0, f"s{i}", 10, 70, None)
    assert policy._growth_per_session() == 5.0
    # The 50th ended session switches to growth / ends: (49 x 6 + 12) / 50 = 6.12 blocks.
    _play(policy, 0, "s49", 10, 130, None)
    assert policy._growth_per_session() == pytest.approx(6.12)
    # Growth and ends older than 15 min leave the window.
    clock.now += 901.0
    assert policy._growth_per_session() == 5.0


def test_growth_reserve_is_charged_to_every_session():
    policy = _growth_policy(limit=0.1)  # 100 blocks per generator
    # Each group reserves 2 seats of 10 + 5 blocks, then its 2 sessions grow to 40 blocks each.
    for group_id, generator in [(0, 0), (1, 1)]:
        assert _admit(policy, group_id, num_prompt_tokens=100) == generator
        for session in ("s0", "s1"):
            _play(policy, group_id, session, 100, 400)
    policy.observe({})
    # Both generators: 80 blocks + 2 sessions x 5 = 90; a third group would add 2 x 15.
    # Without the reserve, 80 + 2 x 10 = 100 would fit.
    assert _admit(policy, 2, num_prompt_tokens=100) is None
    assert policy.summary() == (
        "growth 5 blocks per session; KV blocks / limit per generator: [0.9, 0.9]"
    )


def test_live_session_reserves_its_expected_remaining_growth():
    policy = _growth_policy()
    assert _admit(policy, 0) == 0
    # 50 sessions end having grown G = 4 (26 of them) or 16 (24) blocks.
    for i in range(50):
        _play(policy, 0, f"ended{i}", 10, 50 if i < 26 else 170, None)
    # Two live sessions grew g = 0 and g = 12: R = (26 x 4 + 24 x 16 + 12) / 50 = 10.
    _play(policy, 0, "young", 10)
    _play(policy, 0, "old", 10, 130)
    assert policy._growth_per_session() == pytest.approx(10.0)
    policy.observe({})
    # Each reserves max(mean(G - g | G > g), R - g): young max(9.76, 10), old max(16 - 12, -2).
    assert policy._reserved_growth(0) == pytest.approx(10.0 + 4.0)


def test_new_group_reserves_the_mean_sessions_of_finished_groups():
    policy = _growth_policy(group_size=8)
    # Until a group finishes: two sessions per sample, or the largest group so far.
    assert _admit(policy, 0) == 0
    assert policy._new_group_sessions() == 16
    for i in range(18):
        _play(policy, 0, f"g0/s{i}", 10)
    assert policy._new_group_sessions() == 18
    # Finished groups opened 18 and 7 sessions; validation groups (negative ids) do not count.
    policy.release(0, num_completion_tokens=50)
    for group_id, num_sessions in [(1, 7), (-1, 1)]:
        _admit(policy, group_id, skip_queue=group_id < 0)
        for i in range(num_sessions):
            _play(policy, group_id, f"g{group_id}/s{i}", 10)
        policy.release(group_id, num_completion_tokens=50)
    assert policy._new_group_sessions() == 13  # ceil((18 + 7) / 2)
