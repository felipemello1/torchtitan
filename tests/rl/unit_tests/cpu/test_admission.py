# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`UsageAIMDAdmission` against prime-rl's controller: the expected caps below come from prime-rl's
constants (`src/prime_rl/orchestrator/concurrency.py:42-89`), written out as numbers."""

import pytest

import torchtitan.rl.distributed.routing.admission as admission_module
from torchtitan.rl.distributed.routing.admission import UsageAIMDAdmission
from torchtitan.rl.distributed.routing.types import EngineLoad, KVCacheBudget

# 1,000 blocks of 10 tokens and 100-token contexts: each generator holds 100 max-length contexts.
_BUDGET = KVCacheBudget(
    num_blocks=1000,
    block_size=10,
    num_growing_groups=1,
    fixed_blocks_per_session=0,
    max_model_len=100,
)


def _policy(**config) -> UsageAIMDAdmission:
    config.setdefault("initial_inflight", 100)
    return UsageAIMDAdmission.Config(**config).build(
        budgets=[_BUDGET, _BUDGET], group_size=1
    )


def _loads(
    kv_usage: float = 0.5,
    *,
    num_waiting: int = 0,
    num_preemptions: int = 0,
) -> list[EngineLoad]:
    """Two generators: the first carries the given load, the second an idle one."""
    return [
        EngineLoad(
            kv_usage=kv_usage,
            num_running=10,
            num_waiting=num_waiting,
            num_preemptions=num_preemptions,
        ),
        EngineLoad(kv_usage=0.0, num_running=10, num_waiting=0, num_preemptions=0),
    ]


def _with_groups_in_flight(policy: UsageAIMDAdmission, num_groups: int):
    policy.admitted_groups = set(range(num_groups))
    return policy


def _admit(policy: UsageAIMDAdmission, group_id: int, *, skip_queue=False):
    return policy.admit(
        group_id, num_prompt_tokens=10, serving=[0, 1], skip_queue=skip_queue
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
    policy = UsageAIMDAdmission.Config(**config).build(
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
    in_flight = 89 if case == "cap not binding" else 100  # binding at 90
    policy = _with_groups_in_flight(_policy(), in_flight)
    policy.observe(
        _loads(
            0.85 if case == "high usage" else 0.5,
            num_waiting=5 if case == "vLLM queue" else 0,
        )
    )
    if case == "stale poll":
        now = admission_module.time.monotonic()
        monkeypatch.setattr(admission_module.time, "monotonic", lambda: now + 15.0)

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
    # The first poll's count is the baseline, not a new preemption.
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


@pytest.mark.parametrize("num_polls, expected_cap", [(5, 100), (6, 90)])
def test_persistent_vllm_queue_cuts_the_cap(num_polls, expected_cap):
    policy = _with_groups_in_flight(_policy(), 100)
    # 11 waiting > 0.5 x 20 running, summed over both generators.
    for _ in range(num_polls):
        policy.observe(_loads(num_waiting=11))
    assert policy.cap == expected_cap


@pytest.mark.parametrize("initial_inflight, burst", [(100, 10), (5, 1)])
def test_admissions_per_poll_are_smoothed(initial_inflight, burst):
    policy = _policy(initial_inflight=initial_inflight)
    policy.observe(_loads())
    admitted = [_admit(policy, group_id) for group_id in range(burst + 1)]
    assert admitted.count(None) == 1 and admitted[-1] is None

    # A finished group gives its admission back; the next poll gives a new burst.
    policy.release(0, num_completion_tokens=50)
    assert _admit(policy, 100) is not None
    policy.observe(_loads())
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
