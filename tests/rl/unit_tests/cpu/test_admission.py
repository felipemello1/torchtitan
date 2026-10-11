# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`KVGrowthEstimateAdmission` on a fake clock. 10-token blocks and no fixed blocks: an N-token
session holds N / 10 blocks, and a max-length context (200 tokens) 20."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import torchtitan.rl.distributed.routing.admission as admission_module
from torchtitan.rl.distributed.routing.admission import KVGrowthEstimateAdmission
from torchtitan.rl.distributed.routing.types import KVCacheBudget

_BUDGET = KVCacheBudget(
    num_blocks=1000,
    block_size=10,
    num_growing_groups=1,
    fixed_blocks_per_session=0,
    max_model_len=200,
)


class _Clock:
    """Stands in for `time.monotonic` in the admission module only."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr(
            admission_module, "time", SimpleNamespace(monotonic=lambda: self.now)
        )


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    return _Clock(monkeypatch)


def _policy(limit=1.0, group_size=1) -> KVGrowthEstimateAdmission:
    return KVGrowthEstimateAdmission.Config(limit=limit).build(
        budgets=[_BUDGET, _BUDGET], group_size=group_size
    )


def _admit(policy, group_id: int, serving=(0, 1)) -> int | None:
    return policy.admit(group_id, serving=list(serving))


def _play(policy, group_id: int, session: str, *contexts: int | None) -> None:
    """Make a session's calls: each context is its tokens at a call start or end; None ends it."""
    for num_tokens in contexts:
        if num_tokens is None:
            policy.release_session(group_id, session_id=session)
        else:
            policy.set_session_tokens(
                group_id, session_id=session, num_tokens=num_tokens
            )


def test_r_is_the_growth_per_ended_session(clock):
    policy = _policy()
    assert _admit(policy, 0) == 0
    # 49 sessions grow from 1 to 7 blocks, then end: still the cold start.
    for i in range(49):
        _play(policy, 0, f"s{i}", 10, 70, None)
    policy.summary()
    assert policy.growth_per_session is None
    # The 50th ended session gives R = growth / ends: (49 x 6 + 12) / 50 = 6.12 blocks.
    _play(policy, 0, "s49", 10, 130, None)
    policy.summary()
    assert policy.growth_per_session == pytest.approx(6.12)
    # 15 min with no session ending: R keeps its last estimate; the cold start does not come back.
    clock.now += 901.0
    policy.summary()
    assert policy.growth_per_session == pytest.approx(6.12)
    # Then only the sessions that ended in the last 15 min count: 2 that grew 3 blocks.
    for i in range(2):
        _play(policy, 0, f"late{i}", 10, 40, None)
    policy.summary()
    assert policy.growth_per_session == pytest.approx(3.0)


def test_cold_start_ends_after_50_sessions_however_slowly_they_end(clock):
    policy = _policy()
    assert _admit(policy, 0) == 0
    # Long sessions: 30 end, then 20 more 20 minutes later, so the 15-min window never holds 50.
    for i in range(30):
        _play(policy, 0, f"early{i}", 10, 40, None)
    clock.now += 1200.0
    for i in range(20):
        _play(policy, 0, f"late{i}", 10, 50, None)
    policy.summary()
    # R comes from the 20 in the window, which grew 4 blocks each.
    assert policy.growth_per_session == pytest.approx(4.0)


def test_every_session_counts_as_a_max_length_context_until_50_end():
    policy = _policy()
    assert _admit(policy, 0) == 0
    # 49 sessions end having grown G = 10 (25 of them) or 2 (24) blocks; a live one grew g = 4.
    for i in range(49):
        _play(policy, 0, f"ended{i}", 10, 110 if i < 25 else 30, None)
    _play(policy, 0, "live", 10, 50)
    assert _admit(policy, 1) == 1
    policy.summary()
    # Fewer than 50 ended: the live session, and group 1's session not started yet, count as
    # max-length contexts.
    assert policy.charged == [20.0, 20.0]
    # The 50th ends: R = (25 x 10 + 25 x 2 + 4) / 50 = 6.08. The live session expects
    # max(10 - 4, R - 4) = 6 more; group 1's charges the mean first call, 1, plus R.
    _play(policy, 0, "ended49", 10, 30, None)
    policy.summary()
    assert policy.charged == pytest.approx([5 + 6.0, 1 + 6.08])


def test_live_session_expects_its_remaining_growth():
    policy = _policy()
    assert _admit(policy, 0) == 0
    # 50 sessions end having grown G = 4 (26 of them) or 16 (24) blocks.
    for i in range(50):
        _play(policy, 0, f"ended{i}", 10, 50 if i < 26 else 170, None)
    # Two live sessions grew g = 0 and g = 12: R = (26 x 4 + 24 x 16 + 12) / 50 = 10.
    _play(policy, 0, "young", 10)
    _play(policy, 0, "old", 10, 130)
    policy.summary()
    assert policy.growth_per_session == pytest.approx(10.0)
    # Each charges its blocks plus max(mean(G - g | G > g), R - g): young 1 + max(9.76, 10),
    # old 13 + max(16 - 12, -2).
    assert policy.charged[0] == pytest.approx(1 + 10.0 + 13 + 4.0)


def test_remaining_growth_averages_the_ended_sessions_that_grew_more():
    remaining = admission_module._remaining_growth([4, 10, 16])
    # A session that grew 4 counts only 10 and 16 as above it; past 16, none is.
    assert [remaining(grown) for grown in (0, 4, 12, 16, 20)] == [10, 9, 4, 0, 0]


def test_session_not_started_charges_the_mean_first_call_plus_r():
    policy = _policy(group_size=2)
    assert _admit(policy, 0) == 0
    # 50 sessions end having grown 5 blocks: R = 5.
    for i in range(50):
        _play(policy, 0, f"ended{i}", 10, 60, None)
    policy.release_group(0)
    # Group 1's first session starts at 3 blocks; its other session has not started.
    assert _admit(policy, 1) == 0
    _play(policy, 1, "a", 30)
    policy.summary()
    assert policy.charged == [(3 + 5.0) + (3 + 5.0), 0.0]
    # With no training session live, a new session still charges the last mean first call, 3:
    # "a" ended having grown 0, so R = 250 / 51.
    _play(policy, 1, "a", None)
    policy.release_group(1)
    assert _admit(policy, 2) == 0
    policy.summary()
    assert policy.charged[0] == pytest.approx(2 * (3 + 250 / 51))


def test_no_session_charges_more_than_a_max_length_context():
    policy = _policy()
    assert _admit(policy, 0) == 0
    # 200 live sessions grow 5 blocks each while only 50 end, having grown 1: just after the cold
    # start, R = (200 x 5 + 50 x 1) / 50 = 21 blocks, past a max-length context (20).
    for i in range(200):
        _play(policy, 0, f"live{i}", 10, 60)
    for i in range(50):
        _play(policy, 0, f"ended{i}", 10, 20, None)
    assert _admit(policy, 1) == 1
    policy.summary()
    assert policy.growth_per_session == pytest.approx(21.0)
    # A live session: 6 + max(0, 21 - 5) = 22 -> 20. Group 1's session not started: 1 + 21 -> 20.
    assert policy.charged == [200 * 20.0, 20.0]


def test_sessions_live_at_group_release_count_as_ended():
    policy = _policy(group_size=2)
    # Sessions that end only with their group, e.g. after their rollout raised: 25 groups of 2
    # sessions that grew 3 blocks.
    for group_id in range(25):
        _admit(policy, group_id)
        for session in ("s0", "s1"):
            _play(policy, group_id, session, 10, 40)
        policy.release_group(group_id)
    # 50 sessions ended, so R is learned: 50 x 3 / 50.
    policy.summary()
    assert policy.growth_per_session == pytest.approx(3.0)


def test_validation_sessions_teach_nothing():
    policy = _policy()
    # 60 validation groups (negative ids) of one session that grows 2 blocks, then ends.
    for i in range(1, 61):
        assert _admit(policy, -i) is not None
        _play(policy, -i, "v", 10, 30, None)
        policy.release_group(-i)
    policy.summary()
    assert policy.growth_per_session is None


def test_new_group_waits_on_the_generator_with_the_most_room():
    # 70 blocks per generator; each group counts one 20-block context.
    policy = _policy(limit=0.07)
    assert [_admit(policy, group_id) for group_id in range(7)] == [
        0,
        1,
        0,
        1,
        0,
        1,
        None,
    ]
    policy.summary()
    assert policy.charged == [60.0, 60.0]


def test_generator_charging_nothing_goes_first_even_with_less_room():
    # Limits 1,000 and 980 blocks; a group of 60 sessions counts 1,200, over both.
    policy = KVGrowthEstimateAdmission.Config(limit=1.0).build(
        budgets=[_BUDGET, replace(_BUDGET, num_blocks=980)], group_size=60
    )
    # A validation session on generator 0 leaves both with 980 blocks of room.
    assert _admit(policy, -1) == 0
    assert _admit(policy, 0) == 1


def test_generator_charging_nothing_takes_a_group_larger_than_its_limit():
    # 10 blocks per generator; a group of 2 sessions counts 40.
    policy = _policy(limit=0.01, group_size=2)
    assert [_admit(policy, group_id) for group_id in range(3)] == [0, 1, None]
    policy.release_group(1)
    # Group 0's sessions both end; the group stays placed, but charges nothing.
    for session in ("a", "b"):
        _play(policy, 0, session, 10, None)
    assert _admit(policy, 2) == 0
    assert _admit(policy, 3) == 1


def test_validation_groups_never_wait_and_count_one_session():
    policy = _policy(limit=0.01, group_size=2)
    assert [_admit(policy, group_id) for group_id in range(3)] == [0, 1, None]
    # Both generators are over their limit; validation groups still start, one session each.
    assert _admit(policy, -1) == 0
    assert _admit(policy, -2) == 1
    policy.summary()
    assert policy.charged == [40.0 + 20.0, 40.0 + 20.0]


def test_idle_group_frees_its_generator_for_any_group(clock, monkeypatch):
    monkeypatch.setattr(admission_module, "IDLE_TTL_S", 100.0)
    # 10 blocks per generator, one generator serving; a group of 2 sessions counts 40.
    policy = _policy(limit=0.01, group_size=2)
    assert _admit(policy, 0, serving=[0]) == 0
    _play(policy, 0, "hung", 10)
    assert _admit(policy, 1, serving=[0]) is None
    # Group 0 makes no call for longer than the TTL: generator 0 charges nothing again.
    clock.now += 101.0
    assert _admit(policy, 1, serving=[0]) == 0


def test_ended_session_and_released_group_free_their_charge():
    policy = _policy(limit=0.02)  # 20 blocks: one max-length context
    assert _admit(policy, 0, serving=[0]) == 0
    _play(policy, 0, "s0", 100)
    assert _admit(policy, 1, serving=[0]) is None
    _play(policy, 0, "s0", None)
    assert _admit(policy, 1, serving=[0]) == 0
    assert _admit(policy, 2, serving=[0]) is None
    policy.release_group(1)
    assert _admit(policy, 2, serving=[0]) == 0


def test_call_returning_after_its_session_ended_charges_nothing():
    policy = _policy()
    assert _admit(policy, 0) == 0
    # The rollout releases its session while a call is still running (e.g. an agent timeout); the
    # call returns afterwards.
    _play(policy, 0, "t", 50, None, 120)
    policy.summary()
    assert policy.charged[0] == 0.0


def test_session_that_ended_before_its_first_call_stops_charging():
    policy = _policy(group_size=2)
    assert _admit(policy, 0) == 0
    policy.summary()
    assert policy.charged[0] == 2 * 20.0
    # One rollout fails before its first call; its session will never start.
    policy.release_session(0, session_id="never")
    policy.summary()
    assert policy.charged[0] == 20.0
