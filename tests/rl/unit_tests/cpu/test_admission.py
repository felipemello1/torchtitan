# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.rl.generator.engine import KVCacheBudget
from torchtitan.rl.routing.admission import KVRoomAdmission


def _budget(num_blocks: int = 3411) -> KVCacheBudget:
    """r8's engine (Qwen3.8-27B, MTP-3): 3,030 usable blocks; a 4,000-token session holds 6 blocks
    idle and charges 22 while it runs."""
    return KVCacheBudget(
        num_blocks=num_blocks,
        free_floor_blocks=381,
        attention_block_sizes=(1664,),
        num_mamba_groups=3,
        mamba_blocks_per_running_request=4,
        max_model_len=262144,
        data_parallel_size=1,
    )


def _policy(*budgets: KVCacheBudget) -> KVRoomAdmission:
    return KVRoomAdmission.Config().build(budgets=list(budgets or [_budget()]))


def _seed_first_prompt(policy: KVRoomAdmission, num_tokens: int = 4000) -> None:
    """Run one call of a 1-session group, so new sessions are charged `num_tokens` prompts."""
    policy.add_group(10**6, num_sessions=1, validation=False)
    policy.place()
    policy.start_call(
        "seed", session_id="seed", group_id=10**6, num_prompt_tokens=num_tokens
    )
    policy.end_call("seed", num_tokens=num_tokens)
    policy.release_group(10**6)


def test_budget_counts_what_the_engine_holds():
    budget = _budget()

    assert budget.usable_blocks == 3030
    # 3 attention blocks + 1 state block per Gated-DeltaNet group
    assert budget.idle_session_blocks(4000) == 6
    # + 3 groups x (1 + 3 MTP blocks) + 1 floor block per KV cache group
    assert budget.running_session_blocks(4000) == 22


def test_cold_start_places_one_group_per_generator_until_a_first_prompt_is_seen():
    policy = _policy(_budget(), _budget())
    for group_id in range(3):
        policy.add_group(group_id, num_sessions=16, validation=False)

    # A max-length session charges 158 + 3 + 12 + 4 = 177: one group is 2,832 of 2,878.
    assert policy.place() == {0: 0, 1: 1}

    policy.start_call("r0", session_id="g0/s0", group_id=0, num_prompt_tokens=4000)
    # Group 0's sessions and new ones now charge 22 each: group 2 fits next to group 0.
    assert policy.place() == {2: 0}


def test_validation_goes_first_by_arrival_then_training_by_id():
    policy = _policy(_budget(num_blocks=100_000))
    policy.add_group(5, num_sessions=16, validation=False)
    policy.add_group(-2, num_sessions=2, validation=True)
    policy.add_group(3, num_sessions=16, validation=False)
    policy.add_group(-1, num_sessions=2, validation=True)

    assert list(policy.place()) == [-2, -1, 3, 5]


def test_a_group_that_fits_nowhere_keeps_later_groups_waiting():
    policy = _policy()
    _seed_first_prompt(policy)
    policy.add_group(1, num_sessions=120, validation=False)  # 120 x 22 = 2,640
    assert policy.place() == {1: 0}

    policy.add_group(2, num_sessions=16, validation=False)  # 2,640 + 352 > 2,878
    policy.add_group(3, num_sessions=2, validation=False)  # would fit alone
    assert policy.place() == {}
    assert policy.num_waiting_groups == 2


def test_training_goes_to_the_most_room_and_validation_to_the_fewest_validation_sessions():
    policy = _policy(_budget(), _budget())
    _seed_first_prompt(policy)

    policy.add_group(-1, num_sessions=2, validation=True)
    assert policy.place() == {-1: 0}
    policy.add_group(1, num_sessions=16, validation=False)
    assert policy.place() == {1: 1}  # generator 1 has more room
    policy.add_group(-2, num_sessions=2, validation=True)
    assert policy.place() == {-2: 1}  # less room, but no validation session yet


def test_a_late_release_of_an_older_call_keeps_the_newer_calls_size():
    budget = _budget()
    policy = _policy(budget)
    policy.add_group(0, num_sessions=1, validation=False)
    policy.place()

    policy.start_call("turn0", session_id="s", group_id=0, num_prompt_tokens=4000)
    policy.start_call("turn1", session_id="s", group_id=0, num_prompt_tokens=6000)
    policy.end_call("turn0", num_tokens=5000)
    # turn1 still runs, at its own prompt.
    assert policy.ledger(0) == budget.running_session_blocks(6000)

    policy.end_call("turn1", num_tokens=7000)
    assert policy.ledger(0) == budget.idle_session_blocks(7000)


def test_a_compacted_prompt_lowers_the_charge():
    budget = _budget()
    policy = _policy(budget)
    policy.add_group(0, num_sessions=1, validation=False)
    policy.place()
    policy.start_call("turn0", session_id="s", group_id=0, num_prompt_tokens=200_000)
    policy.end_call("turn0", num_tokens=201_000)

    policy.start_call("turn1", session_id="s", group_id=0, num_prompt_tokens=10_000)

    assert policy.ledger(0) == budget.running_session_blocks(10_000)


def test_not_started_siblings_are_charged_their_first_siblings_prompt():
    budget = _budget()
    policy = _policy(budget)
    policy.add_group(0, num_sessions=4, validation=False)
    policy.place()

    policy.start_call("r0", session_id="s0", group_id=0, num_prompt_tokens=9000)

    assert policy.ledger(0) == 4 * budget.running_session_blocks(9000)


def test_released_sessions_stop_charging_and_late_releases_are_ignored():
    budget = _budget()
    policy = _policy(budget)
    policy.add_group(0, num_sessions=2, validation=False)
    policy.place()
    policy.start_call("r0", session_id="s0", group_id=0, num_prompt_tokens=4000)
    policy.start_call("r1", session_id="s1", group_id=0, num_prompt_tokens=4000)

    assert policy.release_session("s0") == 0
    # s0 is gone, and it no longer counts as a session not started.
    assert policy.ledger(0) == budget.running_session_blocks(4000)
    policy.end_call("r0", num_tokens=4500)  # its call's release arrives late
    assert policy.release_session("s0") is None

    policy.release_group(0)
    policy.end_call("r1", num_tokens=4500)
    assert policy.ledger(0) == 0


def test_a_retried_attempt_is_charged_on_its_groups_generator():
    budget = _budget()
    policy = _policy(budget)
    policy.add_group(0, num_sessions=1, validation=False)
    policy.place()
    policy.start_call("r0", session_id="attempt0", group_id=0, num_prompt_tokens=4000)
    policy.end_call("r0", num_tokens=4000)

    policy.start_call("r1", session_id="attempt1", group_id=0, num_prompt_tokens=4000)

    assert policy.ledger(0) == budget.idle_session_blocks(
        4000
    ) + budget.running_session_blocks(4000)


def test_a_waiting_group_can_be_released():
    policy = _policy()
    policy.add_group(0, num_sessions=16, validation=False)
    policy.add_group(1, num_sessions=16, validation=False)
    policy.place()

    policy.release_group(1)

    assert policy.num_waiting_groups == 0


def test_rejects_data_parallel_generators():
    budget = KVCacheBudget(
        num_blocks=3411,
        free_floor_blocks=381,
        attention_block_sizes=(1664,),
        num_mamba_groups=3,
        mamba_blocks_per_running_request=4,
        max_model_len=262144,
        data_parallel_size=2,
    )
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        _policy(budget)


def _idle_sessions(
    policy: KVRoomAdmission, sizes: list[int], *, group_id: int = 0, validation=False
) -> None:
    """Place a group with one idle session per size (in tokens), named f"g{group_id}/s{i}"."""
    policy.add_group(group_id, num_sessions=len(sizes), validation=validation)
    policy.place()
    for i, size in enumerate(sizes):
        request_id = f"g{group_id}/r{i}"
        policy.start_call(
            request_id,
            session_id=f"g{group_id}/s{i}",
            group_id=group_id,
            num_prompt_tokens=size,
        )
        policy.end_call(request_id, num_tokens=size)


# 21 idle sessions of 145 attention blocks each hold 21 x 148 = 3,108 > 3,030 usable blocks.
_OVER_H = [240_000 - i for i in range(21)]


def test_pause_drains_to_90_percent_then_resumes_the_longest_paused_up_to_95():
    policy = _policy()
    _idle_sessions(policy, _OVER_H)

    assert policy.rebalance(now=0.0) == []

    # Pause the 3 smallest (g0/s20, s19, s18): 3,108 - 3 x 148 = 2,664 <= 90% of 3,030. Then the
    # first-paused one fits back under 95% (2,878): 2,812.
    assert policy.num_paused_sessions == 2
    assert policy.ledger(0) == 2812
    assert policy.start_call(
        "next18", session_id="g0/s18", group_id=0, num_prompt_tokens=1
    )
    assert not policy.start_call(
        "next20", session_id="g0/s20", group_id=0, num_prompt_tokens=241_000
    )


def test_validation_sessions_pause_after_training_sessions():
    policy = _policy()
    _seed_first_prompt(policy)
    _idle_sessions(policy, [100_000] * 2, group_id=-1, validation=True)
    _idle_sessions(policy, [240_000] * 20)

    policy.rebalance(now=0.0)

    # 2 x 64 + 20 x 148 = 3,088: training sessions pause, though the validation ones are smaller.
    assert policy.num_paused_sessions == 2
    assert policy.start_call("v", session_id="g-1/s0", group_id=-1, num_prompt_tokens=1)


def test_running_sessions_pause_when_their_call_returns():
    budget = _budget()
    policy = _policy(budget)
    policy.add_group(0, num_sessions=19, validation=False)
    policy.place()
    for i in range(19):  # 19 running x 164 = 3,116, none idle
        policy.start_call(
            f"r{i}", session_id=f"s{i}", group_id=0, num_prompt_tokens=240_000 + i
        )

    policy.rebalance(now=0.0)
    assert policy.num_paused_sessions == 0

    # s0 was picked (smallest); it pauses as its call returns, and leaves the ledger.
    policy.end_call("r0", num_tokens=241_000)
    assert policy.num_paused_sessions == 1
    assert policy.ledger(0) == 18 * budget.running_session_blocks(240_018)


def test_resume_goes_longest_paused_first_and_runs_the_waiting_call():
    budget = _budget()
    policy = _policy(budget)
    _idle_sessions(policy, _OVER_H)
    policy.rebalance(now=0.0)  # g0/s19 and s20 stay paused, ledger 2,812
    assert not policy.start_call(
        "waiting", session_id="g0/s20", group_id=0, num_prompt_tokens=241_000
    )

    # Two releases free 296 blocks: 2,516 + 148 (s19) + 164 (s20, running) = 2,828 <= 2,878.
    policy.release_session("g0/s0")
    policy.release_session("g0/s1")

    assert policy.rebalance(now=10.0) == ["waiting"]
    assert policy.num_paused_sessions == 0
    assert policy.ledger(0) == 2516 + 148 + budget.running_session_blocks(241_000)


def test_a_generator_with_paused_sessions_takes_no_new_group():
    policy = _policy(_budget(), _budget())
    _idle_sessions(policy, _OVER_H)
    policy.rebalance(now=0.0)

    policy.add_group(1, num_sessions=1, validation=False)

    assert policy.place() == {1: 1}


def test_a_generator_with_only_paused_sessions_resumes_them_at_once():
    policy = _policy()
    _idle_sessions(policy, _OVER_H)
    policy.rebalance(now=0.0)
    for i in range(19):  # every unpaused session ends
        policy.release_session(f"g0/s{i}")

    policy.rebalance(now=1.0)

    assert policy.num_paused_sessions == 0


def _call_again(
    policy: KVRoomAdmission, session_ids: list[str], *, num_tokens: int
) -> None:
    for session_id in session_ids:
        policy.start_call(
            "again", session_id=session_id, group_id=0, num_prompt_tokens=num_tokens
        )
        policy.end_call("again", num_tokens=num_tokens)


def test_a_session_paused_past_the_bound_swaps_in_for_idle_sessions():
    policy = _policy()
    _idle_sessions(policy, _OVER_H)
    policy.rebalance(now=0.0)  # g0/s19 and s20 paused, ledger 2,812
    # g0/s0-s4 grow to 161 blocks: 2,877. A paused session (148) no longer fits under 2,878.
    _call_again(policy, [f"g0/s{i}" for i in range(5)], num_tokens=262_000)
    unpaused = [f"g0/s{i}" for i in range(19)]
    for now in (500.0, 1000.0, 1500.0):
        assert policy.rebalance(now=now) == []
        _call_again(
            policy, unpaused[5:], num_tokens=240_000
        )  # active, so not reclaimed as hung
        _call_again(policy, unpaused[:5], num_tokens=262_000)
    assert policy.num_paused_sessions == 2
    for session_id in ("g0/s19", "g0/s20"):  # their next calls wait
        assert not policy.start_call(
            session_id, session_id=session_id, group_id=0, num_prompt_tokens=240_000
        )

    # Past 1,800 s each swaps in for the smallest idle sessions, never going over 2,878:
    # 2,877 - 2 x 148 + 164 = 2,745 (s19 running), then 2,745 - 148 + 164 = 2,761 (s20).
    assert policy.rebalance(now=1801.0) == ["g0/s19", "g0/s20"]

    assert policy.num_swaps == 2
    assert policy.ledger(0) == 2761
    assert not policy.start_call(
        "s5", session_id="g0/s5", group_id=0, num_prompt_tokens=1
    )


def test_a_hung_session_pauses_while_a_group_waits():
    policy = _policy()
    _seed_first_prompt(policy)
    _idle_sessions(policy, [240_000] * 10)  # 1,480
    policy.add_group(1, num_sessions=80, validation=False)  # 80 x 22 = 1,760: waits
    assert policy.place() == {}
    policy.rebalance(now=0.0)
    for i in range(9):  # every session but g0/s9 runs again: 1,624, under H
        policy.start_call(
            f"again{i}", session_id=f"g0/s{i}", group_id=0, num_prompt_tokens=240_000
        )

    policy.rebalance(now=601.0)

    # It stays paused, not charged, until it calls: then it resumes when it fits.
    assert policy.num_paused_sessions == 1
    assert policy.ledger(0) == 9 * 164
    assert not policy.start_call(
        "back", session_id="g0/s9", group_id=0, num_prompt_tokens=240_000
    )
    assert policy.rebalance(now=602.0) == ["back"]


def test_a_cancelled_waiting_call_is_forgotten():
    policy = _policy()
    _idle_sessions(policy, _OVER_H)
    policy.rebalance(now=0.0)
    policy.start_call(
        "waiting", session_id="g0/s20", group_id=0, num_prompt_tokens=241_000
    )

    policy.cancel_call("waiting")
    policy.release_session("g0/s0")
    policy.release_session("g0/s1")

    assert policy.rebalance(now=10.0) == []
