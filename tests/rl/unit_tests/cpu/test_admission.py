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
