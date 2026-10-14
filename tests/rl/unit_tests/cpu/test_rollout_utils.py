# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `TrainingSampleBuilder.rollout_to_training_samples` and `split_prompt`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import Rollout, RolloutGroup, RolloutStatus, RolloutTurn
from torchtitan.rl.rollout.types import RolloutTurnID, split_prompt

_GROUP_ID = "step=1/group=0"

# rollout_to_training_samples is the TrainingSampleBuilder override hook; the transform is config-independent.
rollout_to_training_samples = (
    TrainingSampleBuilder.Config().build().rollout_to_training_samples
)


def _turn(
    *,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    version: int,
    max_version: int | None = None,
    content: str = "x",
) -> RolloutTurn:
    return RolloutTurn(
        rollout_id=RolloutTurnID(group_id=_GROUP_ID, rollout_id=0, turn_id=0),
        # The full prompt for now; `_scored_rollout` stores it as a delta like the Rollouter does
        prompt_prefix_len=0,
        prompt_delta_token_ids=prompt_token_ids,
        completion_token_ids=np.array(completion_token_ids, dtype=np.int32),
        completion_logprobs=np.full(len(completion_token_ids), -0.1, dtype=np.float32),
        min_policy_version=version,
        max_policy_version=version if max_version is None else max_version,
        completion_message={"role": "assistant", "content": content},
    )


def _scored_rollout(
    turns: list[RolloutTurn], *, reward: float, advantage: float
) -> Rollout:
    previous_token_ids: list[int] = []
    for rollout_turn in turns:
        prompt_token_ids = rollout_turn.prompt_delta_token_ids
        prefix_len, delta = split_prompt(prompt_token_ids, previous_token_ids)
        rollout_turn.prompt_prefix_len = prefix_len
        rollout_turn.prompt_delta_token_ids = delta
        previous_token_ids = (
            prompt_token_ids + rollout_turn.completion_token_ids.tolist()
        )
    return Rollout(
        group_id=_GROUP_ID,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=turns,
        reward=reward,
        advantage=advantage,
    )


def test_single_turn_packs_one_training_sample() -> None:
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=2)],
        reward=1.0,
        advantage=0.5,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    assert training_sample.token_ids.tolist() == [1, 2, 4, 5]
    assert training_sample.loss_mask.tolist() == [False, False, True, True]
    assert training_sample.logprobs.tolist() == pytest.approx([0.0, 0.0, -0.1, -0.1])
    assert training_sample.min_policy_version == 2
    # advantage on the two completion tokens, 0.0 on the prompt
    assert training_sample.advantage.tolist() == pytest.approx([0.0, 0.0, 0.5, 0.5])
    assert training_sample.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    # single turn at version 2 -> min == max == 2
    assert training_sample.max_policy_version == 2


def test_packs_min_and_max_version_across_turns() -> None:
    # Two turns sampled at different versions (a weight pull landed between them); the packed
    # training_sample records the min (oldest, turn-0) and max (newest, turn-1) version.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=3),
            _turn(
                prompt_token_ids=[1, 2, 4, 5, 9], completion_token_ids=[7], version=4
            ),
        ],
        reward=1.0,
        advantage=0.5,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    assert training_sample.token_ids.tolist() == [1, 2, 4, 5, 9, 7]
    assert training_sample.min_policy_version == 3  # oldest (turn 0)
    assert training_sample.max_policy_version == 4  # newest (turn 1)


def test_multiturn_with_growing_prefix_packs_into_one_training_sample() -> None:
    # Each turn's prompt extends the prior turn's prompt+completion with the env reply
    # ([8], then [9]), so the whole trajectory packs into one training_sample.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=2),
            _turn(
                prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[5, 6], version=2
            ),
            _turn(
                prompt_token_ids=[1, 2, 4, 8, 5, 6, 9],
                completion_token_ids=[7],
                version=2,
            ),
        ],
        reward=0.8,
        advantage=-0.2,
    )
    [training_sample] = rollout_to_training_samples(rollout)
    #               P     P    a0    E1    a1    a1    E2    a2
    assert training_sample.token_ids.tolist() == [1, 2, 4, 8, 5, 6, 9, 7]
    assert training_sample.loss_mask.tolist() == [
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert training_sample.logprobs.tolist() == pytest.approx(
        [0.0, 0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1]
    )
    # advantage broadcast onto every assistant token, 0.0 on prompt/env tokens
    assert training_sample.advantage.tolist() == pytest.approx(
        [0.0, 0.0, -0.2, 0.0, -0.2, -0.2, 0.0, -0.2]
    )
    assert training_sample.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )


def test_history_edit_branches_into_separate_training_samples() -> None:
    # Turn 1's prompt does NOT extend turn 0's prompt+completion (the env rewrote history),
    # so the trajectory splits into two training_samples.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[90, 91], completion_token_ids=[5], version=1),
        ],
        reward=0.5,
        advantage=0.1,
    )
    first, second = rollout_to_training_samples(rollout)
    assert first.token_ids.tolist() == [1, 2, 4]
    assert first.loss_mask.tolist() == [False, False, True]
    # first segment opens at turn 0, the branch (history edit) opens at turn 1
    assert first.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=0
    )
    assert second.token_ids.tolist() == [90, 91, 5]
    assert second.loss_mask.tolist() == [False, False, True]
    assert second.rollout_id == RolloutTurnID(
        group_id=_GROUP_ID, rollout_id=0, turn_id=1
    )
    assert first.advantage.tolist() == pytest.approx([0.0, 0.0, 0.1])
    assert second.advantage.tolist() == pytest.approx([0.0, 0.0, 0.1])


def test_rewrite_that_keeps_a_prefix_branches_then_continues() -> None:
    # Turn 1 keeps [1, 2] but rewrites turn 0's completion [4] as [7] (e.g. thinking stripped), so it
    # stores only [7] and opens a new training_sample; turn 2 continues it with the env reply [9].
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 7], completion_token_ids=[5], version=1),
            _turn(
                prompt_token_ids=[1, 2, 7, 5, 9], completion_token_ids=[6], version=2
            ),
        ],
        reward=0.5,
        advantage=0.1,
    )
    stored = [
        (rollout_turn.prompt_prefix_len, rollout_turn.prompt_delta_token_ids.tolist())
        for rollout_turn in rollout.turns
    ]
    assert stored == [(0, [1, 2]), (2, [7]), (4, [9])]
    first, second = rollout_to_training_samples(rollout)
    assert first.token_ids.tolist() == [1, 2, 4]
    assert second.token_ids.tolist() == [1, 2, 7, 5, 9, 6]
    assert second.loss_mask.tolist() == [False, False, False, True, False, True]
    assert second.rollout_id.turn_id == 1
    assert (second.min_policy_version, second.max_policy_version) == (1, 2)


def test_routed_expert_ids_follow_the_trainer_inputs_across_turns_and_branches() -> None:
    # The same turns as above. Each turn's generator rows are tagged turn * 100 + position;
    # a turn keeps the rows from its prefix's last token on, like its prompt delta.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 7], completion_token_ids=[5], version=1),
            _turn(
                prompt_token_ids=[1, 2, 7, 5, 9], completion_token_ids=[6], version=2
            ),
        ],
        reward=0.5,
        advantage=0.1,
    )
    for turn_idx, rollout_turn in enumerate(rollout.turns):
        # One completion token: prompt + completion - 1 rows.
        num_rows = rollout_turn.num_prompt_tokens
        rows = np.arange(num_rows).reshape(-1, 1, 1) + 100 * turn_idx
        rollout_turn.routed_expert_ids = rows[
            max(rollout_turn.prompt_prefix_len - 1, 0) :
        ]

    first, second = rollout_to_training_samples(rollout)

    # [1, 2, 4] -> rows of inputs [1, 2]. [1, 2, 7, 5, 9, 6] -> rows of [1, 2, 7, 5, 9]: the branch
    # copies the shared prefix's row from the first sample, then takes each turn's delta.
    assert first.routed_expert_ids.flatten().tolist() == [0, 1]
    assert second.routed_expert_ids.flatten().tolist() == [0, 101, 102, 203, 204]


def _routed_group(*, group_id: int) -> RolloutGroup:
    """Two rollouts of two turns; the second turn continues the first, so its rows start at
    position 2. Rows are tagged turn * 100 + position."""
    rollouts = []
    for reward in [1.0, 0.0]:
        turns = [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[5], version=1),
        ]
        turns[0].routed_expert_ids = np.arange(2).reshape(-1, 1, 1)
        turns[1].routed_expert_ids = np.arange(102, 104).reshape(-1, 1, 1)
        rollouts.append(_scored_rollout(turns, reward=reward, advantage=reward - 0.5))
    return RolloutGroup(group_id=group_id, rollouts=rollouts)


@pytest.mark.parametrize(
    "unrouted_turn_ids",
    # Which turns of rollout 1 lack routed expert ids
    [
        pytest.param([0, 1], id="every_turn"),
        pytest.param([0], id="first_turn"),
        pytest.param([1], id="second_turn"),
    ],
)
def test_replay_drops_a_group_with_a_turn_without_routed_expert_ids(
    unrouted_turn_ids: list[int],
) -> None:
    builder = TrainingSampleBuilder.Config().build(replay_routed_experts=True)
    unrouted_group = _routed_group(group_id=1)
    for turn_id in unrouted_turn_ids:
        unrouted_group.rollouts[1].turns[turn_id].routed_expert_ids = None

    routed = builder.build_from_group(rollout_group=_routed_group(group_id=0))
    unrouted = builder.build_from_group(rollout_group=unrouted_group)

    assert [
        training_sample.routed_expert_ids.flatten().tolist()
        for training_sample in routed.training_samples
    ] == [[0, 1, 102, 103]] * 2
    assert unrouted.group_id == 1
    assert unrouted.training_samples == []
    aggregated = m.MetricsProcessor._aggregate_metrics(
        routed.metrics + unrouted.metrics
    )
    assert (
        aggregated[
            "training_sample_builder/num_groups_dropped_no_routed_expert_ids/sum"
        ]
        == 1.0
    )


def test_replay_keeps_a_group_whose_rollout_has_an_empty_later_turn() -> None:
    # A later turn aborted before its first token has no rows; only its rollout is dropped.
    builder = TrainingSampleBuilder.Config().build(replay_routed_experts=True)
    group = _routed_group(group_id=0)
    aborted = group.rollouts[1].turns[1]
    aborted.completion_token_ids = np.empty(0, dtype=np.int32)
    aborted.completion_logprobs = np.empty(0, dtype=np.float32)
    aborted.routed_expert_ids = None

    output = builder.build_from_group(rollout_group=group)

    assert len(output.training_samples) == 1
    aggregated = m.MetricsProcessor._aggregate_metrics(output.metrics)
    assert (
        "training_sample_builder/num_groups_dropped_no_routed_expert_ids/sum"
        not in (aggregated)
    )
    assert (
        aggregated["training_sample_builder/num_rollouts_dropped_empty_completion/sum"]
        == 1.0
    )


@pytest.mark.parametrize(
    "prompt_prefix_len, prompt_delta_token_ids",
    [
        # past the previous turn's end: would train on a truncated prompt
        (9, [8]),
        # a full prompt stored as the delta: would open a sample per turn
        (0, [1, 2, 4, 8]),
    ],
)
def test_prefix_that_is_not_the_longest_raises(
    prompt_prefix_len: int, prompt_delta_token_ids: list[int]
) -> None:
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    rollout.turns.append(
        RolloutTurn(
            rollout_id=RolloutTurnID(group_id=_GROUP_ID, rollout_id=0, turn_id=1),
            prompt_prefix_len=prompt_prefix_len,
            prompt_delta_token_ids=np.array(prompt_delta_token_ids, dtype=np.int32),
            completion_token_ids=np.array([5], dtype=np.int32),
            completion_logprobs=np.array([-0.1], dtype=np.float32),
            min_policy_version=1,
            max_policy_version=1,
        )
    )
    with pytest.raises(ValueError, match="split_prompt"):
        rollout_to_training_samples(rollout)


@pytest.mark.parametrize(
    "prompt, previous, expected",
    [
        ([1, 2, 4, 5, 9], [1, 2, 4, 5], (4, [9])),  # continues
        ([1, 2, 7], [1, 2, 4, 5], (2, [7])),  # history rewritten after [1, 2]
        ([1, 2], [1, 2, 4, 5], (2, [])),  # strict prefix of the previous tokens
        ([1, 2], [], (0, [1, 2])),  # first turn
        ([], [1, 2], (0, [])),  # empty prompt
    ],
)
def test_split_prompt(prompt, previous, expected) -> None:
    prompt_prefix_len, prompt_delta_token_ids = split_prompt(
        prompt, previous_token_ids=previous
    )
    assert (prompt_prefix_len, prompt_delta_token_ids.tolist()) == expected


def test_empty_completion_on_a_later_turn_raises() -> None:
    # An empty completion is only expected on the first turn (initial prompt too long). On any later
    # turn it is an anomaly, so building the training samples raises.
    rollout = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1),
        ],
        reward=0.0,
        advantage=0.0,
    )
    with pytest.raises(ValueError, match="no completion"):
        rollout_to_training_samples(rollout)


def test_build_from_group_drops_a_rollout_with_an_empty_later_completion() -> None:
    # e.g. turn 1 aborted before its first token: drop that rollout and count it, keep the others
    empty_later = _scored_rollout(
        [
            _turn(prompt_token_ids=[1, 2], completion_token_ids=[4], version=1),
            _turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1),
        ],
        reward=0.0,
        advantage=1.0,
    )
    kept = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2], completion_token_ids=[5], version=1)],
        reward=1.0,
        advantage=1.0,
    )
    kept.rollout_id = 1

    group = (
        TrainingSampleBuilder.Config()
        .build()
        .build_from_group(
            rollout_group=RolloutGroup(group_id=_GROUP_ID, rollouts=[empty_later, kept])
        )
    )

    assert [sample.rollout_id.rollout_id for sample in group.training_samples] == [1]
    [dropped] = [
        metric
        for metric in group.metrics
        if metric.key == "training_sample_builder/num_rollouts_dropped_empty_completion"
    ]
    assert dropped.value.value == 1.0


def test_empty_completion_on_first_turn_is_skipped() -> None:
    # First turn with no completion = initial prompt too long; expected, so the rollout yields no samples.
    rollout = _scored_rollout(
        [_turn(prompt_token_ids=[1, 2, 4, 8], completion_token_ids=[], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    assert rollout_to_training_samples(rollout) == []


def test_group_filters_preserve_group_id_for_acknowledgement() -> None:
    builder = TrainingSampleBuilder.Config().build()

    failed = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=7, rollouts=[])
    )

    empty_completion = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[], version=1)],
        reward=0.0,
        advantage=0.0,
    )
    empty_completion.group_id = 8
    untrainable = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=8, rollouts=[empty_completion])
    )

    first = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=1)],
        reward=1.0,
        advantage=0.0,
    )
    second = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[3], version=1)],
        reward=1.0,
        advantage=0.0,
    )
    first.group_id = second.group_id = 9
    zero_std = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=9, rollouts=[first, second])
    )

    invalid_logprob = _scored_rollout(
        [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=1)],
        reward=1.0,
        advantage=1.0,
    )
    invalid_logprob.group_id = 10
    invalid_logprob.turns[0].completion_logprobs = np.array([np.nan], dtype=np.float32)
    no_valid_tokens = builder.build_from_group(
        rollout_group=RolloutGroup(group_id=10, rollouts=[invalid_logprob])
    )

    filtered_groups = [failed, untrainable, zero_std, no_valid_tokens]
    assert [group.group_id for group in filtered_groups] == [7, 8, 9, 10]
    assert all(not group.training_samples for group in filtered_groups)


def test_topk_rows_align_with_token_ids_across_turns() -> None:
    first = _turn(prompt_token_ids=[1, 2], completion_token_ids=[4, 5], version=0)
    first.completion_topk_token_ids = torch.tensor([[4, 9], [5, 8]], dtype=torch.int32)
    first.completion_topk_logprobs = torch.tensor([[-0.1, -2.0], [-0.2, -3.0]])
    second = _turn(
        prompt_token_ids=[1, 2, 4, 5, 9], completion_token_ids=[7], version=0
    )
    second.completion_topk_token_ids = torch.tensor([[7, 3]], dtype=torch.int32)
    second.completion_topk_logprobs = torch.tensor([[-0.3, -1.5]])

    [training_sample] = rollout_to_training_samples(
        _scored_rollout([first, second], reward=1.0, advantage=0.5)
    )

    # Zero rows on the prompt and the env reply (token 9), like their 0.0 logprobs.
    assert training_sample.token_ids.tolist() == [1, 2, 4, 5, 9, 7]
    assert training_sample.topk_token_ids.tolist() == [
        [0, 0],
        [0, 0],
        [4, 9],
        [5, 8],
        [0, 0],
        [7, 3],
    ]
    torch.testing.assert_close(
        training_sample.topk_logprobs,
        torch.tensor(
            [[0, 0], [0, 0], [-0.1, -2.0], [-0.2, -3.0], [0, 0], [-0.3, -1.5]]
        ),
    )


def test_zero_std_groups_split_into_all_success_and_all_failure() -> None:
    # Three groups: all-1 is all_success, all-0 is all_failure, mixed is neither.
    builder = TrainingSampleBuilder.Config().build()
    metrics: list[m.Metric] = []
    for group_id, rewards in enumerate([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]):
        rollouts = [
            _scored_rollout(
                [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=0)],
                reward=reward,
                advantage=0.0,
            )
            for reward in rewards
        ]
        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        metrics += builder.build_from_group(rollout_group=group).metrics
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    for prefix in [
        "rollout_reward/group_zero_std_frac",
        # Without a length reward, the graded reward is the reward.
        "rollout_reward/task_zero_std_frac",
    ]:
        assert aggregated[f"{prefix}/mean"] == pytest.approx(2 / 3)
        assert aggregated[f"{prefix}/all_success/mean"] == pytest.approx(1 / 3)
        assert aggregated[f"{prefix}/all_failure/mean"] == pytest.approx(1 / 3)


def test_task_zero_std_split_reads_the_graded_reward() -> None:
    # The length reward makes an all-solved and an all-failed group non-zero-std; the task split
    # still finds them by the graded reward, and every group still trains.
    builder = TrainingSampleBuilder.Config().build()
    metrics: list[m.Metric] = []
    num_training_samples = []
    groups = [
        # (final reward, length reward)
        [(1.05, 0.05), (0.95, -0.05)],
        [(0.0, 0.0), (-0.05, -0.05)],
        [(1.05, 0.05), (-0.05, -0.05)],
    ]
    for group_id, rewards in enumerate(groups):
        rollouts = [
            _scored_rollout(
                [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=0)],
                reward=reward,
                advantage=0.0,
            )
            for reward, _ in rewards
        ]
        for rollout, (_, length_reward) in zip(rollouts, rewards, strict=True):
            rollout.reward_breakdown = {"length_reward": length_reward}
        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        output = builder.build_from_group(rollout_group=group)
        metrics += output.metrics
        num_training_samples.append(len(output.training_samples))
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert aggregated["rollout_reward/group_zero_std_frac/mean"] == 0.0
    prefix = "rollout_reward/task_zero_std_frac"
    assert aggregated[f"{prefix}/mean"] == pytest.approx(2 / 3)
    assert aggregated[f"{prefix}/all_success/mean"] == pytest.approx(1 / 3)
    assert aggregated[f"{prefix}/all_failure/mean"] == pytest.approx(1 / 3)
    assert num_training_samples == [2, 2, 2]


def test_drop_unsolved_groups_by_keeps_only_groups_with_a_solve() -> None:
    # Partial credit spreads the unsolved group's rewards, so the zero-std drop keeps it.
    builder = TrainingSampleBuilder.Config(
        drop_unsolved_groups_by="RewardFromVerifiers"
    ).build()
    groups = [
        # (reward, RewardFromVerifiers score)
        [(0.1, 0.0), (0.2, 0.0)],  # no solve -> dropped
        [(1.0, 1.0), (0.2, 0.0)],  # one solve -> trains
    ]
    metrics: list[m.Metric] = []
    num_training_samples = []
    for group_id, scores in enumerate(groups):
        rollouts = [
            _scored_rollout(
                [_turn(prompt_token_ids=[1], completion_token_ids=[2], version=0)],
                reward=reward,
                advantage=0.0,
            )
            for reward, _ in scores
        ]
        for rollout, (_, solved) in zip(rollouts, scores, strict=True):
            rollout.reward_breakdown = {"RewardFromVerifiers": solved}
        output = builder.build_from_group(
            rollout_group=RolloutGroup(group_id=group_id, rollouts=rollouts)
        )
        metrics += output.metrics
        num_training_samples.append(len(output.training_samples))
    aggregated = m.MetricsProcessor._aggregate_metrics(metrics)
    assert num_training_samples == [0, 2]
    assert aggregated["training_sample_builder/num_groups_dropped_unsolved/sum"] == 1.0
