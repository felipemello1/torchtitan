# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence

from renderers import Message

from torchtitan.experiments.rl.env_types import MessageEnv, ResetOutput, StepOutput
from torchtitan.experiments.rl.tasks.alphabet_sort.data import AlphabetSortInput


class AlphabetSortEnv(MessageEnv):
    """Serves one AlphabetSort episode, one turn at a time.

    `reset` poses the first turn (and restarts the episode); each `step_message`
    poses the next follow-up turn, and the episode ends once every turn has been
    answered. The env doesn't score — the rubric does, afterward.
    """

    def __init__(self, *, env_input: AlphabetSortInput) -> None:
        self._input = env_input

    async def reset(self) -> ResetOutput:
        self._follow_up_names = iter(self._input.presented_names[1:])
        prompt = _first_turn_prompt(
            self._input.presented_names[0], self._input.sort_by_first
        )
        return ResetOutput(messages=[{"role": "user", "content": prompt}])

    async def step_message(self, msg: Message) -> StepOutput:
        names = next(self._follow_up_names, None)
        if names is None:  # every turn answered
            return StepOutput(done=True)
        prompt = _follow_up_prompt(names, self._input.sort_by_first)
        return StepOutput(messages=[{"role": "user", "content": prompt}])


def _first_turn_prompt(names: Sequence[str], sort_by_first: bool) -> str:
    sort_key = "FIRST" if sort_by_first else "LAST"
    return (
        f"Sort these names in alphabetical order by {sort_key} name: "
        f"{', '.join(names)}\n\n"
        "Use exactly this format:\n"
        "<alphabetical_sorted>\n"
        "Name1\n"
        "Name2\n"
        "</alphabetical_sorted>"
    )


def _follow_up_prompt(names: Sequence[str], sort_by_first: bool) -> str:
    sort_key = "FIRST" if sort_by_first else "LAST"
    return (
        f"Now sort ALL of these names alphabetically by {sort_key} name: "
        f"{', '.join(names)}\n\n"
        "These are in addition to the prior list. Mark any NEW names "
        "(that weren't in the prior list) with `// new name!` at the end.\n\n"
        "Use exactly this format:\n"
        "<combined_alphabetical_sorted>\n"
        "Name1\n"
        "Name2 // new name!\n"
        "</combined_alphabetical_sorted>"
    )
