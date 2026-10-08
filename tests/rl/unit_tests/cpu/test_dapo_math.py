# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DAPO-Math dataset, environment, and rubric."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from datasets import Dataset

from torchtitan.rl.examples.dapo_math import (
    AIME2025Dataset,
    DapoMathDataset,
    DapoMathEnv,
    DapoMathSample,
    data as math_data,
    Intellect3MathDataset,
    RewardMathVerify,
    rubric as math_rubric,
    score_math_response,
)
from torchtitan.rl.rollout import Rollout, RolloutStatus, RolloutTurn
from torchtitan.rl.types import RolloutTurnID


def _dapo_rows() -> list[dict]:
    return [
        {
            "source_prompt": [{"role": "user", "content": "problem 1"}],
            "prompt": "problem 1",
            "ground_truth": "34",
        },
        {
            "source_prompt": [{"role": "user", "content": "problem 2"}],
            "prompt": "problem 2",
            "ground_truth": "113",
        },
        {
            "source_prompt": [{"role": "user", "content": "problem 3"}],
            "prompt": "problem 3",
            "ground_truth": "7",
        },
    ]


def test_dapo_dataset_is_deterministic_and_resumable(monkeypatch) -> None:
    monkeypatch.setattr(math_data, "load_dataset", lambda *args, **kwargs: _dapo_rows())
    config = DapoMathDataset.Config(seed=7)
    first = config.build()
    second = config.build()
    assert [next(first) for _ in range(3)] == [next(second) for _ in range(3)]

    checkpoint = first.state_dict()
    expected = [next(first) for _ in range(3)]
    resumed = config.build()
    resumed.load_state_dict(checkpoint)
    assert [next(resumed) for _ in range(3)] == expected
    assert all(r"Answer: \boxed{" in sample.prompt for sample in expected)


def _cycling_dataset(
    num_rows: int, *, skip_solved_prompts: bool
) -> math_data._CyclingDataset:
    samples = [
        DapoMathSample(prompt=f"q{i}", ground_truth=str(i)) for i in range(num_rows)
    ]
    return math_data._CyclingDataset(
        samples, seed=0, shuffle=False, skip_solved_prompts=skip_solved_prompts
    )


def test_dapo_dataset_skips_solved_problems_after_resume(monkeypatch) -> None:
    monkeypatch.setattr(math_data, "load_dataset", lambda *args, **kwargs: _dapo_rows())
    config = DapoMathDataset.Config(shuffle=False, skip_solved_prompts=True)
    dataset = config.build()
    first_problem, _, _ = [next(dataset) for _ in range(3)]
    dataset.mark_solved(first_problem)

    resumed = config.build()
    resumed.load_state_dict(dataset.state_dict())
    assert [next(resumed).ground_truth for _ in range(4)] == ["113", "7", "113", "7"]


def test_cycling_dataset_loads_a_state_saved_before_solved_rows() -> None:
    state = _cycling_dataset(3, skip_solved_prompts=True).state_dict()
    del state["solved_rows"]
    resumed = _cycling_dataset(3, skip_solved_prompts=True)
    resumed.load_state_dict(state)
    assert [next(resumed).ground_truth for _ in range(3)] == ["0", "1", "2"]


def test_cycling_dataset_draws_solved_rows_again_with_the_flag_off() -> None:
    dataset = _cycling_dataset(2, skip_solved_prompts=True)
    dataset.mark_solved(next(dataset))
    resumed = _cycling_dataset(2, skip_solved_prompts=False)
    resumed.load_state_dict(dataset.state_dict())
    resumed.mark_solved(DapoMathSample(prompt="q1", ground_truth="1"))
    assert [next(resumed).ground_truth for _ in range(3)] == ["1", "0", "1"]


def test_cycling_dataset_ignores_a_sample_that_is_no_longer_a_row() -> None:
    # A pending sample saved before its row's gold changed is replayed after the resume.
    dataset = _cycling_dataset(2, skip_solved_prompts=True)
    dataset.mark_solved(DapoMathSample(prompt="q0", ground_truth="changed"))
    assert [next(dataset).ground_truth for _ in range(2)] == ["0", "1"]


def test_cycling_dataset_raises_once_every_row_is_solved() -> None:
    dataset = _cycling_dataset(2, skip_solved_prompts=True)
    for _ in range(2):
        dataset.mark_solved(next(dataset))
    with pytest.raises(RuntimeError, match="every row is solved"):
        next(dataset)


def test_aime_dataset_combines_both_subsets(monkeypatch) -> None:
    def load_dataset(repo_id, subset, *, split):
        del repo_id, split
        answer = r"42^\circ" if subset == "AIME2025-I" else r"\boxed{42}"
        return Dataset.from_list([{"question": f"{subset} question", "answer": answer}])

    monkeypatch.setattr(math_data, "load_dataset", load_dataset)
    dataset = AIME2025Dataset.Config(num_samples=2).build()
    samples = [next(dataset), next(dataset)]
    assert [sample.ground_truth for sample in samples] == [r"42^\circ", r"\boxed{42}"]
    assert "AIME2025-I question" in samples[0].prompt
    assert "AIME2025-II question" in samples[1].prompt
    assert all(r"Answer: \boxed{" in sample.prompt for sample in samples)


def test_aime_dataset_restarts_after_configured_num_samples(monkeypatch) -> None:
    def load_dataset(repo_id, subset, *, split):
        del repo_id, split
        return Dataset.from_list([{"question": f"{subset} question", "answer": "42"}])

    monkeypatch.setattr(math_data, "load_dataset", load_dataset)
    dataset = AIME2025Dataset.Config(num_samples=1).build()
    first = next(dataset)
    assert next(dataset) == first


def test_intellect3_dataset_keeps_partially_solved_problems(monkeypatch) -> None:
    pass_rates = [0.0, 0.125, 0.875, 1.0]

    def load_dataset(repo_id, subset, *, split):
        assert (repo_id, subset, split) == (
            "PrimeIntellect/INTELLECT-3-RL",
            "math",
            "train",
        )
        return Dataset.from_list(
            [
                {
                    "question": f"q{i}",
                    "answer": rf"{i}\sqrt{{2}}",
                    "avg@8_qwen3_4b_thinking_2507": rate,
                }
                for i, rate in enumerate(pass_rates)
            ]
        )

    monkeypatch.setattr(math_data, "load_dataset", load_dataset)
    dataset = Intellect3MathDataset.Config(shuffle=False).build()
    samples = [next(dataset) for _ in range(3)]
    # Inclusive bounds keep 1/8 and 7/8 and drop 0/8 and 8/8, so the third sample wraps to q1.
    assert [sample.ground_truth for sample in samples] == [
        r"1\sqrt{2}",
        r"2\sqrt{2}",
        r"1\sqrt{2}",
    ]
    assert "q1" in samples[0].prompt
    assert r"Answer: \boxed{" in samples[0].prompt


def _intellect3_dataset(
    monkeypatch, rows: list[tuple[str, str]]
) -> Intellect3MathDataset:
    def load_dataset(repo_id, subset, *, split):
        del repo_id, subset, split
        return Dataset.from_list(
            [
                {
                    "question": question,
                    "answer": answer,
                    "avg@8_qwen3_4b_thinking_2507": 0.5,
                }
                for question, answer in rows
            ]
        )

    monkeypatch.setattr(math_data, "load_dataset", load_dataset)
    return Intellect3MathDataset.Config(shuffle=False).build()


def test_intellect3_dataset_drops_rows_that_mis_score_answers(
    monkeypatch,
) -> None:
    # Shortened INTELLECT-3-RL rows. Dropped: a boxed `C` scores 0 against gold `135`, ` n `
    # only names the variable, and `0, -6` scores 0 against `-6`. Kept (the last seven): they
    # only look like option lists or two-value questions, or their gold holds both values.
    rows = [
        (r"What is $\tfrac1A+\tfrac1B$? A) 133 B) 134 C) 135 D) 136 E) 137", "135"),
        ("Find the volume.\n- **A)** $1024$\n- **B)** $1200$\n- **C)** $1280$", "1280"),
        (r"How many? $\textbf{a)}\ 0 \qquad\textbf{b)}\ 1 \qquad\textbf{c)}\ 2$", "2"),
        (r"For which positive integers $n$ is $x^n+(x+1)^n$ an integer?", " n "),
        (r"Find $f(-2)$ and $f(4)$.", "-6"),
        (r"Find the maximum and minimum values of $x^3-12x$ on $[-3,5]$.", "-16"),
        ("Menchikov A.B.  Find all pairs of natural numbers a and k.", "1,k"),
        (r"In triangle ABC, $\angle A: \angle B: \angle C=2: 3: 4$. Find AC.", "26"),
        (r"In triangle ABC, find $\cos(3A)+\cos(3B)+\cos(3C)$.", "1"),
        (r"Find the values of $m$ and $n$.", "m=6, n=9"),
        (r"Find the product of the maximum and minimum values of $x+y$.", "4"),
        (r"Find \(CD\) if \(BC = 7\) and \(AC = 24\).", r"8\sqrt{7}"),
        (r"(I) Find the values of $a$ and $m$; (II) Find the sum of all zeros.", "6"),
    ]
    dataset = _intellect3_dataset(monkeypatch, rows)
    # Only the last seven survive, so the eighth sample wraps to the first.
    golds = [next(dataset).ground_truth for _ in range(8)]
    assert golds == ["1,k", "26", "1", "m=6, n=9", "4", r"8\sqrt{7}", "6", "1,k"]


def test_cycling_dataset_rejects_a_state_saved_with_another_row_count(
    monkeypatch,
) -> None:
    monkeypatch.setattr(math_data, "load_dataset", lambda *args, **kwargs: _dapo_rows())
    state = DapoMathDataset.Config().build().state_dict()
    monkeypatch.setattr(
        math_data, "load_dataset", lambda *args, **kwargs: _dapo_rows()[:2]
    )
    with pytest.raises(ValueError, match="covers 3 rows, but the dataset has 2"):
        DapoMathDataset.Config().build().load_state_dict(state)


def test_intellect3_dataset_unescapes_golds(monkeypatch) -> None:
    # Real golds, stored JSON-escaped: Math-Verify reads `0 \\text{ or } 5` as 5.
    rows = [
        ("q0", r"0 \\text{ or } 5"),
        ("q1", r"\\{1/2, 2\\}"),
        ("q2", r"\\frac{1}{2}"),
        ("q3", r"\sqrt{2}"),
    ]
    dataset = _intellect3_dataset(monkeypatch, rows)
    golds = [next(dataset).ground_truth for _ in range(4)]
    assert golds == [r"0 \text{ or } 5", r"\{1/2, 2\}", r"\frac{1}{2}", r"\sqrt{2}"]
    assert score_math_response(r"\boxed{0 \text{ or } 5}", golds[0]) == 1.0
    assert score_math_response(r"\boxed{5}", golds[0]) == 0.0
    assert score_math_response(r"\boxed{\{\frac12, 2\}}", golds[1]) == 1.0


def test_env_is_single_turn() -> None:
    env = DapoMathEnv.Config().build(
        env_input=DapoMathSample(prompt="solve me", ground_truth="3"),
    )
    initial = asyncio.run(env.init())
    assert initial.init_prompt_messages == [{"role": "user", "content": "solve me"}]
    assert asyncio.run(env.step({"role": "assistant", "content": "Answer: 3"})).done


def _rollout(response: str) -> Rollout:
    return Rollout(
        group_id=0,
        rollout_id=0,
        status=RolloutStatus.COMPLETED,
        turns=[
            RolloutTurn(
                rollout_id=RolloutTurnID(group_id=0, rollout_id=0, turn_id=0),
                prompt_token_ids=[1],
                completion_token_ids=[2],
                completion_logprobs=[-0.1],
                completion_message={"role": "assistant", "content": response},
            )
        ],
    )


def test_math_verifier_requires_a_boxed_answer() -> None:
    assert score_math_response(r"work\nAnswer: \boxed{34}", "34") == 1.0
    assert score_math_response(r"work\n\boxed{\frac{68}{2}}", "34") == 1.0
    assert score_math_response("work\nAnswer: $34$", "34") == 0.0
    assert score_math_response("work\nAnswer: 34", "34") == 0.0
    assert score_math_response("work mentions 34", "34") == 0.0


def test_math_verifier_parses_non_integer_gold_answers() -> None:
    assert score_math_response(r"\boxed{2\sqrt{3}}", r"2\sqrt{3}") == 1.0
    assert score_math_response(r"\boxed{2}", r"2\sqrt{3}") == 0.0
    assert score_math_response(r"\boxed{(1,2)}", "(1,2)") == 1.0
    assert score_math_response(r"\boxed{\frac{\pi}{4}}", r"\pi/4") == 1.0


def test_math_verifier_uses_the_last_boxed_answer() -> None:
    response = r"Work: \boxed{2003^{2002^{2001}}}" "\n" r"Answer: \boxed{34}"
    assert score_math_response(response, "34") == 1.0


def test_math_verifier_rejects_unboxed_large_intermediate_expression() -> None:
    response = r"Work: \[2003^{2002^{2001}}\]" "\n" r"Final: \[Answer: 009\]"
    assert score_math_response(response, "241") == 0.0


def test_math_verifier_works_in_rollout_worker_thread() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(score_math_response, r"work\nAnswer: \boxed{34}", "34")
        assert result.result() == 1.0


def test_math_verifier_times_out_in_rollout_worker_thread(monkeypatch, caplog) -> None:
    verify_called = False
    caplog.set_level(logging.WARNING, logger=math_rubric.__name__)

    def busy_verify(*args, **kwargs) -> bool:
        nonlocal verify_called
        del args, kwargs
        verify_called = True
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pass
        return True

    monkeypatch.setattr(math_rubric, "parse", lambda *args, **kwargs: [34])
    monkeypatch.setattr(math_rubric, "verify", busy_verify)
    monkeypatch.setattr(math_rubric, "_MATH_VERIFY_TIMEOUT_SECONDS", 0.01)

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(score_math_response, r"work\nAnswer: \boxed{34}", "34")
        assert result.result(timeout=1) == 0.0

    assert verify_called
    assert "Math-Verify timed out after 0.01 seconds" in caplog.text


def test_reward_handles_equivalent_latex_and_units() -> None:
    reward = RewardMathVerify.Config().build()
    sample = DapoMathSample(prompt="problem", ground_truth=r"336^\circ")
    assert asyncio.run(reward(_rollout(r"work\nAnswer: \boxed{336}"), sample)) == 1.0
    assert asyncio.run(reward(_rollout(r"work\nAnswer: \boxed{335}"), sample)) == 0.0
