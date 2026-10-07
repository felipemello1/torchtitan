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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from datasets import Dataset

from torchtitan.rl.examples.dapo_math import (
    AIME2025Dataset,
    DapoMathDataset,
    DapoMathEnv,
    DapoMathSample,
    data as math_data,
    MathEvalBenchmark,
    MathEvalDataset,
    PerBenchmarkRubric,
    RewardMathVerify,
    rubric as math_rubric,
    score_math_response,
)
from torchtitan.rl.observability.controller import compute_rollout_metrics
from torchtitan.rl.observability.metrics import MetricsProcessor
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


def test_math_eval_dataset_reads_benchmarks_in_order(monkeypatch) -> None:
    rows_by_repo_id = {
        "MathArena/aime_2026": [
            {"problem": "aime problem 1", "answer": 277},
            {"problem": "aime problem 2", "answer": 62},
        ],
        "MathArena/hmmt_feb_2026": [
            {"problem": "hmmt problem 1", "answer": r"-\frac{1}{21}"}
        ],
    }
    load_calls = []

    def load_dataset(repo_id, *, split, revision):
        load_calls.append((repo_id, split, revision))
        return Dataset.from_list(rows_by_repo_id[repo_id])

    monkeypatch.setattr(math_data, "load_dataset", load_dataset)
    dataset = MathEvalDataset.Config(
        benchmarks=(
            MathEvalBenchmark(
                repo_id="MathArena/aime_2026", revision="rev1", tier="core"
            ),
            MathEvalBenchmark(
                repo_id="MathArena/hmmt_feb_2026",
                revision="rev2",
                tier="hard",
                split="test",
            ),
        )
    ).build()
    samples = [next(dataset) for _ in range(3)]

    assert load_calls == [
        ("MathArena/aime_2026", "train", "rev1"),
        ("MathArena/hmmt_feb_2026", "test", "rev2"),
    ]
    assert [
        (sample.tier, sample.benchmark, sample.ground_truth) for sample in samples
    ] == [
        ("core", "aime_2026", "277"),
        ("core", "aime_2026", "62"),
        ("hard", "hmmt_feb_2026", r"-\frac{1}{21}"),
    ]
    assert "aime problem 1" in samples[0].prompt
    assert all(r"Answer: \boxed{" in sample.prompt for sample in samples)
    # The next pass starts over in the same order.
    assert next(dataset) == samples[0]


def test_math_eval_suite_has_240_problems_with_their_gold() -> None:
    """Load the pinned suite from the Hub; every gold must verify against itself."""
    try:
        dataset = MathEvalDataset.Config().build()
    except ConnectionError as exc:  # offline; a wrong revision still fails
        pytest.skip(f"math eval datasets unavailable: {exc}")
    samples = [next(dataset) for _ in range(240)]
    assert Counter(sample.benchmark for sample in samples) == {
        "aime_2026": 30,
        "hmmt_feb_2026": 33,
        "hmmt_nov_2025": 30,
        "BeyondAIME": 100,
        "apex-shortlist": 47,
    }
    # Hand-checked gold: AIME 2026 problem 1 and HMMT Nov 2025 problem 20.
    hmmt_nov_golds = [
        sample.ground_truth for sample in samples if sample.benchmark == "hmmt_nov_2025"
    ]
    assert samples[0].ground_truth == "277"
    assert hmmt_nov_golds[19] == r"\frac{\sqrt{7} + 1}{2}"
    for sample in samples:
        response = f"Answer: \\boxed{{{sample.ground_truth}}}"
        assert score_math_response(response, sample.ground_truth) == 1.0, sample


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


def test_per_benchmark_rubric_logs_reward_under_benchmark_and_tier() -> None:
    rubric = PerBenchmarkRubric.Config(reward_fns=[RewardMathVerify.Config()]).build()
    rollouts = [_rollout(r"Answer: \boxed{34}"), _rollout(r"Answer: \boxed{35}")]
    sample = DapoMathSample(
        prompt="problem", ground_truth="34", benchmark="aime_2026", tier="core"
    )
    outputs = asyncio.run(rubric.score_group(rollouts, sample))
    assert [output.reward_breakdown for output in outputs] == [
        {"RewardMathVerify": 1.0, "aime_2026": 1.0, "core": 1.0},
        {"RewardMathVerify": 0.0, "aime_2026": 0.0, "core": 0.0},
    ]

    for rollout, output in zip(rollouts, outputs, strict=True):
        rollout.reward = output.reward
        rollout.reward_breakdown = output.reward_breakdown
    metrics = MetricsProcessor._aggregate_metrics(
        compute_rollout_metrics(prefix="validation", rollouts=rollouts)
    )
    assert metrics["validation_reward/component/aime_2026/mean"] == 0.5
    assert metrics["validation_reward/component/core/mean"] == 0.5

    untagged_sample = DapoMathSample(prompt="problem", ground_truth="34")
    untagged = asyncio.run(rubric.score_group(rollouts[:1], untagged_sample))
    assert untagged[0].reward_breakdown == {"RewardMathVerify": 1.0}
