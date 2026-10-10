# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the Terminal-Bench Verifiers recipe."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.config import ConfigLoader
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.rl.controller import Controller, ValidationConfig
from torchtitan.rl.examples.verifiers.data import (
    VerifiersTaskDataset,
    VerifiersTaskSample,
)
from torchtitan.rl.examples.verifiers.generation_server import (
    VerifiersGenerationMetadata,
)
from torchtitan.rl.examples.verifiers.terminal_bench import taskset
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    TerminalBenchTerminusHarness,
    TerminalBenchTerminusHarnessConfig,
)
from torchtitan.rl.generator import SamplingConfig
from torchtitan.rl.rollout import RolloutStatus
from torchtitan_recipes.rl.verifiers_terminal_bench import (
    _terminal_bench_rollouter_config,
)
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.serve import env_config_data
from verifiers.v1.tasksets.harbor import HarborEnvConfig
from verifiers.v1.utils.loaders import load_harness, resolve_env_config

TRAIN_DATASET = "local/tmax@v1"
EVAL_DATASET = "terminal-bench/terminal-bench-2-1"
MAX_CONTEXT_LENGTH = 32768
MAX_TURNS = 64
MAX_CONCURRENT_ROLLOUTS = 64


def _rollouter_config(train_dataset: str, validation_dataset: str):
    return _terminal_bench_rollouter_config(
        train_dataset,
        validation_dataset,
        max_context_length=MAX_CONTEXT_LENGTH,
        max_turns=MAX_TURNS,
        max_concurrent_rollouts=MAX_CONCURRENT_ROLLOUTS,
    )


def test_prompt_at_the_context_cap_never_reaches_the_generator() -> None:
    """A prompt of `max_context_length` tokens fails Verifiers' pre-flight check, which stops the
    rollout as `context_length`; vLLM would reject it in the engine loop (no room to sample)."""
    from openai import AsyncOpenAI, InternalServerError
    from renderers import OverlongPromptError
    from renderers.client import generate

    prompt_lengths = []

    async def generate_fn(prompt_token_ids, **kwargs):
        prompt_lengths.append(len(prompt_token_ids))
        raise RuntimeError("generator reached")

    async def send(client, server, num_tokens: int) -> None:
        await generate(
            client=client,
            renderer=SimpleNamespace(get_stop_token_ids=lambda: [0]),
            messages=[],
            model=server.model_id,
            prompt_ids=[1] * num_tokens,
            sampling_params={"torchtitan_group_id": 0},
            extra_headers={"X-Session-ID": "group=0/rollout=0"},
        )

    async def run_test() -> None:
        config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET).generation_server
        server = config.build()
        server.generate_fns[0] = generate_fn
        await server.start()
        try:
            client = AsyncOpenAI(
                base_url=server.base_url, api_key="unused", max_retries=0
            )
            with pytest.raises(OverlongPromptError):
                await send(client, server, MAX_CONTEXT_LENGTH)
            with pytest.raises(InternalServerError, match="generator reached"):
                await send(client, server, MAX_CONTEXT_LENGTH - 1)
        finally:
            await server.close()

    asyncio.run(run_test())
    assert prompt_lengths == [MAX_CONTEXT_LENGTH - 1]


class _RecordingRuntime:
    """Records the program the harness prepares, how it runs it, and other commands."""

    def __init__(self) -> None:
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.files: dict[str, bytes] = {}

    async def prepare_uv_script(self, source: str, env: dict[str, str]) -> list[str]:
        self.source = source
        return ["terminus_harness"]

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.argv = argv
        self.program_env = env
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def run(self, argv: list[str], env: dict[str, str]) -> None:
        self.commands.append((argv, env))

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data


def _run_terminus_program(
    tmp_path, monkeypatch, system_prompt: str | None = None, **options: bool
) -> SimpleNamespace:
    """Launch the harness with ``options``, then run the program it prepared with a
    recording stand-in for Harbor's ``Terminus2``.

    Returns the program's argv, the keyword arguments it passes to ``Terminus2``
    (``kwargs``), the agent it built (``terminus``), and the calls it makes on the
    agent, in order (``calls``).
    """
    harbor_terminus = pytest.importorskip("harbor.agents.terminus_2")
    from harbor.models.trial.paths import EnvironmentPaths

    harness_config = _rollouter_config(
        TRAIN_DATASET, EVAL_DATASET
    ).verifiers_env_server.environment.agent.harness
    harness = load_harness(harness_config.model_copy(update=options))
    runtime = _RecordingRuntime()
    asyncio.run(
        harness.launch(
            SimpleNamespace(model="torchtitan"),
            SimpleNamespace(id="trace"),
            runtime,
            "http://127.0.0.1:1/v1",
            "secret",
            {},
            SimpleNamespace(
                prompt="Fix the parser in /app.", system_prompt=system_prompt
            ),
        )
    )
    assert runtime.source.count('"harbor==0.22.0"') == 1
    # Run the program on a local copy of the prompts file the harness wrote.
    [(prompts_path, prompts)] = runtime.files.items()
    local_prompts = tmp_path / "prompts.json"
    local_prompts.write_bytes(prompts)
    argv = [arg.replace(prompts_path, str(local_prompts)) for arg in runtime.argv]

    recorded = SimpleNamespace(argv=runtime.argv, kwargs={}, calls=[])

    async def llm_call(*args: object, message_history: list[dict]) -> list[dict]:
        return message_history

    class RecordingSession:
        async def send_keys(
            self, keys: list[str], min_timeout_sec: float = 0.0
        ) -> None:
            recorded.calls.append(("send_keys", keys, min_timeout_sec))

    class RecordingTerminus2:
        def __init__(self, **kwargs: object) -> None:
            recorded.kwargs.update(kwargs)
            recorded.terminus = self
            self._llm = SimpleNamespace(call=llm_call)
            self._session = RecordingSession()

        async def setup(self, environment: object) -> None:
            recorded.calls.append(("setup",))

        async def run(self, instruction: str, *args: object) -> None:
            recorded.calls.append(("run", instruction))

    monkeypatch.setattr(harbor_terminus, "Terminus2", RecordingTerminus2)
    monkeypatch.setattr(EnvironmentPaths, "agent_dir", PurePosixPath("/"))
    monkeypatch.setenv("TMUX_TMPDIR", str(tmp_path / "tmux"))
    monkeypatch.setattr(sys, "argv", argv)
    program: dict[str, object] = {"__name__": "terminus_harness"}
    exec(compile(runtime.source, "terminus_harness.py", "exec"), program)
    asyncio.run(program["main"]())
    return recorded


def test_harness_options_reach_terminus2(tmp_path, monkeypatch) -> None:
    defaults = _run_terminus_program(tmp_path, monkeypatch).kwargs
    assert defaults["interleaved_thinking"] is True
    assert defaults["enable_summarize"] is False

    flipped = _run_terminus_program(
        tmp_path, monkeypatch, interleaved_thinking=False, enable_summarize=True
    ).kwargs
    assert flipped["interleaved_thinking"] is False
    assert flipped["enable_summarize"] is True


def test_prompts_reach_terminus2_outside_its_argv(tmp_path, monkeypatch) -> None:
    """The program runs next to the agent's commands, so an agent's ``pkill -f
    parser`` must not match it."""
    recorded = _run_terminus_program(
        tmp_path, monkeypatch, system_prompt="Answer in JSON."
    )
    assert not any("parser" in arg or "JSON" in arg for arg in recorded.argv)
    assert recorded.calls[-1] == ("run", "Fix the parser in /app.")
    history = asyncio.run(
        recorded.terminus._llm.call(message_history=[{"role": "user", "content": "hi"}])
    )
    assert history[0] == {"role": "system", "content": "Answer in JSON."}


def test_agent_shell_runs_inside_the_login_shell(tmp_path, monkeypatch) -> None:
    """As under Harbor's terminal recording, an ``exit`` ends only the inner shell."""
    assert _run_terminus_program(tmp_path, monkeypatch).calls == [
        ("setup",),
        ("send_keys", ["bash", "Enter"], 1.0),
        ("send_keys", ["clear", "Enter"], 0.0),
        ("run", "Fix the parser in /app."),
    ]


def test_tmux_server_outlives_launch_until_cleanup() -> None:
    """Verifiers scores between ``launch`` and ``cleanup``; the agent's tmux session
    and the jobs started from it must still be alive then."""
    harness = TerminalBenchTerminusHarness(TerminalBenchTerminusHarnessConfig())
    runtime = _RecordingRuntime()
    trace = SimpleNamespace(id="trace")
    asyncio.run(
        harness.launch(
            SimpleNamespace(model="torchtitan"),
            trace,
            runtime,
            "http://127.0.0.1:1/v1",
            "secret",
            {},
            SimpleNamespace(prompt="Fix the parser in /app.", system_prompt=None),
        )
    )
    assert runtime.commands == []

    asyncio.run(harness.cleanup(trace, runtime))
    [(argv, env)] = runtime.commands
    assert "tmux kill-server" in argv[-1]
    assert env == {"TMUX_TMPDIR": "/tmp/vf-terminus-2-trace"}
    assert runtime.program_env["TMUX_TMPDIR"] == env["TMUX_TMPDIR"]
    # The prompts file cleanup removes is the one launch wrote.
    assert '"$TMUX_TMPDIR.prompts.json"' in argv[-1]
    assert list(runtime.files) == ["/tmp/vf-terminus-2-trace.prompts.json"]


def test_agent_runs_inside_docker_and_verifier_uses_same_taskset() -> None:
    config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    environment = config.verifiers_env_server.environment

    assert isinstance(environment, HarborEnvConfig)
    assert isinstance(environment.agent.runtime, vf.DockerConfig)
    assert isinstance(environment.agent.harness, TerminalBenchTerminusHarnessConfig)
    assert environment.agent.harness.version == "0.22.0"
    assert config.generation_server.max_rollout_tokens == MAX_CONTEXT_LENGTH - 1
    assert environment.agent.max_turns == MAX_TURNS
    assert environment.agent.timeout.rollout == 7200
    assert environment.taskset == config.train_dataset.verifiers_taskset
    assert config.train_dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET
    assert config.verifiers_env_server.local_taskset_module == taskset.__name__
    worker_config = resolve_env_config(env_config_data(environment))
    assert worker_config.agent.harness == environment.agent.harness
    assert isinstance(
        load_harness(worker_config.agent.harness), TerminalBenchTerminusHarness
    )


def test_worker_process_resolves_the_harness_from_a_fresh_interpreter() -> None:
    """The env-server worker shares no ``sys.modules`` with the controller.

    It imports only the local taskset module and then rebuilds the environment
    config from JSON, so that one import must be enough to make the harness id
    resolvable. Resolving in the test process would pass regardless, because the
    controller side has already registered the alias there.
    """
    config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    environment = json.dumps(env_config_data(config.verifiers_env_server.environment))
    worker = f"""
import json
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from verifiers.v1.utils.loaders import load_harness, resolve_env_config

environment = json.loads({environment!r})
environment["taskset"]["id"] = register_local_taskset_alias(
    {config.verifiers_env_server.local_taskset_module!r}
)
env_config = resolve_env_config(environment)
print(type(load_harness(env_config.agent.harness)).__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", worker], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("TerminalBenchTerminusHarness")


def test_training_cannot_read_benchmark_as_training_data() -> None:
    with pytest.raises(ValueError, match="different datasets"):
        _rollouter_config(EVAL_DATASET, EVAL_DATASET)


def test_group_rewards_get_the_length_reward(monkeypatch) -> None:
    """A Terminal-Bench group is graded first, truncated rollouts included, then gets Kimi's
    length reward over completion tokens (terminal output does not count)."""
    from verifiers.v1.types import AssistantMessage, UserMessage

    def trace(completion_lens, *, reward, stop_condition=None, ok=True):
        # The task prompt, then per turn: the sampled reply and 3,000 tokens of terminal output.
        nodes = [SimpleNamespace(token_ids=[0] * 10, mask=[False] * 10, sampled=False)]
        for completion_len in completion_lens:
            nodes += [
                SimpleNamespace(
                    token_ids=[1] * completion_len,
                    mask=[True] * completion_len,
                    sampled=True,
                    message=AssistantMessage(content="{}"),
                ),
                SimpleNamespace(
                    token_ids=[2] * 3000,
                    mask=[False] * 3000,
                    sampled=False,
                    message=UserMessage(content="$ ls"),
                ),
            ]
        token_ids = [token_id for node in nodes for token_id in node.token_ids]
        span = SimpleNamespace(duration=1.0)
        return SimpleNamespace(
            id="trace",
            agent=SimpleNamespace(trainable=True),
            nodes=nodes,
            branches=[
                SimpleNamespace(
                    nodes=nodes, token_ids=token_ids, logprobs=[-0.1] * len(token_ids)
                )
            ],
            ok=ok,
            is_truncated=stop_condition is not None,
            stop_condition=stop_condition,
            reward=reward,
            task=SimpleNamespace(key="task"),
            timing=SimpleNamespace(
                setup=span,
                agent=SimpleNamespace(duration=1.0, model=span, harness=span),
                scoring=span,
            ),
            calls=[],
            errors=[],
        )

    class EnvClient:
        def __init__(self, traces) -> None:
            self.traces = traces

        async def run(self, *, sampling, **kwargs):
            return SimpleNamespace(
                ok=True, errors=[], traces=[self.traces[sampling.seed]]
            )

    def run_group(traces):
        rollouter._verifiers_env_client = EnvClient(traces)
        return asyncio.run(
            rollouter.run_group_rollouts(
                generate_fn=None,
                sample=VerifiersTaskSample(verifiers_task_data={}),
                group_id=0,
                group_size=len(traces),
                sampling=SamplingConfig(seed=0),
            )
        )

    # Skip loading the Harbor datasets; the env server is never started.
    monkeypatch.setattr(VerifiersTaskDataset, "__init__", lambda self, config: None)
    config = _rollouter_config(TRAIN_DATASET, EVAL_DATASET)
    config.rubric.length_reward_weight = 0.1
    rollouter = config.build()
    rollouter._generation_server = SimpleNamespace(
        model_id="torchtitan",
        generate_fns={},
        pop_generation_metadata=lambda trace_id: VerifiersGenerationMetadata(
            min_policy_version=0, max_policy_version=0, metrics=[]
        ),
    )
    rollouter._verifiers_train_client_config = object()

    # lam = 0.5 - (len - 200) / 800 over completion tokens: +0.5, 0, -0.5 at 200 / 600 / 1,000.
    group = run_group(
        [
            trace([100, 100], reward=1.0),
            trace([300, 300], reward=1.0),
            # Stopped at the context cap, then graded: its tests passed.
            trace([500, 500], reward=1.0, stop_condition="context_length"),
            trace([100, 100], reward=0.0),
            # Errored after 4,000 tokens (e.g. an agent timeout): not graded, no length
            # reward, and the group's range stays 200 to 1,000.
            trace([2000, 2000], reward=0.0, ok=False),
        ]
    )
    rollouts = group.rollouts
    assert rollouts[2].status == RolloutStatus.TRUNCATED_LENGTH
    assert rollouts[2].reward_breakdown == pytest.approx(
        {"RewardFromVerifiers": 1.0, "length_reward": -0.05}
    )
    assert rollouts[4].reward_breakdown == {"errored": 0.0, "length_reward": 0.0}
    assert [rollout.reward for rollout in rollouts] == pytest.approx(
        [1.05, 1.0, 0.95, 0.0, 0.0]
    )

    # An all-fail group gets the length term too, so its rewards are no longer all equal.
    group = run_group([trace([100, 100], reward=0.0), trace([500, 500], reward=0.0)])
    assert [rollout.reward for rollout in group.rollouts] == pytest.approx([0.0, -0.05])


def _terminal_bench_config(name: str) -> Controller.Config:
    return ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.rl.verifiers_terminal_bench",
            "--config",
            name,
        ]
    )


def test_training_recipe_uses_separate_datasets() -> None:
    config = _terminal_bench_config("rl_grpo_qwen35_9b_terminal_bench")
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.num_prompts_per_train_step == 12
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.async_loop.target_offpolicy_steps == 3
    assert config.trainer.training.max_context_length == 131072
    assert config.trainer.training.dtype == "float32"
    assert config.trainer.training.mixed_precision_param == "bfloat16"
    assert config.trainer.training.mixed_precision_reduce == "float32"
    (optimizer,) = config.trainer.optim.optimizer.optimizers
    assert optimizer.fused
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert isinstance(config.trainer.activation_checkpoint, FullAC.Config)
    assert config.trainer.checkpointer.interval == 20
    assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    assert config.num_generators == 8
    assert config.generator.parallelism.data_parallel_degree == 1
    assert config.rollouter.train_dataset.verifiers_taskset.dataset == TRAIN_DATASET
    assert config.rollouter.validation_dataset.verifiers_taskset.dataset == EVAL_DATASET


def _num_kv_heads(model: object) -> int:
    for layer in model.layers:
        attention = getattr(layer, "attention", None)
        if hasattr(attention, "n_kv_heads"):
            return attention.n_kv_heads
    raise AssertionError("model has no full-attention layer")


def _num_experts(model: object) -> int | None:
    for layer in model.layers:
        moe = getattr(layer, "moe", None)
        if moe is not None:
            return moe.num_experts
    return None


@pytest.mark.parametrize(
    ("name", "trainer_gpus", "num_generators", "gpus_per_generator"),
    [
        ("rl_grpo_qwen35_9b_terminal_bench", 8, 8, 1),
        ("rl_grpo_qwen35_35b_a3b_terminal_bench", 8, 2, 4),
    ],
)
def test_recipe_layouts_fit_the_model(
    name: str,
    trainer_gpus: int,
    num_generators: int,
    gpus_per_generator: int,
) -> None:
    """Each recipe's trainer and generator layouts respect the model's shape.

    The invariants are the ones a launch would otherwise trip over one at a
    time: tensor parallelism divides the KV heads in both roles, and for the MoE
    model expert parallelism divides the experts, is at least the trainer TP
    degree, and equals DP x TP in the generator. The GPU totals pin the intended
    16-GPU footprint.
    """
    config = _terminal_bench_config(name)
    trainer = config.trainer.parallelism
    generator = config.generator.parallelism
    num_kv_heads = _num_kv_heads(config.model)
    num_experts = _num_experts(config.model)

    assert (
        trainer.data_parallel_replicate_degree
        * trainer.data_parallel_shard_degree
        * trainer.tensor_parallel_degree
        * trainer.context_parallel_degree
        == trainer_gpus
    )
    assert config.num_generators == num_generators
    assert generator.data_parallel_degree * generator.tensor_parallel_degree == (
        gpus_per_generator
    )
    assert num_kv_heads % trainer.tensor_parallel_degree == 0
    assert num_kv_heads % generator.tensor_parallel_degree == 0

    if num_experts is None:
        assert trainer.expert_parallel_degree == 1
        assert generator.expert_parallel_degree == 1
        assert config.generator.cuda_graph.mode == "FULL_DECODE_ONLY"
    else:
        assert num_experts % trainer.expert_parallel_degree == 0
        assert trainer.expert_parallel_degree >= trainer.tensor_parallel_degree
        assert (
            trainer.data_parallel_shard_degree * trainer.tensor_parallel_degree
        ) % trainer.expert_parallel_degree == 0
        assert generator.expert_parallel_degree == (
            generator.data_parallel_degree * generator.tensor_parallel_degree
        )
        assert num_experts % generator.expert_parallel_degree == 0
        # The standard MoE dispatcher reads split sizes back to the host, which
        # CUDA graph capture does not allow.
        assert config.generator.cuda_graph.mode == "NONE"


@pytest.mark.parametrize(
    "name",
    [
        "rl_grpo_qwen35_9b_terminal_bench",
        "rl_grpo_qwen35_35b_a3b_terminal_bench",
    ],
)
def test_recipes_share_the_loop_and_keep_fp32_master_weights(name: str) -> None:
    """Every model trains on the same loop; only size-dependent settings differ.

    Master weights stay fp32 (the default): at a 1e-6 learning rate bf16
    parameters round most updates away. The env server must run every rollout
    the controller keeps in flight, or the excess queues and the generators idle.
    """
    config = _terminal_bench_config(name)
    assert config.trainer.training.dtype == "float32"
    assert config.async_loop.num_samples_per_prompt == 32
    assert config.async_loop.num_training_steps == 100
    assert config.async_loop.training_sample_builder.drop_zero_std_reward_groups
    assert config.generator.sampling.max_tokens == 16384
    agent = config.rollouter.verifiers_env_server.environment.agent
    assert agent.max_turns == 120
    assert (
        config.rollouter.generation_server.max_rollout_tokens
        == config.trainer.training.max_context_length - 1
    )
    loop = config.async_loop
    serve = config.rollouter.verifiers_env_server.serve
    assert serve.pool.num_workers * serve.max_concurrent >= (
        loop.max_active_rollout_groups * loop.num_samples_per_prompt
    )


def test_35b_sandoq_1x2_recipe_fits_three_hosts(monkeypatch) -> None:
    """A 4-GPU Dist-MoE trainer on one host and eight TP1 engines on two; the pool sizes the
    env server at 24 rollouts per worker."""
    pytest.importorskip("harbor")
    from torchtitan_recipes.rl.verifiers_plugins import terminal_bench_sandoq

    monkeypatch.syspath_prepend(str(Path(terminal_bench_sandoq.__file__).parent))
    monkeypatch.setenv("VF_SANDBOX_PROVIDER", "oci-runner")
    monkeypatch.setenv("OCI_RUNNER_TASK_NETWORK", "host")
    monkeypatch.setenv("DOME_SANDOQ_POOL", "920")
    monkeypatch.delenv("DOME_V2_PROMPTS", raising=False)
    monkeypatch.delenv("DOME_V2_THINKING_BUDGET", raising=False)
    monkeypatch.delenv("DOME_V2_LENGTH_REWARD_WEIGHT", raising=False)
    config = _terminal_bench_config("rl_grpo_qwen3_5_35b_a3b_base_terminal_bench_1x2")

    trainer = config.trainer.parallelism
    assert (
        trainer.data_parallel_shard_degree,
        trainer.tensor_parallel_degree,
        trainer.expert_parallel_degree,
    ) == (2, 2, 4)
    assert config.trainer.override.imports == [
        "torchtitan_recipes.overrides.dist_moe.dist_moe_routed_experts"
    ]
    assert config.trainer.training.num_tokens_per_microbatch_per_dp_rank == 131072
    assert config.num_generators == 8
    assert config.generator.parallelism.tensor_parallel_degree == 1
    assert config.generator.cuda_graph.mode == "FULL"
    assert config.generator.extra_vllm_engine_args == {"watermark": 0.03}
    loop = config.async_loop
    assert (loop.num_prompts_per_train_step, loop.num_samples_per_prompt) == (12, 16)
    assert loop.target_offpolicy_steps == 5
    assert loop.validation == ValidationConfig(steps=78, freq=10, overlap_training=True)
    assert config.rollouter.thinking_budget.max_thinking_tokens == 12288
    # Kimi's length reward assumes the mean baseline.
    assert config.rollouter.rubric.length_reward_weight == 0.1
    assert not config.rollouter.advantage.should_std_normalize
    serve = config.rollouter.verifiers_env_server.serve
    assert serve.pool.num_workers == 39
    assert serve.pool.num_workers * serve.max_concurrent >= 920
