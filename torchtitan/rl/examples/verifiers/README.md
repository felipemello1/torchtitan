# Verifiers integration

This directory contains the optional bridge between TitanRL and
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers). Other TitanRL
recipes do not import or require Verifiers.

The ownership boundary is:

- TitanRL loads samples, schedules rollout groups, routes generation, tracks
  policy versions, computes advantages, and trains the policy.
- Verifiers runs the environment, agent or tool loop, runtime, and task reward.
- The adapter converts samples and traces between the two systems.

Reusable bridge code lives alongside this README. Each experiment has its own
package and dependencies: [`dapo_math/`](./dapo_math) (single-turn math) and
[`terminal_bench/`](./terminal_bench) (multi-turn terminal agent on Harbor tasks).

## Integration flow

One group, top to bottom. Each column is a process, or the agent in its runtime.

```text
     controller      rollout worker                    agent        router     generator
     (Rollouter)     (VerifiersRolloutWorker: env,     (runtime)
                     interception, GenerationServer)
          |                  |                            |            |           |
 1        |--run_group------>|                            |            |           |
          |                  |                            |            |           |
 2        |                  |--start, per rollout------->|            |           |
          |                  |                            |            |           |
 3        |                  |<--chat---------------------|            |           |
          |                  |                            |            |           |
 4        |                  |--reserve------------------------------->|           |
          |                  |--generate------------------------------------------->|
          |                  |                            |            |           |
 5        |                  |--reply-------------------->|            |           |
          |                  |                            |            |           |
 6        |                  |--grade-------------------->|            |           |
          |                  |                            |            |           |
 7        |                  |--release_session----------------------->|           |
          |                  |                            |            |           |
 8        |<--scored group---|                            |            |           |
          |                  |                            |            |           |
 9        |--release_groups--------------------------------------------->|           |
```

1. The controller sends one task to a rollout worker. The task is plain data from
   [`data.py`](./data.py).
2. The worker's Verifiers env runs the task once per rollout of the group (`env.run_slot`). It
   starts the agent in the env's runtime, e.g. a local process or a Docker container. Verifiers
   reports each trace it starts (`on_trace`), and the worker registers that trace id in its
   GenerationServer as a session of this group.
3. Each agent turn, the agent sends a chat request to Verifiers' interception server, in the
   worker. Verifiers' train client renders the messages to token ids and sends them to the worker's
   [`GenerationServer`](./generation_server.py), over HTTP on the loopback address.
4. The GenerationServer finds the call's group by its session id, the trace id (a session it
   doesn't know gets 422). It calls TitanRL's `generate_fn`, which reserves a generator at the
   router and then calls that generator.
5. The GenerationServer answers with the tokens and logprobs, and keeps the call's policy version
   and generator metrics: the trace has no field for them. The agent gets its reply. Steps 3 to
   5 repeat every turn.
6. When the agent stops, Verifiers grades the rollout, e.g. by running the task's tests.
7. The worker releases the routing session of each trace Verifiers made, a retried attempt's
   included.
8. The worker turns the rollout's trainable trace into TitanRL rollout turns
   ([`rollouter.py`](./rollouter.py)), scores the group, computes its advantages, and returns it.
9. The group makes no more model calls. Through the router, the controller tells each generator to
   forget the group: the policy version its prompts are cached under, and any KV cache it still
   holds for the group.

Shutdown:

1. Stop admitting groups.
2. Close the generators, so model calls still waiting fail fast.
3. Drain the running groups.
4. Each worker's `close` cancels its episodes, which stops their sandboxes. Then it stops serving
   the env, then the GenerationServer.
5. Stop the worker processes.

A killed worker skips `close`, so its sandboxes stay up.

How many processes: the controller and the rollout workers (one per `rollouts_per_worker` training
rollouts in flight, rounded up). Each worker loads the env and runs its episodes; an agent runs in the
env's runtime.

## Inside a Verifiers rollout worker

A Verifiers worker runs everything on the worker's one event loop and its executor threads, like any
rollout worker: see [Rollout workers](../../rollout/ROLLOUT_WORKERS.md) for the worker itself, its
settings and its metrics. Verifiers adds these pieces (numbers from that page's CPU bench, Verifiers
0.4.1, Terminal-Bench timing):

| Piece | How many per worker | Runs on | Cost |
| --- | --- | --- | --- |
| Interception servers, which the agents call | 1 per 32 rollouts (`ElasticInterceptionPoolConfig.multiplex`, `verifiers==0.4.1 verifiers/v1/configs/interception.py:78`) | the event loop | ~1.5 MiB and no thread each. 100 rollouts, 2 runs each: one server per rollout was no faster. |
| Renderers: chat messages -> token ids | 1 per 256 model calls in flight (`VerifiersRolloutWorker.Config.renderer_multiplex`, [`rollouter.py`](./rollouter.py)) | executor threads, one render at a time per renderer | ~200 MiB of private memory each with rollouts running (3 extra renderers: 607 MiB). The `renderer_multiplex` docstring's ~170 MB is the memory growth at the moment one loads. A worker keeps each one until it exits. At 64-160 rollouts per worker, a worker never built a second one. At 960 rollouts per worker it built 4, and 1 renderer instead gave the same calls/s with 607 MiB less. |
| `GenerationServer` | 1 | the event loop | Verifiers' train client calls it over loopback HTTP with the token ids as JSON: 2.7 ms of loop CPU per model call at 12K prompt tokens, 9.1 ms at 64K. |

A renderer runs one render at a time because it takes a lock
(`verifiers==0.4.1 verifiers/v1/clients/train.py:227-242`). One worker with 960 rollouts and one
renderer made ~14 model calls/s. At 2.7 ms per render (that run's median), its lock was busy ~4% of
the time.

**Local Docker runtime** (`vf.DockerConfig()`, as the Terminal-Bench recipe uses). Measured with
rootless Podman 5.8.5 behind the `docker` command. Docker itself wasn't measured. Below, a running
rollout is one whose agent is still working, as opposed to one that finished and waits for the rest
of its group:

- Each running rollout keeps one `docker exec` CLI process attached while its agent runs. The harness
  runs the agent with `runtime.run` ([`terminal_bench/harness.py`](./terminal_bench/harness.py)),
  which waits on `docker exec` (`verifiers==0.4.1 verifiers/v1/runtimes/backends/docker.py:689-695`).
  With Podman, each CLI process holds ~24 MiB of memory (PSS: shared pages split across the processes
  that map them) and 16 threads. For example, at 600 running rollouts on one host, that is ~14 GiB and
  ~9,800 threads (extrapolated linearly from 64 measured rollouts).
- Each rollout's cleanup runs `docker rm --force` in `asyncio.to_thread`
  (`verifiers==0.4.1 verifiers/v1/runtimes/backends/docker.py:674-688`,
  `verifiers/v1/runtimes/base.py:245`). Under Podman it took 10.7 s: Podman waits its 10 s stop
  timeout, because the container's first process, `sleep infinity` (`docker.py:358-363`), ignores
  the stop signal. So each cleanup holds an executor thread for 10.7 s. With 4 threads and 160
  rollouts per worker, model calls waited ~10 s for a thread. 32 threads, the default, fix it (sizing
  rule: [`num_threads_per_worker`](../../rollout/ROLLOUT_WORKERS.md#num_threads_per_worker)). A cleanup
  with `podman rm --force --time 0` took 0.18 s. Verifiers 0.4.1 doesn't pass `--time`.
- One egress proxy per sandbox, on the worker's loop, lets the agent reach the interception server.
  It looks up its target's address on every request, on an executor thread
  (`verifiers==0.4.1 verifiers/v1/runtimes/egress.py:243`).
- In total, a local container added ~6 ms of loop CPU per model call: 23.7 vs 17.5 ms with a local
  process runtime. ~2 ms of it is the egress proxy.

## Add an experiment

For a task `my_task`:

```text
torchtitan/rl/examples/verifiers/my_task/
  __init__.py
  data.py                                  # your Taskset and its TasksetConfig
  requirements.txt                         # its dependencies
torchtitan_recipes/rl/verifiers_my_task.py # recipe functions that return a Controller.Config
```

Install its dependencies with `pip install -r torchtitan/rl/examples/verifiers/my_task/requirements.txt`.
Start the recipe from a copy of
[`verifiers_dapo_math.py`](../../../../torchtitan_recipes/rl/verifiers_dapo_math.py), and change its
`Rollouter.Config`: your `TasksetConfig` in its training and validation datasets, and your agent,
harness and runtime in `worker.environment`. Pass the same training taskset object to the
training dataset and to `worker.environment`: the worker builds each task with the env's taskset.
So the validation dataset's taskset needs the same task class and task config.
Then run:

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_my_task \
  --config <recipe function name>
```

See [DAPO Math](./dapo_math) for a complete single-turn example and
[Terminal-Bench](./terminal_bench) for a multi-turn agent example. This
integration pins Verifiers 0.4.1 and uses its `verifiers.v1` API.
