# Generator routing

Two layers route each generate call:

- `InterGeneratorRouter` (one actor per job) picks a generator mesh. It never sees the payload: `reserve`
  returns the generator's handle, the caller sends the call there, then calls `release_reservation`.
- `IntraGeneratorRouter` (rank 0 of each generator) picks a data-parallel rank inside the mesh.

Both use a `RoutingStrategy` (`strategies.py`). The default keeps a session's calls (one multi-turn rollout)
on one generator, so each turn reuses that generator's cached prefix.

## KV-room admission

Multi-turn rollouts keep their history in the KV cache between turns (the vLLM engine's
`session_kv_holder`). When more sessions run than the cache holds, the holder releases idle histories,
and their next turns recompute them. Admission (`InterGeneratorRouter.Config.admission`) starts a
rollout group only when a generator has room for it, and sends all the group's calls there.

The router keeps one ledger per generator (`KVRoomAdmission`), in KV blocks, from each generator's
`KVCacheBudget` (read from vLLM's KV cache spec at start):

    idle session (between turns)       its context's attention blocks + a state block per Mamba group
    call outstanding, or not started   + the blocks vLLM gives a running request + the holder's floor
    usable blocks (H)                  the pool minus the holder's free floor

The controller calls `acquire_group` before a group builds its envs, and `release_groups` after it.
Each call reports its prompt size to `reserve` and its prompt + completion size to `release_reservation`.

Example (one generator, H = 3,030 blocks; a group starts while ledger + its charge <= 95% of H = 2,878):

    50 live sessions charge 2,700. Group 41 (16 fresh sessions x 22 = 352) waits: 3,052 > 2,878.
        Its envs don't start.
    Two rollouts end (-180): 2,520 + 352 = 2,872 <= 2,878. Group 41 is placed; its envs start,
        and all its calls go to this generator.

Groups wait in order: validation groups first, by arrival, then training groups, lowest id first. A
generator with no live session takes a group even over the mark. Before the first call shows a prompt
size, a new session is charged the max model length, so each generator takes one group at the start.

Not supported yet: generators with data parallelism (`KVRoomAdmission` raises), sliding-window attention,
and Mamba groups outside vLLM's "align" mode (`VLLMEngine.kv_cache_budget` raises).
