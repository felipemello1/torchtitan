# Observability Implementation

Planning docs, PR plans, and rules at:
`/home/felipemello/forge/reports/observability/implementationV3/`

Start with `index.md`, then `instructions.md`, then your PR plan.

## Your mission

Implement all PRs (PR0 through PR8). **DO NOT RUSH.** Each PR must be
fully verified before moving to the next. Quality over speed.

You do not have to agree with everything the verifier subagents say. But do
not rationalize bad decisions as if they were OK — if a verifier catches a
real issue, fix it.

## MANDATORY: Per-PR checklist (DO NOT SKIP ANY STEP)

**For EVERY PR, complete ALL of these steps IN ORDER before moving on:**

1. **Write library code** (copy from reference with `cp`, then targeted edits)
2. **Write unit tests** — run them, they must pass
3. **Update toy_spmd.py** to use the NEW features from THIS PR
   - The toy example must only use features from this PR + prior PRs
   - This update goes in THIS PR's commit
4. **Run toy example:** `torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py`
5. **Verify outputs** (JSONL files, content correctness, loss decreases)
6. **Launch 2 verifier subagents** that check code + test output
7. **Fix any real issues** found by verifiers
8. **Commit** — one commit per PR, includes library + tests + toy example update
9. **Only then** move to the next PR

Each PR builds on the previous. We need 100% certainty that each PR works
end-to-end before moving on. **Unit tests alone are NOT sufficient.**
**Do not rush to the next PR.** Thoroughness beats speed.

## Integration test pattern

The toy example (`toy_spmd.py`) is a clean training script that produces outputs.
Integration tests are separate pytest files that read those outputs and validate.

Flow per PR:
1. `toy_spmd.py` runs, writes JSONL/JSON to `/tmp/toy_spmd_output/`
2. `test_integration_prN.py` reads the output and validates correctness

The toy example should NOT contain validation logic — keep it clean.
Integration tests go in `tests/unit_tests/observability/test_integration_prN.py`.

## Quick rules (details in instructions.md)

- Copy from reference libraries (sixlib, msl_tools, llama4x), not from `~/forge/.../observability/` (V2 code had bugs)
- When copying, use `cp` then targeted edits — don't rewrite from scratch
- Every non-verbatim change needs a reason tag
- No internal references (sixlib, msl_tools, scuba, TBR, borescope) in code
- All experiment metrics through JSONL — no dict plumbing
- ContextVar for step/step_tags only; rank/source on formatter (see design/decision_003)
- Writer lifecycle: open in setup, close explicitly. No try/finally, no atexit.
- If anything is changed/deleted/added outside of what the PR plan specifies, log it in `log.md` with justification
- After copying from reference, launch a diff comparison subagent to verify faithfulness

## Environment

- Repo: `~/torchtitan` (branch: `observability`)
- 8 GPUs available
- PyTorch 2.7.0 nightly, Monarch (`torchmonarch`)
- PR0-PR6: dummy TinyMLP (fast). PR7+: Qwen 0.5B first, then Llama 3.1 8B.

## Test commands

```bash
# Unit tests
pytest tests/unit_tests/observability/ -vv

# Toy SPMD example (integration)
torchrun --nproc_per_node=4 torchtitan/experiments/observability/toy_spmd.py
```

## Process details

See `orchestration_prompt.md` for the full execution loop, verifier protocol,
logging format, and decision framework.
