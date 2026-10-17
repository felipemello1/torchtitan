# FairTitan TorchTitan stack, 2026-10-17 (not for merge)

Base: `snapshot/swe35b-1866484` @ `3096c9aad`, the TorchTitan commit the 10-node Qwen3.5-35B-A3B run (job 1866484)
ran. That branch is frozen; this one adds the fixes below on top. Each line is the fork PR and the commits cherry-picked
from it (`-x` in each commit message).

## Included

| PR | What | Commits picked | How |
|---|---|---|---|
| #355 | top-k logprobs cross Monarch as numpy | f1190ffb8 | clean |
| #351 | generation server: token-id validation in one pass | ce14321d4 | clean |
| #352 | generation server: msgspec request decode (+ msgspec dependency) | 42be6d8ff, 961b1ad8e | import conflict (numpy + msgspec), kept both |
| #350 | generation server: routed-experts base64 spliced into the response | d3d93b439, 35b72e190 | test import conflict, kept both |
| #353 | concurrent `sync_log_step` | 2db0567ba, e1b6a9ad1 | controller change applied; its test file (`test_trainer_loop.py`) doesn't exist on this base, so the test is dropped |
| #354 | controller event-loop stall watchdog (opt-in) | 9ebdcaee0 | ported by hand to this base's layout (`torchtitan/rl/observability/loop_stall_watchdog.py`, `torchtitan/rl/controller.py`) |
| #263 | opt-in controller heap census, plus the fix that empties its object lists (no census-held cycle) | 08203ee4b, 087fc6322 | both-sides conflicts in controller/test, kept both |
| #318 | Triton grouped GEMMs behind `RoutedExperts._expert_mlp` (override, opt-in) | 684701d42, 9832ae09b, f2296f0f9, ed71798fc, fa693768d | clean |
| #317 | EP=1 Triton routed experts, router, small-batch GEMVs (override, opt-in; `match_trainer_bitwise` opt-in) | a0e058fbd … 1a674f853 (9) | clean |

The MoE overrides stay off unless a recipe enables them.

## Not included
- #348 (numpy prompt in the generation server), #362 (small tensors as numpy): open PRs not yet approved.
- #343, #346 (Dist-MoE count skip, batch-invariant router region): not in this pass.
- #313 (manual controller GC): its knob isn't needed with the current defaults; add it when a recipe turns GC off.
