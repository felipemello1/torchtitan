# Async RL weight-sync: run setup (env + torchstore)

The trainer->generator weight sync is CPU-staged through torchstore. For same-host runs, set these
env vars on the launch command. They are run configuration, **not** code changes:

```bash
TORCHSTORE_MUTABLE_SHM=1   # GET returns the shared-memory tensor directly instead of cloning each
                          #   shard -> pull ~9s -> ~2.5s. Safe: read-then-assemble; validated over full
                          #   runs (reward keeps learning, no corruption).
OMP_NUM_THREADS=1         # avoid BLAS/OpenMP oversubscription during the CPU-staged copy
MKL_NUM_THREADS=1         #   (many actor processes per host).
USE_TORCHCOMMS=0          # use the MonarchRDMA / SharedMemory path (same-host resolves to SharedMemory).
```

## torchstore

Install from `main` (has the separate push copy-stream and CPU-staged `transfer_dtype` casting). Use
`--no-deps` so it does not pull a different `torch` / `torchcomms` into the env:

```bash
pip install --no-deps --force-reinstall 'torchstore @ git+https://github.com/meta-pytorch/torchstore.git'
```

## Measurement caveat

`timing/weight_sync/trainer_push_model_state_dict` is a wall-clock span around an `await`. When the push
overlaps the next step's fwd/bwd it absorbs that fwd/bwd time, so it reads ~3s while the real push work is
~0.4s. The pull (~2.8s) is real.

For an accurate push number, time the copy where it is a non-yielding region (torchstore's copy stream),
not across the `await`. See the TODO in `components/weight_sync.py`.
