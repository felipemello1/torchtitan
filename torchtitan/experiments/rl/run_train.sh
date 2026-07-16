#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience, e.g.
# MODULE=search_r1 CONFIG=rl_grpo_qwen3_8b_search_r1 ./torchtitan/experiments/rl/run_train.sh
MODULE=${MODULE:-"alphabet_sort"}
CONFIG=${CONFIG:-"rl_grpo_qwen3_0_6b_varlen"}

# Monarch-spawned trainer/generator procs must import torchtitan from this checkout.
# Run from the repo root (the default hf_assets_path is repo-root-relative).
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# CPU-staged weight-sync defaults. Single-threaded BLAS avoids oversubscription across
# per-host actor procs. USE_TORCHCOMMS=0 selects MonarchRDMA for cross-host transfers;
# same-host transfers resolve to SharedMemory first.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export USE_TORCHCOMMS=${USE_TORCHCOMMS:-0}

# Same-host GET returns the shared-memory tensor directly instead of cloning each shard.
export TORCHSTORE_MUTABLE_SHM=${TORCHSTORE_MUTABLE_SHM:-1}

# rl/train.py sets this before importing torch; export it here so the launch
# contract is visible and overridable in one place.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

python3 -m torchtitan.experiments.rl.train \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    "$@"
