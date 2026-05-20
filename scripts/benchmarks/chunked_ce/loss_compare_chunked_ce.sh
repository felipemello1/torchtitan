#!/usr/bin/env bash
# Wrapper around scripts/loss_compare.py for chunked CE validation.
#
# Run from a clean worktree. scripts/loss_compare.py checks out the requested commits.

set -euo pipefail

BASELINE_COMMIT=${BASELINE_COMMIT:-upstream/main}
TEST_COMMIT=${TEST_COMMIT:-HEAD}
OUT=${OUT:-/tmp/chunked_ce_v2_loss_compare}
STEPS=${STEPS:-10}
NGPUS=${NGPUS:-4}
SEQ_LEN=${SEQ_LEN:-2048}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-8}
NUM_CHUNKS=${NUM_CHUNKS:-8}
MODULE=${MODULE:-llama3}
CONFIG=${CONFIG:-llama3_debugmodel}
PARALLELISM_OPTIONS=${PARALLELISM_OPTIONS:-"--parallelism.data_parallel_shard_degree=4"}
COMPILE_OPTIONS=${COMPILE_OPTIONS:-}
EXTRA_OPTIONS=${EXTRA_OPTIONS:-}

cd "$(git rev-parse --show-toplevel)"

COMMON_OPTIONS="--training.seq_len=${SEQ_LEN} --training.local_batch_size=${LOCAL_BATCH_SIZE} --loss.num-chunks=${NUM_CHUNKS} ${PARALLELISM_OPTIONS} ${COMPILE_OPTIONS} ${EXTRA_OPTIONS}"

python scripts/loss_compare.py \
    "$BASELINE_COMMIT" \
    "$TEST_COMMIT" \
    --baseline-module="$MODULE" \
    --test-module="$MODULE" \
    --baseline-config="$CONFIG" \
    --test-config="$CONFIG" \
    --baseline-options="$COMMON_OPTIONS" \
    --test-options="$COMMON_OPTIONS" \
    --steps="$STEPS" \
    --baseline-ngpus="$NGPUS" \
    --test-ngpus="$NGPUS" \
    --output-folder="$OUT"
