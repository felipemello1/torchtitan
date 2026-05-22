#!/usr/bin/env bash
# Real torchtitan SFT training comparison: branch vs upstream/main.
# Reproduces the R101 table in RFC.md §8.1.
#
# Usage:
#   export CANDIDATE_WORKTREE=/path/to/branch-worktree
#   export MAIN_WORKTREE=/path/to/upstream-main-worktree
#   export OUT=/tmp/chunked_ce_v2_real_train
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash parity_real_titan.sh
#
# Expected: deterministic loss + grad_norm parity under FSDP=4. Stdout
# is only 5-sig-fig precision; use TensorBoard exports for bitwise checks.

set -uo pipefail

CANDIDATE=${CANDIDATE_WORKTREE:?must export CANDIDATE_WORKTREE}
MAIN=${MAIN_WORKTREE:?must export MAIN_WORKTREE}
OUT=${OUT:?must export OUT}

TORCHRUN=${TORCHRUN:-torchrun}
STEPS=${STEPS:-10}
SEQ_LEN=${SEQ_LEN:-2048}
BS=${BS:-8}
NUM_CHUNKS=${NUM_CHUNKS:-8}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_ALLOC_CONF="expandable_segments:True"

mkdir -p "$OUT/logs"
mkdir -p "$OUT/results/$(date +%Y-%m-%d_real_titan_sft)/raw_logs"

run_titan() {
    local label="$1"
    local root="$2"
    local ngpu="$3"
    local extra="$4"

    local log="$OUT/logs/${label}.log"
    echo
    echo "=== [$label] root=$root ngpu=$ngpu extra=$extra ==="

    (
        cd "$root"
        "$TORCHRUN" --nproc_per_node="$ngpu" \
            --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
            --local-ranks-filter 0 --role rank --tee 3 \
            -m torchtitan.train \
            --module llama3 --config llama3_debugmodel \
            --debug.deterministic \
            --debug.seed=42 \
            --training.steps="$STEPS" \
            --training.seq_len="$SEQ_LEN" \
            --training.local_batch_size="$BS" \
            --loss.num-chunks="$NUM_CHUNKS" \
            --metrics.log_freq=1 \
            --metrics.disable-color-printing \
            $extra
    ) > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "  OK -> $log"
    else
        echo "  FAIL rc=$rc — tail:"
        tail -30 "$log"
    fi
    return $rc
}

# 3 parallelism configs × 2 worktrees = 6 runs
run_titan "main_fsdp4"     "$MAIN" 4 "--parallelism.data_parallel_shard_degree=4"
run_titan "candidate_fsdp4"     "$CANDIDATE" 4 "--parallelism.data_parallel_shard_degree=4"
run_titan "main_fsdp2_tp2" "$MAIN" 4 "--parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2"
run_titan "candidate_fsdp2_tp2" "$CANDIDATE" 4 "--parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2"
run_titan "main_tp4"       "$MAIN" 4 "--parallelism.tensor_parallel_degree=4"
run_titan "candidate_tp4"       "$CANDIDATE" 4 "--parallelism.tensor_parallel_degree=4"

# Extra: compile cell to verify §9 Pitfall 3 (compile masks Partial->Replicate)
run_titan "main_fsdp2_tp2_compiled" "$MAIN" 4 \
  "--parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --compile.enable --compile.components ['\"loss\"']"
run_titan "candidate_fsdp2_tp2_compiled" "$CANDIDATE" 4 \
  "--parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 --compile.enable --compile.components ['\"loss\"']"

echo
echo "=== done. Logs in $OUT/logs/ ==="
echo "Extract step rows with:"
echo "  for f in $OUT/logs/*.log; do echo \"=== \$f ===\"; grep \"step:\" \"\$f\" | head -10; done"
