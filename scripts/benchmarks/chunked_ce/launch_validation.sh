#!/usr/bin/env bash
# Validation matrix for the consolidated chunked-CE-reducer branch.
#
# Runs selected validation cells on the worktree at $BASE. Default is the CPU-only cell F.
#
#   # 1. Baseline (before cleanup):
#   BASE=/path/to/codex/branch OUT=/tmp/validation_baseline bash launch_validation.sh
#
#   # 2. Post-cleanup:
#   BASE=/path/to/cleaned/branch OUT=/tmp/validation_postcleanup bash launch_validation.sh
#
#   # 3. Diff:
#   diff -r /tmp/validation_baseline/parsed /tmp/validation_postcleanup/parsed
#
# Each cell writes raw stdout + a parsed table of step:loss:grad_norm:tps:mem
# to $OUT.

set -euo pipefail

BASE=${BASE:?must export BASE worktree path}
OUT=${OUT:-/tmp/chunked_ce_v2_validation}
PYBIN=${PYBIN:-/home/felipemello/.conda/envs/titan/bin/python}
RL_PYBIN=${RL_PYBIN:-/home/felipemello/.conda/envs/titan_rl/bin/python}
TORCHRUN=${TORCHRUN:-/home/felipemello/.conda/envs/titan/bin/torchrun}
RL_HF_ASSETS_PATH=${RL_HF_ASSETS_PATH:-/home/felipemello/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}
RL_CONFIG=${RL_CONFIG:-rl_grpo_qwen3_0_6b_batch_invariant}
RL_STEPS=${RL_STEPS:-1}
RL_PROMPTS=${RL_PROMPTS:-1}
RL_VALIDATION_SAMPLES=${RL_VALIDATION_SAMPLES:-1}
RL_GROUP_SIZE=${RL_GROUP_SIZE:-2}
RL_MAX_TOKENS=${RL_MAX_TOKENS:-20}
RL_DIRECT_RDMA=${RL_DIRECT_RDMA:-0}
RL_EXTRA=${RL_EXTRA:-}

CELLS=${CELLS:-F}   # space-separated; choose from A B C D E F

needs_cuda=0
for cell in $CELLS; do
    case "$cell" in
        A|B|C|D|E) needs_cuda=1 ;;
    esac
done
if [ "$needs_cuda" -eq 1 ]; then
    : "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES explicitly, e.g. 0,1,2,3 for codex or 4,5,6,7 for opus}"
fi
export PYTORCH_ALLOC_CONF="expandable_segments:True"

mkdir -p "$OUT/logs" "$OUT/parsed"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

write_run_info() {
    local out="$OUT/parsed/RUN_INFO.txt"
    {
        echo "BASE=$BASE"
        echo "BASE_HEAD=$(cd "$BASE" && git rev-parse HEAD)"
        echo "BASE_STATUS=$(cd "$BASE" && git status --short | head -10)"
        echo "PYBIN=$PYBIN"
        echo "PYBIN_VERSION=$($PYBIN --version 2>&1)"
        echo "RL_PYBIN=$RL_PYBIN"
        echo "RL_PYBIN_VERSION=$($RL_PYBIN --version 2>&1)"
        echo "RL_CONFIG=$RL_CONFIG"
        echo "RL_HF_ASSETS_PATH=$RL_HF_ASSETS_PATH"
        echo "RL_DIRECT_RDMA=$RL_DIRECT_RDMA"
        echo "TORCHSTORE_RDMA_ENABLED=${TORCHSTORE_RDMA_ENABLED:-0}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
        nvidia-smi --query-gpu=index,name,memory.used --format=csv || true
        echo "DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$out"
    echo "run info -> $out"
}

run_sft_cell() {
    local label="$1" ngpu="$2" extra="$3"
    local extra_compile="$4"  # e.g. "" or "--compile.enable --compile.components ['\"loss\"']"
    local seq_len="${5:-2048}" bs="${6:-8}" steps="${7:-10}"

    local log="$OUT/logs/${label}.log"
    local parsed="$OUT/parsed/${label}.txt"
    echo
    echo "=== [$label] cd $BASE; ngpu=$ngpu seq_len=$seq_len bs=$bs steps=$steps extra=$extra ==="

    (
        cd "$BASE"
        "$TORCHRUN" --nproc_per_node="$ngpu" \
            --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
            --local-ranks-filter 0 --role rank --tee 3 \
            -m torchtitan.train \
            --module llama3 --config llama3_debugmodel \
            --debug.deterministic --debug.seed=42 \
            --training.steps="$steps" \
            --training.seq_len="$seq_len" \
            --training.local_batch_size="$bs" \
            --loss.num-chunks=8 \
            --metrics.log_freq=1 \
            --metrics.disable-color-printing \
            $extra $extra_compile
    ) > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  FAIL rc=$rc — tail:"; tail -30 "$log"
        return $rc
    fi

    grep -E "step: " "$log" \
        | sed -E 's/.*step: +([0-9]+).*loss: +([0-9.]+).*grad_norm: +([0-9.]+).*memory: +([0-9.]+)GiB.*tps: +([0-9,]+).*/step=\1 loss=\2 grad_norm=\3 mem_gib=\4 tps=\5/' \
        > "$parsed"
    echo "  OK; parsed -> $parsed"
    return 0
}

run_rl_cell() {
    local label="$1"
    # RL smoke: vLLM generator + Qwen3-0.6B.
    # NOTE: requires titan_rl env (has vllm) instead of titan.
    local log="$OUT/logs/${label}.log"
    local parsed="$OUT/parsed/${label}.txt"
    echo
    echo "=== [$label] RL smoke ==="
    (
        cd "$BASE"
        export TORCHTITAN_RL_DIRECT_RDMA="$RL_DIRECT_RDMA"
        export TORCHSTORE_RDMA_ENABLED="${TORCHSTORE_RDMA_ENABLED:-0}"
        "$RL_PYBIN" -m torchtitan.experiments.rl.grpo \
            --module rl \
            --config "$RL_CONFIG" \
            --hf_assets_path "$RL_HF_ASSETS_PATH" \
            --dump_folder "$OUT/${label}_dump" \
            --num_steps "$RL_STEPS" \
            --num_prompts_per_step "$RL_PROMPTS" \
            --num_validation_samples "$RL_VALIDATION_SAMPLES" \
            --generator.sampling.n "$RL_GROUP_SIZE" \
            --generator.sampling.max_tokens "$RL_MAX_TOKENS" \
            --trainer.chunked_loss_num_chunks 8 \
            --metrics.no-enable_wandb \
            --metrics.enable_tensorboard \
            --metrics.console_log_keys_train "loss/mean,train/grad_norm/mean,perf/tokens_per_second,train/memory/max_active_gib,reward/_mean,reward/group_std/mean,reward/zero_std_frac" \
            $RL_EXTRA
    ) > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  FAIL rc=$rc — tail:"; tail -80 "$log"
        return "$rc"
    fi

    grep -F "Train | Step:" "$log" \
        | sed -E 's/.*Train [|] Step: *([0-9]+) */step=\1 /; s/  +/ /g' \
        > "$parsed"
    echo "  OK; parsed -> $parsed"
    return 0
}

run_unit_cell() {
    local label="$1"
    local core_log="$OUT/logs/${label}_core.log"
    local rl_log="$OUT/logs/${label}_rl.log"
    echo
    echo "=== [$label] CPU unit tests ==="
    (
        cd "$BASE"
        "$PYBIN" -m pytest tests/unit_tests/test_loss.py
    ) > "$core_log" 2>&1
    local core_rc=$?
    grep -E "passed|failed|error" "$core_log" | tail -5 || true
    echo "  core log -> $core_log"

    (
        cd "$BASE"
        "$RL_PYBIN" -m pytest torchtitan/experiments/rl/tests/test_grpo_metrics.py
    ) > "$rl_log" 2>&1
    local rl_rc=$?
    grep -E "passed|failed|error" "$rl_log" | tail -5 || true
    echo "  rl log -> $rl_log"
    return $((core_rc + rl_rc))
}

# -----------------------------------------------------------------------------
# Cell registry
# -----------------------------------------------------------------------------

cell_A() {
    run_sft_cell "A_sft_fsdp4" 4 "--parallelism.data_parallel_shard_degree=4" ""
}

cell_B() {
    run_sft_cell "B_sft_fsdp2_tp2" 4 \
        "--parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2" ""
}

cell_C() {
    run_sft_cell "C_sft_tp4" 4 "--parallelism.tensor_parallel_degree=4" ""
}

cell_D() {
    run_sft_cell "D_sft_fsdp2_tp2_L128k_compiled" 4 \
        "--parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2" \
        "--compile.enable --compile.components ['\"loss\"']" \
        131072 1 5
}

cell_E() {
    run_rl_cell "E_rl_qwen3_06b_nonzero"
}

cell_F() {
    run_unit_cell "F_cpu_unit_tests"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

write_run_info

for cell in $CELLS; do
    case "$cell" in
        A) cell_A ;;
        B) cell_B ;;
        C) cell_C ;;
        D) cell_D ;;
        E) cell_E ;;
        F) cell_F ;;
        *) echo "unknown cell: $cell" >&2 ;;
    esac
done

echo
echo "=== done. Logs in $OUT/logs/; parsed in $OUT/parsed/ ==="
echo
echo "Compare baseline vs post-cleanup with:"
echo "  diff -r /tmp/validation_baseline/parsed /tmp/validation_postcleanup/parsed"
