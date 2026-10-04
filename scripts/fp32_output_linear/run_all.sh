#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Regenerate every FP32OutputLinear number: one text file per script in <out_dir>, stderr in
# <out_dir>/logs. See README.md for which output line backs which number.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 scripts/fp32_output_linear/run_all.sh <out_dir> [--with-27b]
#
# Needs 2 GPUs: fsdp_fp32_grad.py uses both, everything else runs on the first, one at a time.
# --with-27b also downloads Qwen3.5-27B (54 GB) and regenerates the PR's "Controlled results".
# Env: PYTHON (default: python), CACHE_DIR (default: ~/.cache/fp32_output_linear).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <out_dir> [--with-27b]" >&2
  exit 1
fi
OUT=$1
WITH_27B=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PYTHON:-python}
CACHE=${CACHE_DIR:-$HOME/.cache/fp32_output_linear}
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$OUT/logs"

{
  echo "date: $(date '+%F %T')"
  echo "torchtitan: $(git -C "$REPO" rev-parse HEAD) ($(git -C "$REPO" log -1 --format=%s))"
  echo "uncommitted changes: $(git -C "$REPO" status --short | wc -l) files"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
  "$PY" -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '|', torch.cuda.get_device_name())"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,clocks.max.sm --format=csv
} > "$OUT/environment.txt"

# run <name> <command...>: stdout to <out_dir>/<name>.txt, stderr to <out_dir>/logs/<name>.log.
# A failing script is reported and the others still run.
FAILED=()
run() {
  local name=$1
  shift
  echo "$(date '+%T') $name"
  if ! "$@" > "$OUT/$name.txt" 2> "$OUT/logs/$name.log"; then
    echo "  FAILED: see $OUT/logs/$name.log"
    FAILED+=("$name")
  fi
}

FIRST_GPU=${CUDA_VISIBLE_DEVICES:-0}
FIRST_GPU=${FIRST_GPU%%,*}
one_gpu() { CUDA_VISIBLE_DEVICES=$FIRST_GPU "$@"; }

if [ "$WITH_27B" = "--with-27b" ]; then
  run prepare_data one_gpu "$PY" "$HERE/prepare_data.py" --cache-dir "$CACHE" --with-27b
else
  run prepare_data one_gpu "$PY" "$HERE/prepare_data.py" --cache-dir "$CACHE"
fi
run docstring_table one_gpu "$PY" "$HERE/docstring_table.py" --cache-dir "$CACHE"
run rounding one_gpu "$PY" "$HERE/rounding.py" --cache-dir "$CACHE"
run split_k one_gpu "$PY" "$HERE/split_k.py" --cache-dir "$CACHE"
run split_compile one_gpu "$PY" "$HERE/split_compile.py"
run split_compile_TORCH_COMPILE_DISABLE one_gpu env TORCH_COMPILE_DISABLE=1 \
  "$PY" "$HERE/split_compile.py" --sections chunk
run grad_weight_layouts one_gpu "$PY" "$HERE/grad_weight_layouts.py"
run backward_memory one_gpu "$PY" "$HERE/backward_memory.py"
run gemm_throughput one_gpu "$PY" "$HERE/gemm_throughput.py"
run router_gate_comparison one_gpu "$PY" "$HERE/router_gate_comparison.py"
run lm_head_variants_qwen3_8b one_gpu "$PY" "$HERE/lm_head_variants.py" --cache-dir "$CACHE" --model qwen3_8b
if [ "$WITH_27B" = "--with-27b" ]; then
  run lm_head_variants_qwen3_5_27b one_gpu "$PY" "$HERE/lm_head_variants.py" --cache-dir "$CACHE" --model qwen3_5_27b
fi
run fsdp_fp32_grad "$PY" -m torch.distributed.run --nproc-per-node=2 "$HERE/fsdp_fp32_grad.py"
echo "$(date '+%T') done: $OUT"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "failed: ${FAILED[*]}"
  exit 1
fi
