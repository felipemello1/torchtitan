#!/bin/bash
# Run the repros for ../README.md from this folder.
# CPU repros run anywhere; the GPU repros need one CUDA GPU (the README's numbers are from one GB300).
# Usage: ./run_all.sh [cpu|gpu|all]   (default: all)
set -u
cd "$(dirname "$0")"
MODE=${1:-all}

run() {
    echo "== $*"
    python "$@" 2>&1 | grep -v -i "warn" | tail -20
}

if [ "$MODE" = cpu ] || [ "$MODE" = all ]; then
    for f in 03_make_fx_tensor_constant.py 05c_shared_norm_call_sites_cpu.py 06_hop_outer_attr_store.py \
             07_autograd_fn_param_base.py 22_cache_under_fake_tensors.py 23_make_fx_int_specialization.py \
             27_regional_partition_scaling.py; do
        run "$f"
    done
    run 04_sac_wrapped_region_recompile.py cold --cpu
    run 04_sac_wrapped_region_recompile.py warm --cpu
fi

if [ "$MODE" = gpu ] || [ "$MODE" = all ]; then
    for f in 01_mix_order_guards.py 02_cache_ignores_fake.py 05a_grad_mode_guard.py 05b_shared_norm_call_sites.py \
             05d_size1_specialization.py 08_opaque_custom_op.py 09_symbolic_hidden_dim.py 10a_symbolic_small_dim.py \
             10b_small_k_in_tile.py 11_extern_gemv.py 12_inline_recompute.py 13_cat_lowering.py \
             14a_masked_loads_symbolic_T.py 14b_standalone_masked_kernels.py 15_int64_indexing_symbolic.py \
             16_sinkhorn_transposed_reads.py 17_complex_ops.py 20_fma_bitwise.py 21_maybe_mark_dynamic_traced.py; do
        run "$f"
    done
    run 04_sac_wrapped_region_recompile.py cold
    run 04_sac_wrapped_region_recompile.py warm
    # Request 17: Inductor prints the complex-codegen warning only on a cold compile.
    echo "== 17_complex_ops.py, caches disabled"
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python 17_complex_ops.py 2>&1 | grep "inductor warnings"
    # Request 20: bitwise once FP contraction is off; TRITON_DEFAULT_FP_FUSION has no effect.
    for e in 0 1; do
        echo "== 20_fma_bitwise.py EMULATE_PRECISION_CASTS=$e"
        EMULATE_PRECISION_CASTS=$e TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python 20_fma_bitwise.py 2>&1 | grep -v -i "warn" | tail -4
    done
    echo "== 20_fma_bitwise.py TRITON_DEFAULT_FP_FUSION=0"
    TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TRITON_DEFAULT_FP_FUSION=0 python 20_fma_bitwise.py 2>&1 | grep -v -i "warn" | tail -4
fi
