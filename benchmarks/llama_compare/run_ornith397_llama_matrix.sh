#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/main/Documents/Claude"
LLAMA="${ROOT}/llama/llama.cpp/build/bin/llama-bench"
MODEL_DIR="/home/main/.krasis/models/Ornith-1.0-397B-GGUF"
MODEL="${MODEL_DIR}/deepreinforce-ai_Ornith-1.0-397B-Q4_K_M/deepreinforce-ai_Ornith-1.0-397B-Q4_K_M-00001-of-00007.gguf"
OUT_DIR="${ROOT}/krasis/benchmarks/llama_compare"
GPU_UUID="GPU-ece9afbc-ab6b-d1b9-7e7e-ad73769d6b5d"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

if [[ ! -x "${LLAMA}" ]]; then
  echo "missing llama-bench: ${LLAMA}" >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "missing model entry shard: ${MODEL}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

run_case() {
  local name="$1"
  shift
  local log="${OUT_DIR}/${STAMP}_ornith397_llama_${name}.log"

  {
    echo "# ${name}"
    echo "# started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# llama_commit=$(git -C "${ROOT}/llama/llama.cpp" rev-parse --short HEAD)"
    echo "# gpu_uuid=${GPU_UUID}"
    echo "# model=${MODEL}"
    echo "# command=CUDA_VISIBLE_DEVICES=${GPU_UUID} ${LLAMA} -m ${MODEL} $*"
    CUDA_VISIBLE_DEVICES="${GPU_UUID}" "${LLAMA}" -m "${MODEL}" "$@"
    echo "# finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } 2>&1 | tee "${log}"
}

# Ornith/Qwen3.5-397B has 60 MoE layers. n_cpu_moe=60 keeps all routed expert
# tensors on CPU; lower values let progressively more later MoE layers reside
# on GPU if VRAM admits them.
COMMON=(-r 3 -o jsonl -p 512,2048,8192 -n 0 -n 128 -t 64 -b 2048 -ub 512 -ngl 999 -mmp 1 -dev CUDA0)

run_case "fa0_kv_f16_ncmoe60" "${COMMON[@]}" -fa 0 -ctk f16 -ctv f16 -ncmoe 60
run_case "fa1_kv_f16_ncmoe60" "${COMMON[@]}" -fa 1 -ctk f16 -ctv f16 -ncmoe 60
run_case "fa1_kv_q8_ncmoe60"  "${COMMON[@]}" -fa 1 -ctk q8_0 -ctv q8_0 -ncmoe 60
run_case "fa1_kv_f16_ncmoe52" "${COMMON[@]}" -fa 1 -ctk f16 -ctv f16 -ncmoe 52
run_case "fa1_kv_f16_ncmoe44" "${COMMON[@]}" -fa 1 -ctk f16 -ctv f16 -ncmoe 44
run_case "fa1_kv_f16_ncmoe36" "${COMMON[@]}" -fa 1 -ctk f16 -ctv f16 -ncmoe 36
