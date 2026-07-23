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
  local log="${OUT_DIR}/${STAMP}_ornith397_llama_best_${name}.log"

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

COMMON=(-r 3 -o jsonl -t 64 -b 2048 -ub 512 -ngl 999 -mmp 1 -dev CUDA0 -fa 1 -ctk f16 -ctv f16)

# Gate from maximum GPU MoE residency downward. Lower n_cpu_moe should be better
# if it fits, because more expert layers stay on the RTX PRO 6000.
for n_cpu_moe in 0 8 16 24 32 40 48 56 60; do
  run_case "gate_ncmoe${n_cpu_moe}" "${COMMON[@]}" -p 128 -n 16 -ncmoe "${n_cpu_moe}" || true
done

# Follow-up full rows can be run manually for the best fitting n_cpu_moe:
#   run_case "full_ncmoeN" "${COMMON[@]}" -p 512,2048,8192 -n 0 -n 128 -ncmoe N
