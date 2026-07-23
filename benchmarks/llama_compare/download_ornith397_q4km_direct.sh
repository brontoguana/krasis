#!/usr/bin/env bash
set -euo pipefail

REPO="bartowski/deepreinforce-ai_Ornith-1.0-397B-GGUF"
SUBDIR="deepreinforce-ai_Ornith-1.0-397B-Q4_K_M"
ROOT="/home/main/.krasis/models/Ornith-1.0-397B-GGUF"
OUT_DIR="${ROOT}/${SUBDIR}"
CACHE_DIR="${ROOT}/.cache/huggingface/download/${SUBDIR}"
BASE_URL="https://huggingface.co/${REPO}/resolve/main/${SUBDIR}"
JOBS="${JOBS:-8}"
SEGMENT_BYTES="${SEGMENT_BYTES:-1073741824}"

mkdir -p "${OUT_DIR}"

download_ranges() {
  local url="$1"
  local part="$2"
  local total="$3"
  local have="$4"
  local prefix="$5"

  if (( have >= total )); then
    return 0
  fi

  local start="${have}"
  local seg_index=0
  local pids=()
  local segs=()
  local expected=()

  find "$(dirname "${prefix}")" -maxdepth 1 -type f -name "$(basename "${prefix}").seg.*" -delete

  while (( start < total )); do
    local end=$(( start + SEGMENT_BYTES - 1 ))
    if (( end >= total )); then
      end=$(( total - 1 ))
    fi
    local seg
    seg="$(printf '%s.seg.%05d' "${prefix}" "${seg_index}")"
    segs+=("${seg}")
    expected+=($(( end - start + 1 )))

    echo "  range ${start}-${end} -> $(basename "${seg}")"
    curl --fail --location --retry 20 --retry-delay 10 --retry-all-errors \
      --range "${start}-${end}" --output "${seg}" "${url}" &
    pids+=("$!")

    start=$(( end + 1 ))
    seg_index=$(( seg_index + 1 ))

    if (( ${#pids[@]} >= JOBS )); then
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
      pids=()
    fi
  done

  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  for i in "${!segs[@]}"; do
    local seg="${segs[$i]}"
    local got
    got="$(stat -c %s "${seg}")"
    if [[ "${got}" != "${expected[$i]}" ]]; then
      echo "segment size mismatch for $(basename "${seg}"): got ${got}, expected ${expected[$i]}" >&2
      exit 1
    fi
  done

  for seg in "${segs[@]}"; do
    cat "${seg}" >> "${part}"
  done
  find "$(dirname "${prefix}")" -maxdepth 1 -type f -name "$(basename "${prefix}").seg.*" -delete
}

for idx in 1 2 3 4 5 6 7; do
  shard="$(printf 'deepreinforce-ai_Ornith-1.0-397B-Q4_K_M-%05d-of-00007.gguf' "${idx}")"
  url="${BASE_URL}/${shard}"
  out="${OUT_DIR}/${shard}"
  part="${out}.part"

  if [[ -f "${out}" ]]; then
    echo "already complete: ${shard}"
    continue
  fi

  headers="$(curl -sS -L -I "${url}")"
  etag="$(printf '%s\n' "${headers}" | awk 'BEGIN{IGNORECASE=1} /^x-linked-etag:/ {gsub(/[\r\"]/,"",$2); print $2; exit}')"
  total="$(printf '%s\n' "${headers}" | awk 'BEGIN{IGNORECASE=1} /^x-linked-size:/ {gsub(/\r/,"",$2); print $2; exit}')"
  if [[ -z "${total}" ]]; then
    total="$(printf '%s\n' "${headers}" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub(/\r/,"",$2); value=$2} END{print value}')"
  fi
  if [[ -z "${total}" ]]; then
    echo "could not determine content length for ${shard}" >&2
    exit 1
  fi

  if [[ -n "${etag}" && ! -f "${part}" && -d "${CACHE_DIR}" ]]; then
    cached="$(find "${CACHE_DIR}" -maxdepth 1 -type f -name "*.${etag}.incomplete" -print -quit)"
    if [[ -n "${cached}" ]]; then
      echo "reusing cached partial for ${shard}: $(stat -c %s "${cached}") bytes"
      mv "${cached}" "${part}"
    fi
  fi

  have=0
  if [[ -f "${part}" ]]; then
    have="$(stat -c %s "${part}")"
  fi
  echo "downloading ${shard}: have=${have} total=${total} jobs=${JOBS}"
  download_ranges "${url}" "${part}" "${total}" "${have}" "${part}"
  final_size="$(stat -c %s "${part}")"
  if [[ "${final_size}" != "${total}" ]]; then
    echo "size mismatch for ${shard}: got ${final_size}, expected ${total}" >&2
    exit 1
  fi
  mv "${part}" "${out}"
done
