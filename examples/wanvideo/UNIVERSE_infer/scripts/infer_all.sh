#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVERSE_INFER_DIR="${UNIVERSE_INFER_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
REPO_ROOT="${REPO_ROOT:-"$(cd "${UNIVERSE_INFER_DIR}/../../.." && pwd)"}"

RUN_NAVSIM="${RUN_NAVSIM:-1}"
RUN_NUSCENES="${RUN_NUSCENES:-1}"
RUN_B2D="${RUN_B2D:-1}"

INFER_ALL_MODE="${INFER_ALL_MODE:-sequential}"
INFER_ALL_OUTPUT_ROOT="${INFER_ALL_OUTPUT_ROOT:-${REPO_ROOT}/outputs/universe/all_infer}"
INFER_ALL_LOG_DIR="${INFER_ALL_LOG_DIR:-${INFER_ALL_OUTPUT_ROOT}/logs}"

if [[ "${INFER_ALL_MODE}" != "sequential" && "${INFER_ALL_MODE}" != "parallel" ]]; then
  echo "[UNIVERSE][all] INFER_ALL_MODE must be sequential or parallel, got: ${INFER_ALL_MODE}" >&2
  exit 1
fi

if [[ "${RUN_NAVSIM}" != "1" && "${RUN_NUSCENES}" != "1" && "${RUN_B2D}" != "1" ]]; then
  echo "[UNIVERSE][all] no dataset selected; set at least one of RUN_NAVSIM/RUN_NUSCENES/RUN_B2D=1" >&2
  exit 1
fi

mkdir -p "${INFER_ALL_OUTPUT_ROOT}"
if [[ "${INFER_ALL_MODE}" == "parallel" ]]; then
  mkdir -p "${INFER_ALL_LOG_DIR}"
  if [[
    ( "${RUN_NAVSIM}" == "1" && -z "${NAVSIM_CUDA_VISIBLE_DEVICES:-}" ) ||
    ( "${RUN_NUSCENES}" == "1" && -z "${NUSCENES_CUDA_VISIBLE_DEVICES:-}" ) ||
    ( "${RUN_B2D}" == "1" && -z "${B2D_CUDA_VISIBLE_DEVICES:-}" )
  ]]; then
    echo "[UNIVERSE][all][warn] parallel mode shares visible GPUs unless NAVSIM_CUDA_VISIBLE_DEVICES, NUSCENES_CUDA_VISIBLE_DEVICES, and B2D_CUDA_VISIBLE_DEVICES are set." >&2
  fi
fi

_apply_prefixed_env() {
  local prefix="$1"
  local key var
  # Per-dataset overrides such as NAVSIM_FULL_CKPT or B2D_OUTPUT_DIR are
  # mapped back to common launcher variables before each child script runs.
  for var in \
    CUDA_VISIBLE_DEVICES \
    GPUS \
    GPUS_PER_NODE \
    NUM_GPUS \
    NNODES \
    NUM_NODES \
    RANK \
    NODE_RANK \
    MASTER_ADDR \
    MASTER_PORT \
    SMOKE_TEST \
    MAX_SCENES \
    MAX_EVAL_TOKENS \
    SAVE_VIZ \
    NUM_INFERENCE_STEPS \
    CFG_SCALE \
    SEED \
    LOCAL_MODEL_PATH \
    FULL_CKPT \
    OUTPUT_DIR; do
    key="${prefix}_${var}"
    if [[ -n "${!key+x}" ]]; then
      export "${var}=${!key}"
    fi
  done
}

_prepare_dataset_env() {
  local prefix="$1"
  local default_output_dir="$2"
  local default_master_port="$3"
  local config_key="${prefix}_CONFIG"

  export REPO_ROOT UNIVERSE_INFER_DIR
  unset CONFIG
  unset TOTAL_PROCESSES

  # Each dataset gets its own default output and distributed port, while still
  # allowing explicit overrides from the parent environment.
  export OUTPUT_DIR="${default_output_dir}"
  if [[ "${INFER_ALL_MODE}" == "parallel" ]]; then
    export MASTER_PORT="${default_master_port}"
  fi
  if [[ -n "${!config_key+x}" ]]; then
    export CONFIG="${!config_key}"
  fi

  _apply_prefixed_env "${prefix}"
}

_run_navsim() (
  set -euo pipefail
  _prepare_dataset_env "NAVSIM" "${INFER_ALL_OUTPUT_ROOT}/navsim_v1" "29610"
  echo "[UNIVERSE][all] NavSIM output=${OUTPUT_DIR}"
  bash "${SCRIPT_DIR}/eval_navsim_v1.sh"
)

_run_nuscenes() (
  set -euo pipefail
  _prepare_dataset_env "NUSCENES" "${INFER_ALL_OUTPUT_ROOT}/nuscenes" "29620"
  echo "[UNIVERSE][all] nuScenes output=${OUTPUT_DIR}"
  bash "${SCRIPT_DIR}/infer_nuscenes.sh"
)

_run_b2d() (
  set -euo pipefail
  _prepare_dataset_env "B2D" "${INFER_ALL_OUTPUT_ROOT}/bench2drive" "29630"
  echo "[UNIVERSE][all] Bench2Drive output=${OUTPUT_DIR}"
  bash "${SCRIPT_DIR}/infer_bench2drive.sh"
)

names=()
fns=()
if [[ "${RUN_NAVSIM}" == "1" ]]; then names+=("navsim"); fns+=("_run_navsim"); fi
if [[ "${RUN_NUSCENES}" == "1" ]]; then names+=("nuscenes"); fns+=("_run_nuscenes"); fi
if [[ "${RUN_B2D}" == "1" ]]; then names+=("bench2drive"); fns+=("_run_b2d"); fi

if [[ "${INFER_ALL_MODE}" == "sequential" ]]; then
  # Sequential mode keeps GPU ownership simple; parallel mode below isolates
  # logs so one dataset failure does not hide the others.
  for i in "${!names[@]}"; do
    echo "[UNIVERSE][all] starting ${names[$i]}"
    "${fns[$i]}"
    echo "[UNIVERSE][all] finished ${names[$i]}"
  done
  exit 0
fi

pids=()
logs=()

_cleanup_parallel_jobs() {
  local pid
  for pid in "${pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap _cleanup_parallel_jobs INT TERM

for i in "${!names[@]}"; do
  log_path="${INFER_ALL_LOG_DIR}/${names[$i]}.log"
  logs+=("${log_path}")
  echo "[UNIVERSE][all] starting ${names[$i]} in background; log=${log_path}"
  ( "${fns[$i]}" ) >"${log_path}" 2>&1 &
  pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[UNIVERSE][all] finished ${names[$i]}"
  else
    status=1
    echo "[UNIVERSE][all][error] ${names[$i]} failed; last log lines:" >&2
    tail -n 80 "${logs[$i]}" >&2 || true
  fi
done

exit "${status}"
