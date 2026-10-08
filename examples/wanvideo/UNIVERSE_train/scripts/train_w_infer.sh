#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVERSE_TRAIN_DIR="${UNIVERSE_TRAIN_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
REPO_ROOT="${REPO_ROOT:-"$(cd "${UNIVERSE_TRAIN_DIR}/../../.." && pwd)"}"
UNIVERSE_INFER_DIR="${UNIVERSE_INFER_DIR:-"${REPO_ROOT}/examples/wanvideo/UNIVERSE_infer"}"
CONFIG="${CONFIG:-${NAVSIM_TRAIN_CONFIG:-"${UNIVERSE_TRAIN_DIR}/configs/navsim_v1.yaml"}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-"${SCRIPT_DIR}/train_navsim_v1.sh"}"
INFER_ALL_SCRIPT="${INFER_ALL_SCRIPT:-"${UNIVERSE_INFER_DIR}/scripts/infer_all.sh"}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  export MAX_SCENES="${MAX_SCENES:-8}"
  export NUM_EPOCHS="${NUM_EPOCHS:-1}"
  export SAVE_STEPS="${SAVE_STEPS:-2}"
  export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"
  export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
  export OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/universe/train_navsim_v1_smoke}"
fi

if [ ! -f "${CONFIG}" ]; then
  echo "[UNIVERSE][train_w_infer] config not found: ${CONFIG}" >&2
  exit 1
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[UNIVERSE][train_w_infer] train script not found: ${TRAIN_SCRIPT}" >&2
  exit 1
fi
if [ ! -f "${INFER_ALL_SCRIPT}" ]; then
  echo "[UNIVERSE][train_w_infer] infer_all script not found: ${INFER_ALL_SCRIPT}" >&2
  exit 1
fi

export REPO_ROOT UNIVERSE_TRAIN_DIR UNIVERSE_INFER_DIR CONFIG
USER_EVAL_CKPT_KIND="${EVAL_CKPT_KIND:-}"
USER_AUTO_EVAL_CKPT_KIND="${AUTO_EVAL_CKPT_KIND:-}"
config_exports="$("${PYTHON:-python}" "${UNIVERSE_INFER_DIR}/scripts/load_yaml_config.py" "${CONFIG}")"
eval "${config_exports}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-"${OUTPUT_PATH}"}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${INFER_ALL_OUTPUT_ROOT:-"${OUTPUT_PATH}/infer_each_ckpt"}}"
EVAL_STATE_DIR="${EVAL_STATE_DIR:-"${OUTPUT_PATH}/infer_state"}"
TRAIN_W_INFER_POLL_SECONDS="${TRAIN_W_INFER_POLL_SECONDS:-30}"
EVAL_CKPT_STABLE_SECONDS="${EVAL_CKPT_STABLE_SECONDS:-20}"
if [[ -n "${USER_EVAL_CKPT_KIND}" ]]; then
  EVAL_CKPT_KIND="${USER_EVAL_CKPT_KIND}"
elif [[ -n "${USER_AUTO_EVAL_CKPT_KIND}" ]]; then
  EVAL_CKPT_KIND="${USER_AUTO_EVAL_CKPT_KIND}"
else
  EVAL_CKPT_KIND="all"
fi
if [[ -n "${USER_AUTO_EVAL_CKPT_KIND}" ]]; then
  AUTO_EVAL_CKPT_KIND="${USER_AUTO_EVAL_CKPT_KIND}"
else
  AUTO_EVAL_CKPT_KIND="${EVAL_CKPT_KIND}"
fi
STOP_TRAIN_ON_EVAL_FAIL="${STOP_TRAIN_ON_EVAL_FAIL:-0}"
EVAL_DURING_TRAINING="${EVAL_DURING_TRAINING:-auto}"

case "${EVAL_CKPT_KIND}" in
  ema|raw|all) ;;
  *)
    echo "[UNIVERSE][train_w_infer] EVAL_CKPT_KIND must be ema, raw, or all; got ${EVAL_CKPT_KIND}" >&2
    exit 1
    ;;
esac

RUN_NAVSIM="${RUN_NAVSIM:-1}"
RUN_NUSCENES="${RUN_NUSCENES:-1}"
RUN_B2D="${RUN_B2D:-1}"
INFER_ALL_MODE="${INFER_ALL_MODE:-sequential}"
if [[ -z "${EXTERNAL_EVAL:-}" ]]; then
  if [[ "${AUTO_EVAL:-0}" == "1" ]]; then
    EXTERNAL_EVAL=0
  else
    EXTERNAL_EVAL=1
  fi
fi

case "${EVAL_DURING_TRAINING}" in
  0|1|auto) ;;
  *)
    echo "[UNIVERSE][train_w_infer] EVAL_DURING_TRAINING must be 0, 1, or auto; got ${EVAL_DURING_TRAINING}" >&2
    exit 1
    ;;
esac

_node_rank="${NODE_RANK:-${RANK:-${MLP_ROLE_INDEX:-0}}}"
if [[ "${TRAIN_W_INFER_ON_ALL_NODES:-0}" != "1" && "${_node_rank}" != "0" ]]; then
  echo "[UNIVERSE][train_w_infer] node_rank=${_node_rank}; run training only on nonzero node."
  exec bash "${TRAIN_SCRIPT}"
fi

mkdir -p "${CHECKPOINT_DIR}" "${EVAL_OUTPUT_ROOT}" "${EVAL_STATE_DIR}"

_log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [UNIVERSE][train_w_infer] $*"
}

_file_size() {
  stat -c %s "$1" 2>/dev/null || stat -f %z "$1" 2>/dev/null
}

_count_cuda_devices() {
  local devices="${1// /}"
  if [[ -z "${devices}" ]]; then
    echo ""
    return 0
  fi
  awk -F',' '{print NF}' <<<"${devices}"
}

_has_eval_cuda_override() {
  [[ -n "${EVAL_CUDA_VISIBLE_DEVICES:-}" ]] ||
    [[ -n "${NAVSIM_CUDA_VISIBLE_DEVICES:-}" ]] ||
    [[ -n "${NUSCENES_CUDA_VISIBLE_DEVICES:-}" ]] ||
    [[ -n "${B2D_CUDA_VISIBLE_DEVICES:-}" ]]
}

_list_ckpts() {
  if [ ! -d "${CHECKPOINT_DIR}" ]; then
    return 0
  fi
  case "${EVAL_CKPT_KIND}" in
    ema)
      find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "*-ema.safetensors" | sort -V
      ;;
    raw)
      find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "*.safetensors" ! -name "*-ema.safetensors" | sort -V
      ;;
    all)
      find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name "*.safetensors" | sort -V
      ;;
  esac
}

_ckpt_is_stable() {
  local ckpt="$1"
  local size_before size_after
  size_before="$(_file_size "${ckpt}")" || return 1
  if [[ "${EVAL_CKPT_STABLE_SECONDS}" != "0" ]]; then
    sleep "${EVAL_CKPT_STABLE_SECONDS}"
  fi
  size_after="$(_file_size "${ckpt}")" || return 1
  [[ "${size_before}" == "${size_after}" && "${size_after}" != "0" ]]
}

eval_status=0
train_pid=""

_run_eval_for_ckpt() {
  local ckpt="$1"
  local ckpt_file ckpt_id marker_done marker_failed marker_running out_root log_dir log_file rc
  ckpt_file="$(basename "${ckpt}")"
  ckpt_id="${ckpt_file%.safetensors}"
  marker_done="${EVAL_STATE_DIR}/${ckpt_id}.done"
  marker_failed="${EVAL_STATE_DIR}/${ckpt_id}.failed"
  marker_running="${EVAL_STATE_DIR}/${ckpt_id}.running"
  out_root="${EVAL_OUTPUT_ROOT}/${ckpt_id}"
  log_dir="${out_root}/logs"
  log_file="${log_dir}/infer_all.log"

  if [ -f "${marker_done}" ]; then
    return 0
  fi
  if [[ "${EVAL_RETRY_FAILED:-0}" != "1" && -f "${marker_failed}" ]]; then
    return 0
  fi
  if [ -f "${marker_running}" ]; then
    return 0
  fi
  if ! _ckpt_is_stable "${ckpt}"; then
    _log "checkpoint not stable yet, skip this scan: ${ckpt}"
    return 0
  fi

  mkdir -p "${log_dir}"
  rm -f "${marker_failed}"
  date '+%Y-%m-%d %H:%M:%S' >"${marker_running}"
  _log "start eval ckpt=${ckpt} output=${out_root}"

  set +e
  (
    set -euo pipefail
    export REPO_ROOT UNIVERSE_INFER_DIR
    export FULL_CKPT="${ckpt}"
    export INFER_ALL_OUTPUT_ROOT="${out_root}"
    export INFER_ALL_LOG_DIR="${log_dir}"
    export RUN_NAVSIM RUN_NUSCENES RUN_B2D INFER_ALL_MODE

    export RANK="${EVAL_RANK:-0}"
    export NODE_RANK="${EVAL_NODE_RANK:-0}"
    export NNODES="${EVAL_NUM_NODES:-1}"
    export NUM_NODES="${EVAL_NUM_NODES:-1}"
    export MASTER_ADDR="${EVAL_MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${EVAL_MASTER_PORT:-29650}"
    unset TOTAL_PROCESSES

    if [[ -n "${EVAL_CUDA_VISIBLE_DEVICES:-}" ]]; then
      export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
      eval_gpus="${EVAL_GPUS:-$(_count_cuda_devices "${EVAL_CUDA_VISIBLE_DEVICES}")}"
      if [[ -n "${eval_gpus}" ]]; then
        export GPUS="${eval_gpus}"
        export GPUS_PER_NODE="${eval_gpus}"
        export NUM_GPUS="${eval_gpus}"
      fi
    fi
    if [[ -n "${EVAL_GPUS:-}" ]]; then
      export GPUS="${EVAL_GPUS}"
      export GPUS_PER_NODE="${EVAL_GPUS}"
      export NUM_GPUS="${EVAL_GPUS}"
    fi
    if [[ -n "${EVAL_SMOKE_TEST:-}" ]]; then
      export SMOKE_TEST="${EVAL_SMOKE_TEST}"
    fi

    bash "${INFER_ALL_SCRIPT}"
  ) >"${log_file}" 2>&1
  rc=$?
  set -e

  rm -f "${marker_running}"
  if [ "${rc}" -eq 0 ]; then
    date '+%Y-%m-%d %H:%M:%S' >"${marker_done}"
    _log "eval done ckpt=${ckpt} log=${log_file}"
  else
    eval_status="${rc}"
    date '+%Y-%m-%d %H:%M:%S' >"${marker_failed}"
    _log "eval failed rc=${rc} ckpt=${ckpt}; last log lines:"
    tail -n 80 "${log_file}" >&2 || true
    if [[ "${STOP_TRAIN_ON_EVAL_FAIL}" == "1" && -n "${train_pid}" ]]; then
      kill "${train_pid}" >/dev/null 2>&1 || true
    fi
  fi
}

_scan_and_eval() {
  local ckpt
  while IFS= read -r ckpt; do
    [ -n "${ckpt}" ] || continue
    _run_eval_for_ckpt "${ckpt}"
  done < <(_list_ckpts)
}

_cleanup() {
  if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" >/dev/null 2>&1; then
    _log "stopping training pid=${train_pid}"
    kill "${train_pid}" >/dev/null 2>&1 || true
  fi
}
trap _cleanup INT TERM

if [[ "${EVAL_DURING_TRAINING}" == "auto" ]]; then
  if _has_eval_cuda_override; then
    EVAL_DURING_TRAINING_RESOLVED=1
  else
    EVAL_DURING_TRAINING_RESOLVED=0
  fi
else
  EVAL_DURING_TRAINING_RESOLVED="${EVAL_DURING_TRAINING}"
fi

_log "train command: bash ${TRAIN_SCRIPT}"
_log "watch checkpoints: dir=${CHECKPOINT_DIR} kind=${EVAL_CKPT_KIND}"
_log "eval datasets: navsim=${RUN_NAVSIM} nuscenes=${RUN_NUSCENES} b2d=${RUN_B2D} mode=${INFER_ALL_MODE} in_process_auto_eval=${AUTO_EVAL:-0} external_eval=${EXTERNAL_EVAL} during_training=${EVAL_DURING_TRAINING_RESOLVED}"
if [[ "${EXTERNAL_EVAL}" != "1" ]]; then
  _log "external watcher eval disabled; train script will handle in-process multi-rank auto eval."
elif [[ "${EVAL_DURING_TRAINING_RESOLVED}" != "1" ]]; then
  _log "no eval CUDA override detected; defer checkpoint eval until training exits. Set EVAL_CUDA_VISIBLE_DEVICES to evaluate during training."
fi

(
  set -euo pipefail
  if [[ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
    train_gpus="${TRAIN_GPUS:-$(_count_cuda_devices "${TRAIN_CUDA_VISIBLE_DEVICES}")}"
    if [[ -n "${train_gpus}" ]]; then
      export GPUS="${train_gpus}"
      export GPUS_PER_NODE="${train_gpus}"
      export NUM_GPUS="${train_gpus}"
    fi
  elif [[ -n "${TRAIN_GPUS:-}" ]]; then
    export GPUS="${TRAIN_GPUS}"
    export GPUS_PER_NODE="${TRAIN_GPUS}"
    export NUM_GPUS="${TRAIN_GPUS}"
  fi
  bash "${TRAIN_SCRIPT}"
) &
train_pid="$!"

while kill -0 "${train_pid}" >/dev/null 2>&1; do
  if [[ "${EXTERNAL_EVAL}" == "1" && "${EVAL_DURING_TRAINING_RESOLVED}" == "1" ]]; then
    _scan_and_eval
  fi
  sleep "${TRAIN_W_INFER_POLL_SECONDS}"
done

set +e
wait "${train_pid}"
train_status=$?
set -e
train_pid=""

if [[ "${EXTERNAL_EVAL}" == "1" ]]; then
  _log "training exited rc=${train_status}; final checkpoint scan"
  _scan_and_eval
else
  _log "training exited rc=${train_status}; external checkpoint scan disabled"
fi

if [ "${train_status}" -ne 0 ]; then
  exit "${train_status}"
fi
exit "${eval_status}"
