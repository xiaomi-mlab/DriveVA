#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVERSE_INFER_DIR="${UNIVERSE_INFER_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
REPO_ROOT="${REPO_ROOT:-"$(cd "${UNIVERSE_INFER_DIR}/../../.." && pwd)"}"
CONFIG="${CONFIG:-${B2D_INFER_CONFIG:-"${UNIVERSE_INFER_DIR}/configs/bench2drive.yaml"}}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  export MAX_SCENES="${MAX_SCENES:-8}"
  export SAVE_VIZ="${SAVE_VIZ:-0}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/universe/bench2drive_smoke}"
fi

if [ ! -f "${CONFIG}" ]; then
  echo "[UNIVERSE][b2d] config not found: ${CONFIG}" >&2
  exit 1
fi

export REPO_ROOT UNIVERSE_INFER_DIR CONFIG
config_exports="$("${PYTHON:-python}" "${UNIVERSE_INFER_DIR}/scripts/load_yaml_config.py" "${CONFIG}")"
eval "${config_exports}"
# shellcheck source=/dev/null
source "${UNIVERSE_INFER_DIR}/scripts/distributed_env.sh"
UNIVERSE_setup_distributed_env

export PYTHONPATH="${REPO_ROOT}:${UNIVERSE_INFER_DIR}:${PYTHONPATH:-}"

mkdir -p "${UNIVERSE_TMPDIR}"
export TMPDIR="${TMPDIR:-${UNIVERSE_TMPDIR}}"

if [ ! -d "${B2D_DATA_ROOT}" ]; then
  echo "[UNIVERSE][b2d] B2D_DATA_ROOT not found: ${B2D_DATA_ROOT}" >&2
  exit 1
fi
if [ ! -f "${B2D_ANN_FILE}" ]; then
  echo "[UNIVERSE][b2d] B2D_ANN_FILE not found: ${B2D_ANN_FILE}" >&2
  exit 1
fi

args=(
  --data_root "${B2D_DATA_ROOT}"
  --ann_file "${B2D_ANN_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --local_model_path "${LOCAL_MODEL_PATH}"
  --full_ckpt "${FULL_CKPT}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --cfg_scale "${CFG_SCALE}"
  --seed "${SEED}"
  --num_history_frames "${NUM_HISTORY_FRAMES}"
  --num_future_frames "${NUM_FUTURE_FRAMES}"
  --model_future_frames "${MODEL_FUTURE_FRAMES}"
  --target_fps "${TARGET_FPS}"
  --original_fps "${B2D_ORIGINAL_FPS}"
  --frame_interval "${B2D_FRAME_INTERVAL}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --command_yaw_threshold_deg "${COMMAND_YAW_THRESHOLD_DEG}"
  --viz_max_scenes "${VIZ_MAX_SCENES}"
  --viz_plot_height "${VIZ_PLOT_HEIGHT}"
  --debug_prompt_steps "${DEBUG_PROMPT_STEPS}"
  --projection_debug_steps "${PROJECTION_DEBUG_STEPS}"
  --projection_debug_token "${PROJECTION_DEBUG_TOKEN}"
  --distributed
)

if [[ -n "${MAX_SCENES:-}" ]]; then args+=(--max_scenes "${MAX_SCENES}"); fi
if [[ "${USE_STRUCTURED_PROMPT:-0}" == "1" ]]; then args+=(--use_structured_prompt); fi
if [[ "${USE_FOCAL_UNIFICATION:-0}" == "1" ]]; then args+=(--use_focal_unification); fi
if [[ -n "${FOCAL_TARGET_PX:-}" ]]; then args+=(--focal_target_px "${FOCAL_TARGET_PX}"); fi
if [[ "${SAVE_VIZ:-0}" == "1" ]]; then args+=(--save_viz); else args+=(--no_save_viz); fi
if [[ "${USE_MIXED_LATENT_ATTENTION_MASK:-1}" == "1" ]]; then args+=(--use_mixed_latent_attention_mask); else args+=(--no_use_mixed_latent_attention_mask); fi
if [[ -n "${INFER_OUTPUT_MODE:-}" ]]; then args+=(--infer_output_mode "${INFER_OUTPUT_MODE}"); fi
if [[ "${INFER_TRAJECTORY_ONLY:-0}" == "1" ]]; then args+=(--infer_trajectory_only); else args+=(--no_infer_trajectory_only); fi
if [[ "${SAVE_PROJECTED_TRAJ_IMAGE:-1}" == "1" ]]; then args+=(--save_projected_traj_image); else args+=(--no_save_projected_traj_image); fi
if [[ "${COMPUTE_PLANNING_METRICS:-1}" == "1" ]]; then args+=(--compute_planning_metrics); else args+=(--no_compute_planning_metrics); fi
if [[ "${PRINT_ERRORS:-1}" == "1" ]]; then args+=(--print_errors); else args+=(--no_print_errors); fi
if [[ "${SHOW_EVAL_PROGRESS:-1}" == "1" ]]; then args+=(--show_eval_progress); else args+=(--no_show_eval_progress); fi
if [[ "${SHOW_DENOISE_PROGRESS:-0}" == "1" ]]; then args+=(--show_denoise_progress); fi
if [[ "${SHOW_VAE_PROGRESS:-0}" == "1" ]]; then args+=(--show_vae_progress); fi

echo "[UNIVERSE][b2d] config=${CONFIG}"
echo "[UNIVERSE][b2d] ckpt=${FULL_CKPT}"
echo "[UNIVERSE][b2d] model=${LOCAL_MODEL_PATH}"
echo "[UNIVERSE][b2d] data=${B2D_DATA_ROOT} ann=${B2D_ANN_FILE}"
echo "[UNIVERSE][b2d] eval: history=${NUM_HISTORY_FRAMES} future=${NUM_FUTURE_FRAMES} model_future=${MODEL_FUTURE_FRAMES} steps=${NUM_INFERENCE_STEPS}"
echo "[UNIVERSE][b2d] focal: enabled=${USE_FOCAL_UNIFICATION:-0} target=${FOCAL_TARGET_PX:-}"
echo "[UNIVERSE][b2d] output=${OUTPUT_DIR} save_viz=${SAVE_VIZ} max_scenes=${MAX_SCENES:-all} output_mode=${INFER_OUTPUT_MODE:-legacy} infer_trajectory_only=${INFER_TRAJECTORY_ONLY:-0} mixed_latent_attention=${USE_MIXED_LATENT_ATTENTION_MASK:-1}"
echo "[UNIVERSE][b2d] smoke: enabled=${SMOKE_TEST:-0}"

UNIVERSE_launch_torchrun \
  "${REPO_ROOT}/examples/wanvideo/UNIVERSE_infer/infer_bench2drive.py" \
  "${args[@]}"
