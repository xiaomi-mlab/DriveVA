#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVERSE_INFER_DIR="${UNIVERSE_INFER_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
REPO_ROOT="${REPO_ROOT:-"$(cd "${UNIVERSE_INFER_DIR}/../../.." && pwd)"}"
CONFIG="${CONFIG:-${NUSCENES_INFER_CONFIG:-"${UNIVERSE_INFER_DIR}/configs/nuscenes.yaml"}}"

if [ ! -f "${CONFIG}" ]; then
  echo "[UNIVERSE] config not found: ${CONFIG}" >&2
  exit 1
fi

export REPO_ROOT UNIVERSE_INFER_DIR CONFIG
config_exports="$("${PYTHON:-python}" "${UNIVERSE_INFER_DIR}/scripts/load_yaml_config.py" "${CONFIG}")"
eval "${config_exports}"

# shellcheck source=/dev/null
source "${UNIVERSE_INFER_DIR}/scripts/distributed_env.sh"
UNIVERSE_setup_distributed_env

export PYTHONPATH="${REPO_ROOT}:${UNIVERSE_INFER_DIR}:${REPO_ROOT}/third_party:${REPO_ROOT}/third_party/nuscenes-devkit/python-sdk:${PYTHONPATH:-}"

if [ -z "${NUSCENES_DATAROOT:-}" ]; then
  echo "[UNIVERSE] set NUSCENES_DATAROOT or NUSCENES_EVAL_DATAROOT" >&2
  exit 1
fi

args=(
  --nuscenes_dataroot "${NUSCENES_DATAROOT}"
  --nuscenes_version "${NUSCENES_VERSION}"
  --split "${NUSCENES_SPLIT}"
  --camera_name "${CAMERA_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --local_model_path "${LOCAL_MODEL_PATH}"
  --full_ckpt "${FULL_CKPT}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --cfg_scale "${CFG_SCALE}"
  --seed "${SEED}"
  --debug_prompt_steps "${DEBUG_PROMPT_STEPS}"
  --trajectory_condition_mode "${TRAJECTORY_CONDITION_MODE}"
  --num_history_frames "${NUM_HISTORY_FRAMES}"
  --num_future_frames "${NUM_FUTURE_FRAMES}"
  --model_future_frames "${MODEL_FUTURE_FRAMES}"
  --scene_future_extra_seconds "${SCENE_FUTURE_EXTRA_SECONDS}"
  --target_fps "${TARGET_FPS}"
  --metric_horizons_s "${METRIC_HORIZONS_S}"
  --ego_box_length_m "${EGO_BOX_LENGTH_M}"
  --ego_box_width_m "${EGO_BOX_WIDTH_M}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
)

if [[ -n "${MAX_SCENES:-}" ]]; then args+=(--max_scenes "${MAX_SCENES}"); fi
if [[ -n "${MAX_EVAL_TOKENS:-}" ]]; then args+=(--max_eval_tokens "${MAX_EVAL_TOKENS}"); fi
if [[ -n "${POLICY_ANNO_JSON:-}" ]]; then
  if [[ -f "${POLICY_ANNO_JSON}" ]]; then
    args+=(--policy_anno_json "${POLICY_ANNO_JSON}")
  else
    echo "[UNIVERSE][warn] policy anno json not found, evaluating unfiltered nuScenes tokens: ${POLICY_ANNO_JSON}" >&2
  fi
fi
if [[ "${SAVE_VIZ:-0}" == "1" ]]; then
  args+=(--save_viz --viz_max_tokens "${VIZ_MAX_TOKENS}")
  if [[ -n "${VIZ_DIR:-}" ]]; then args+=(--viz_dir "${VIZ_DIR}"); fi
fi
if [[ "${USE_MIXED_LATENT_ATTENTION_MASK:-1}" == "1" ]]; then args+=(--use_mixed_latent_attention_mask); else args+=(--no_use_mixed_latent_attention_mask); fi
if [[ "${USE_FOCAL_UNIFICATION:-0}" == "1" ]]; then args+=(--use_focal_unification); fi
if [[ -n "${FOCAL_TARGET_PX:-}" ]]; then args+=(--focal_target_px "${FOCAL_TARGET_PX}"); fi
if [[ -n "${INFER_OUTPUT_MODE:-}" ]]; then args+=(--infer_output_mode "${INFER_OUTPUT_MODE}"); fi
if [[ "${INFER_TRAJECTORY_ONLY:-0}" == "1" ]]; then args+=(--infer_trajectory_only); else args+=(--no_infer_trajectory_only); fi
if [[ "${SHOW_DENOISE_PROGRESS:-0}" == "1" ]]; then args+=(--show_denoise_progress); fi
if [[ "${SHOW_VAE_PROGRESS:-0}" == "1" ]]; then args+=(--show_vae_progress); fi

echo "[UNIVERSE] config=${CONFIG}"
echo "[UNIVERSE] ckpt=${FULL_CKPT}"
echo "[UNIVERSE] model=${LOCAL_MODEL_PATH}"
echo "[UNIVERSE] nuscenes: dataroot=${NUSCENES_DATAROOT} version=${NUSCENES_VERSION} split=${NUSCENES_SPLIT}"
echo "[UNIVERSE] eval: history=${NUM_HISTORY_FRAMES} future=${NUM_FUTURE_FRAMES} model_future=${MODEL_FUTURE_FRAMES} steps=${NUM_INFERENCE_STEPS} traj_cond=${TRAJECTORY_CONDITION_MODE}"
echo "[UNIVERSE] focal: enabled=${USE_FOCAL_UNIFICATION:-0} target=${FOCAL_TARGET_PX:-}"
echo "[UNIVERSE] infer: output_mode=${INFER_OUTPUT_MODE:-legacy} infer_trajectory_only=${INFER_TRAJECTORY_ONLY:-0} mixed_latent_attention=${USE_MIXED_LATENT_ATTENTION_MASK:-1}"
echo "[UNIVERSE] metrics: horizons=${METRIC_HORIZONS_S} policy_anno=${POLICY_ANNO_JSON:-none}"

UNIVERSE_launch_torchrun \
  "${REPO_ROOT}/examples/wanvideo/UNIVERSE_infer/infer_nuscenes.py" \
  "${args[@]}"
