#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVERSE_INFER_DIR="${UNIVERSE_INFER_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
REPO_ROOT="${REPO_ROOT:-"$(cd "${UNIVERSE_INFER_DIR}/../../.." && pwd)"}"
CONFIG="${CONFIG:-${NAVSIM_INFER_CONFIG:-"${UNIVERSE_INFER_DIR}/configs/navsim_v1.yaml"}}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  export MAX_EVAL_TOKENS="${MAX_EVAL_TOKENS:-64}"
  export PRINT_ALIGNMENT_PARAMS="${PRINT_ALIGNMENT_PARAMS:-1}"
  export SAVE_VIZ="${SAVE_VIZ:-0}"
  export INFER_TRAJECTORY_ONLY="${INFER_TRAJECTORY_ONLY:-1}"
  export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/universe/navsim_v1_smoke}"
fi

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
export NUPLAN_MAPS_ROOT
export NUPLAN_DATA_ROOT

mkdir -p "${UNIVERSE_TMPDIR}"
export TMPDIR="${TMPDIR:-${UNIVERSE_TMPDIR}}"

args=(
  --repo_root "${REPO_ROOT}"
  --navsim_log_path "${NAVSIM_LOG_PATH}"
  --sensor_blobs_path "${NAVSIM_SENSOR_BLOBS_PATH}"
  --metric_cache_path "${NAVSIM_METRIC_CACHE_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --local_model_path "${LOCAL_MODEL_PATH}"
  --full_ckpt "${FULL_CKPT}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --cfg_scale "${CFG_SCALE}"
  --seed "${SEED}"
  --num_history_frames "${NUM_HISTORY_FRAMES}"
  --num_future_frames "${NUM_FUTURE_FRAMES}"
  --model_future_frames "${MODEL_FUTURE_FRAMES}"
  --scene_future_extra_seconds "${SCENE_FUTURE_EXTRA_SECONDS}"
  --pdm_num_poses "${PDM_NUM_POSES}"
  --pdm_interval_length "${PDM_INTERVAL_LENGTH}"
  --target_fps "${TARGET_FPS}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --trajectory_condition_mode "${TRAJECTORY_CONDITION_MODE}"
  --traffic_agents_policy "${TRAFFIC_AGENTS_POLICY}"
  --viz_total_tokens "${VIZ_TOTAL_TOKENS}"
)

if [[ -n "${LOG_NAMES:-}" ]]; then args+=(--log_names "${LOG_NAMES}"); fi
if [[ -n "${SCENE_FILTER_YAML:-}" ]]; then
  args+=(--scene_filter_yaml "${SCENE_FILTER_YAML}")
  args+=(--scene_filter_yaml_filter_only "${SCENE_FILTER_YAML_FILTER_ONLY:-1}")
fi
if [[ -n "${MAX_SCENES:-}" ]]; then args+=(--max_scenes "${MAX_SCENES}"); fi
if [[ -n "${MAX_EVAL_TOKENS:-}" ]]; then args+=(--max_eval_tokens "${MAX_EVAL_TOKENS}"); fi
if [[ -n "${NUM_EVAL_SHARDS:-}" ]]; then args+=(--num_eval_shards "${NUM_EVAL_SHARDS}"); fi
if [[ -n "${TOKEN_OFFSET:-}" ]]; then args+=(--token_offset "${TOKEN_OFFSET}"); fi
if [[ -n "${RESUME_CSV:-}" ]]; then args+=(--resume_csv "${RESUME_CSV}"); fi
if [[ -n "${VIZ_DIR:-}" ]]; then args+=(--viz_dir "${VIZ_DIR}"); fi
if [[ -n "${VIZ_MAX_TOKENS:-}" ]]; then args+=(--viz_max_tokens "${VIZ_MAX_TOKENS}"); fi
if [[ -n "${DEBUG_PROMPT_STEPS:-}" ]]; then args+=(--debug_prompt_steps "${DEBUG_PROMPT_STEPS}"); fi
if [[ "${USE_FOCAL_UNIFICATION:-0}" == "1" ]]; then args+=(--use_focal_unification); fi
if [[ -n "${FOCAL_TARGET_PX:-}" ]]; then args+=(--focal_target_px "${FOCAL_TARGET_PX}"); fi
if [[ "${SAVE_VIZ:-0}" == "1" ]]; then args+=(--save_viz); fi
if [[ "${USE_MIXED_LATENT_ATTENTION_MASK:-1}" == "1" ]]; then args+=(--use_mixed_latent_attention_mask 1); else args+=(--no_use_mixed_latent_attention_mask); fi
if [[ -n "${INFER_OUTPUT_MODE:-}" ]]; then args+=(--infer_output_mode "${INFER_OUTPUT_MODE}"); fi
if [[ "${INFER_TRAJECTORY_ONLY:-0}" == "1" ]]; then args+=(--infer_trajectory_only); else args+=(--no_infer_trajectory_only); fi
if [[ "${LEGACY_SIMULATE_TRAFFIC_AGENTS:-0}" == "1" ]]; then args+=(--legacy_simulate_traffic_agents); else args+=(--no_legacy_simulate_traffic_agents); fi
if [[ "${PRINT_TOKENS:-0}" == "1" ]]; then args+=(--print_tokens); fi
if [[ "${PRINT_ALIGNMENT_PARAMS:-1}" == "1" ]]; then args+=(--print_alignment_params); else args+=(--no_print_alignment_params); fi
if [[ "${SHOW_EVAL_PROGRESS:-1}" == "1" ]]; then args+=(--show_eval_progress); else args+=(--no_show_eval_progress); fi
if [[ "${SHOW_DENOISE_PROGRESS:-0}" == "1" ]]; then args+=(--show_denoise_progress); fi
if [[ "${SHOW_VAE_PROGRESS:-0}" == "1" ]]; then args+=(--show_vae_progress); fi
if [[ "${ENABLE_NUSCENES_METRICS:-0}" == "1" ]]; then args+=(--enable_nuscenes_metrics); fi
if [[ -n "${NUSCENES_METRIC_HORIZONS_S:-}" ]]; then args+=(--nuscenes_metric_horizons_s "${NUSCENES_METRIC_HORIZONS_S}"); fi

echo "[UNIVERSE] config=${CONFIG}"
echo "[UNIVERSE] ckpt=${FULL_CKPT}"
echo "[UNIVERSE] model=${LOCAL_MODEL_PATH}"
echo "[UNIVERSE] navsim: log=${NAVSIM_LOG_PATH} sensor=${NAVSIM_SENSOR_BLOBS_PATH} cache=${NAVSIM_METRIC_CACHE_PATH}"
echo "[UNIVERSE] eval: history=${NUM_HISTORY_FRAMES} scene_future=${NUM_FUTURE_FRAMES} model_future=${MODEL_FUTURE_FRAMES} steps=${NUM_INFERENCE_STEPS}"
echo "[UNIVERSE] focal: enabled=${USE_FOCAL_UNIFICATION:-0} target=${FOCAL_TARGET_PX:-}"
echo "[UNIVERSE] speed: infer_trajectory_only=${INFER_TRAJECTORY_ONLY} output_mode=${INFER_OUTPUT_MODE:-legacy} mixed_latent_attention=${USE_MIXED_LATENT_ATTENTION_MASK:-1} save_viz=${SAVE_VIZ}"
echo "[UNIVERSE] compat: legacy_simulate_traffic_agents=${LEGACY_SIMULATE_TRAFFIC_AGENTS}"
echo "[UNIVERSE] smoke: enabled=${SMOKE_TEST:-0} max_eval_tokens=${MAX_EVAL_TOKENS:-}"

UNIVERSE_launch_torchrun \
  "${REPO_ROOT}/examples/wanvideo/UNIVERSE_infer/eval_navsim_pdm.py" \
  "${args[@]}"
