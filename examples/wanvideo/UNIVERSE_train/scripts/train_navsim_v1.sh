#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVERSE_TRAIN_DIR="${UNIVERSE_TRAIN_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}"
REPO_ROOT="${REPO_ROOT:-"$(cd "${UNIVERSE_TRAIN_DIR}/../../.." && pwd)"}"
UNIVERSE_INFER_DIR="${UNIVERSE_INFER_DIR:-"${REPO_ROOT}/examples/wanvideo/UNIVERSE_infer"}"
CONFIG="${CONFIG:-${NAVSIM_TRAIN_CONFIG:-"${UNIVERSE_TRAIN_DIR}/configs/navsim_v1.yaml"}}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
  export MAX_SCENES="${MAX_SCENES:-8}"
  export NUM_EPOCHS="${NUM_EPOCHS:-1}"
  export SAVE_STEPS="${SAVE_STEPS:-2}"
  export LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"
  export DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
  export OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/outputs/universe/train_navsim_v1_smoke}"
fi

if [ ! -f "${CONFIG}" ]; then
  echo "[UNIVERSE] config not found: ${CONFIG}" >&2
  exit 1
fi

export REPO_ROOT UNIVERSE_TRAIN_DIR UNIVERSE_INFER_DIR CONFIG
config_exports="$("${PYTHON:-python}" "${UNIVERSE_INFER_DIR}/scripts/load_yaml_config.py" "${CONFIG}")"
eval "${config_exports}"

# shellcheck source=/dev/null
source "${UNIVERSE_INFER_DIR}/scripts/distributed_env.sh"
UNIVERSE_setup_distributed_env

export PYTHONPATH="${REPO_ROOT}:${UNIVERSE_TRAIN_DIR}:${UNIVERSE_INFER_DIR}:${REPO_ROOT}/third_party:${REPO_ROOT}/third_party/nuscenes-devkit/python-sdk:${PYTHONPATH:-}"
export NUPLAN_MAPS_ROOT
export NUPLAN_DATA_ROOT
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-${DDP_TIMEOUT_SECONDS}}"

mkdir -p "${UNIVERSE_TMPDIR}"
export TMPDIR="${TMPDIR:-${UNIVERSE_TMPDIR}}"
export TMP="${TMP:-${UNIVERSE_TMPDIR}}"
export TEMP="${TEMP:-${UNIVERSE_TMPDIR}}"

args=(
  --repo_root "${REPO_ROOT}"
  --navsim_log_path "${NAVSIM_LOG_PATH}"
  --sensor_blobs_path "${SENSOR_BLOBS_PATH}"
  --local_model_path "${LOCAL_MODEL_PATH}"
  --output_path "${OUTPUT_PATH}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_history_frames "${NUM_HISTORY_FRAMES}"
  --num_future_frames "${NUM_FUTURE_FRAMES}"
  --target_fps "${TARGET_FPS}"
  --frame_interval "${FRAME_INTERVAL}"
  --learning_rate "${LR}"
  --num_epochs "${NUM_EPOCHS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --weight_decay "${WEIGHT_DECAY}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --ddp_timeout_seconds "${DDP_TIMEOUT_SECONDS}"
  --warmup_steps "${WARMUP_STEPS}"
  --warmup_start_factor "${WARMUP_START_FACTOR}"
  --log_every_steps "${LOG_EVERY_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --train_log_file "${TRAIN_LOG_FILE}"
  --trainable_models "${TRAINABLE_MODELS}"
  --lora_base_model "${LORA_BASE_MODEL}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --lora_rank "${LORA_RANK}"
  --extra_inputs "${EXTRA_INPUTS}"
  --trajectory_condition_mode "${TRAJECTORY_CONDITION_MODE}"
  --max_timestep_boundary "${MAX_TIMESTEP_BOUNDARY}"
  --min_timestep_boundary "${MIN_TIMESTEP_BOUNDARY}"
  --auto_eval_ckpt_kind "${AUTO_EVAL_CKPT_KIND}"
  --infer_all_output_root "${INFER_ALL_OUTPUT_ROOT}"
)

if [[ -n "${FULL_CKPT:-}" ]]; then args+=(--full_ckpt "${FULL_CKPT}"); fi
if [[ -n "${CACHE_PATH:-}" ]]; then args+=(--cache_path "${CACHE_PATH}"); fi
if [[ -n "${TRAIN_LOG_NAMES:-}" ]]; then args+=(--train_log_names "${TRAIN_LOG_NAMES}"); fi
if [[ -n "${MAX_SCENES:-}" ]]; then args+=(--max_scenes "${MAX_SCENES}"); fi
if [[ -n "${LORA_CHECKPOINT:-}" ]]; then args+=(--lora_checkpoint "${LORA_CHECKPOINT}"); fi
if [[ -n "${GRADIENT_CLIP_NORM:-}" ]]; then args+=(--gradient_clip_norm "${GRADIENT_CLIP_NORM}"); fi

if [[ "${USE_CACHE_ONLY:-0}" == "1" ]]; then args+=(--use_cache_only); fi
if [[ "${FORCE_CACHE_COMPUTATION:-0}" == "1" ]]; then args+=(--force_cache_computation); fi
if [[ "${SKIP_MISSING_FILES:-0}" == "1" ]]; then args+=(--skip_missing_files); fi
if [[ "${PRINT_NAVSIM_TOKENS:-0}" == "1" ]]; then args+=(--print_navsim_tokens); fi
if [[ "${SURROUND_VIEW:-0}" == "1" ]]; then args+=(--surround_view); fi
if [[ "${USE_FOCAL_UNIFICATION:-0}" == "1" ]]; then args+=(--use_focal_unification); fi
if [[ -n "${FOCAL_TARGET_PX:-}" ]]; then args+=(--focal_target_px "${FOCAL_TARGET_PX}"); fi
if [[ -n "${FOCAL_STATS_SAMPLES:-}" ]]; then args+=(--focal_stats_samples "${FOCAL_STATS_SAMPLES}"); fi
if [[ "${USE_TRAJECTORY:-1}" == "1" ]]; then args+=(--use_trajectory); fi
if [[ "${USE_GRADIENT_CHECKPOINTING:-1}" == "1" ]]; then
  args+=(--use_gradient_checkpointing 1)
else
  args+=(--no_use_gradient_checkpointing)
fi
if [[ "${USE_GRADIENT_CHECKPOINTING_OFFLOAD:-0}" == "1" ]]; then args+=(--use_gradient_checkpointing_offload); fi
if [[ "${TRAIN_FUTURE_VIDEO_NOISE_ONLY:-1}" == "1" ]]; then
  args+=(--train_future_video_noise_only 1)
else
  args+=(--no_train_future_video_noise_only)
fi
if [[ "${INFER_REPLACE_HISTORY_LATENTS_BEFORE_DECODE:-1}" == "1" ]]; then
  args+=(--infer_replace_history_latents_before_decode 1)
else
  args+=(--no_infer_replace_history_latents_before_decode)
fi
if [[ "${USE_MIXED_LATENT_ATTENTION_MASK:-1}" == "1" ]]; then
  args+=(--use_mixed_latent_attention_mask 1)
else
  args+=(--no_use_mixed_latent_attention_mask)
fi
if [[ "${INFER_TRAJECTORY_ONLY:-1}" == "1" ]]; then
  args+=(--infer_trajectory_only 1)
else
  args+=(--no_infer_trajectory_only)
fi
if [[ "${USE_EMA:-0}" == "1" ]]; then args+=(--use_ema); fi
if [[ "${EMA_ON_CPU:-0}" == "1" ]]; then args+=(--ema_on_cpu); fi
if [[ "${SAVE_EMA:-0}" == "1" ]]; then args+=(--save_ema); fi
if [[ "${SAVE_RAW_CKPT:-1}" != "1" ]]; then args+=(--no_save_raw_ckpt); fi
if [[ "${AUTO_EVAL:-0}" == "1" ]]; then args+=(--auto_eval); else args+=(--no_auto_eval); fi
if [[ "${AUTO_EVAL_STRICT:-0}" == "1" ]]; then args+=(--auto_eval_strict); fi
args+=(--ema_decay "${EMA_DECAY}" --ema_update_after_step "${EMA_UPDATE_AFTER_STEP}" --ema_update_every "${EMA_UPDATE_EVERY}")

echo "[UNIVERSE] config=${CONFIG}"
echo "[UNIVERSE] ckpt=${FULL_CKPT:-}"
echo "[UNIVERSE] model=${LOCAL_MODEL_PATH}"
echo "[UNIVERSE] navsim train: log=${NAVSIM_LOG_PATH} sensor=${SENSOR_BLOBS_PATH}"
echo "[UNIVERSE] train: output=${OUTPUT_PATH} epochs=${NUM_EPOCHS} save_steps=${SAVE_STEPS} lr=${LR}"
echo "[UNIVERSE] focal: enabled=${USE_FOCAL_UNIFICATION:-0} target=${FOCAL_TARGET_PX:-} stats_samples=${FOCAL_STATS_SAMPLES:-0}"
echo "[UNIVERSE] trajectory: enabled=${USE_TRAJECTORY} mode=${TRAJECTORY_CONDITION_MODE} future=${NUM_FUTURE_FRAMES}"
echo "[UNIVERSE] masks: train_future_video_noise_only=${TRAIN_FUTURE_VIDEO_NOISE_ONLY:-1} mixed_latent_attention=${USE_MIXED_LATENT_ATTENTION_MASK:-1} infer_trajectory_only=${INFER_TRAJECTORY_ONLY:-1}"
echo "[UNIVERSE] auto_eval: enabled=${AUTO_EVAL} kind=${AUTO_EVAL_CKPT_KIND} output=${INFER_ALL_OUTPUT_ROOT} navsim=${RUN_NAVSIM} nuscenes=${RUN_NUSCENES} b2d=${RUN_B2D}"

UNIVERSE_launch_torchrun \
  "${REPO_ROOT}/examples/wanvideo/UNIVERSE_train/train_navsim_v1.py" \
  "${args[@]}"
