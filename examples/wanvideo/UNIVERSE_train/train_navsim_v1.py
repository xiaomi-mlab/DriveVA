"""
UNIVERSE NAVSIM v1 training entrypoint.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import importlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task

from navsim_dataset import DEFAULT_NEGATIVE_PROMPT, NavsimUNIVERSEConfig, NavsimUNIVERSEDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


_NAVSIM_EVAL_LOG_DEFAULT = "/path/to/navsim_v1.1/navsim_logs/test"
_NAVSIM_EVAL_SENSOR_DEFAULT = "/path/to/navsim_v1.1/sensor_blobs/test"
_NAVSIM_EVAL_CACHE_DEFAULT = (
    "/path/to/navsim_v1.1/metric_cache"
)


class _TeeStream:
    def __init__(self, console_stream, file_stream):
        self._console_stream = console_stream
        self._file_stream = file_stream

    def write(self, data):
        written = self._console_stream.write(data)
        self._file_stream.write(data)
        return written if isinstance(written, int) else len(str(data))

    def flush(self):
        self._console_stream.flush()
        self._file_stream.flush()

    def isatty(self):
        return bool(getattr(self._console_stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self._console_stream, "encoding", None)

    def __getattr__(self, name):
        return getattr(self._console_stream, name)


def _is_rank0() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def _enable_train_log_capture(output_path: str, train_log_file: str) -> None:
    if not _is_rank0() or not train_log_file:
        return
    output_dir = Path(output_path).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(train_log_file).expanduser()
    if not log_path.is_absolute():
        log_path = output_dir / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_fp)
    sys.stderr = _TeeStream(original_stderr, log_fp)

    def _restore_streams() -> None:
        try:
            log_fp.flush()
            log_fp.close()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    atexit.register(_restore_streams)


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _normalize_train_ckpt_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    known_pipe_prefixes = (
        "dit.",
        "dit2.",
        "trajectory_encoder.",
        "trajectory_head.",
        "trajectory_decoder.",
        "text_encoder.",
        "vae.",
        "image_encoder.",
        "prompter.",
        "scheduler.",
        "motion_controller.",
        "vace.",
        "animate_adapter.",
        "audio_processor.",
    )
    normalized = {}
    for key, value in state_dict.items():
        norm_key = key[5:] if key.startswith("pipe.") else key
        if not norm_key.startswith(known_pipe_prefixes):
            norm_key = f"dit.{norm_key}"
        normalized[norm_key] = value
    return normalized


def _summarize_ckpt_keys(keys):
    if not keys:
        return ""
    counter = Counter(k.split(".", 1)[0] if "." in k else k for k in keys)
    return ", ".join(f"{name}:{count}" for name, count in counter.most_common())


def _env_first(*names: str, default: Any = None) -> Any:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value) != "":
            return value
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    value = _env_first(*names, default="1" if default else "0")
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _add_optional_arg(argv: list[str], flag: str, value: Any) -> None:
    if value is not None and str(value) != "":
        argv.extend([flag, str(value)])


def _format_focal_stats(stats: Dict[str, float]) -> str:
    if not stats or float(stats.get("count", 0.0)) <= 0.0:
        return "count=0"
    parts = [f"count={int(stats.get('count', 0.0))}"]
    for key in ("raw_focal_mean", "effective_focal_mean", "effective_fx", "effective_fy"):
        median_key = f"{key}_median"
        if median_key not in stats:
            continue
        parts.append(
            f"{key}:"
            f"min={stats.get(f'{key}_min', float('nan')):.3f},"
            f"p25={stats.get(f'{key}_p25', float('nan')):.3f},"
            f"median={stats.get(median_key, float('nan')):.3f},"
            f"p75={stats.get(f'{key}_p75', float('nan')):.3f},"
            f"max={stats.get(f'{key}_max', float('nan')):.3f}"
        )
    return " ".join(parts)


def _module_training_states(root: torch.nn.Module) -> Dict[torch.nn.Module, bool]:
    return {module: bool(module.training) for module in root.modules()}


def _restore_module_training_states(states: Dict[torch.nn.Module, bool]) -> None:
    for module, training in states.items():
        module.train(training)


@contextlib.contextmanager
def _use_ema_trainable_weights(model: "UNIVERSENavsimTrainingModule", enabled: bool):
    if not enabled or not hasattr(model, "has_ema") or not model.has_ema():
        yield False
        return

    shadow = getattr(model, "_ema_shadow", {})
    backup: Dict[str, torch.Tensor] = {}
    try:
        with torch.no_grad():
            for name, param in model.named_parameters():
                ema_value = shadow.get(name)
                if ema_value is None:
                    continue
                backup[name] = param.detach().cpu().clone()
                param.copy_(ema_value.to(device=param.device, dtype=param.dtype))
        yield True
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                raw_value = backup.get(name)
                if raw_value is not None:
                    param.copy_(raw_value.to(device=param.device, dtype=param.dtype))


class InProcessAutoEvalModelLogger(ModelLogger):
    def __init__(
        self,
        *args,
        train_args: argparse.Namespace,
        auto_eval: bool,
        auto_eval_ckpt_kind: str = "ema",
        auto_eval_strict: bool = False,
        infer_all_output_root: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_args = train_args
        self.auto_eval = bool(auto_eval)
        self.auto_eval_ckpt_kind = str(auto_eval_ckpt_kind or "ema").lower()
        self.auto_eval_strict = bool(auto_eval_strict)
        self.infer_all_output_root = infer_all_output_root or os.path.join(train_args.output_path, "infer_each_ckpt")
        self._eval_modules: Dict[str, Any] = {}
        if self.auto_eval_ckpt_kind not in {"raw", "ema", "all"}:
            raise ValueError(f"auto_eval_ckpt_kind must be raw, ema, or all; got {self.auto_eval_ckpt_kind}")

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % int(save_steps) == 0:
            file_name = f"step-{self.num_steps}.safetensors"
            self.save_model(accelerator, model, file_name)
            self._run_auto_eval(accelerator, model, os.path.splitext(file_name)[0])

    def on_epoch_end(self, accelerator, model, epoch_id):
        file_name = f"epoch-{epoch_id}.safetensors"
        self.save_model(accelerator, model, file_name)
        self._run_auto_eval(accelerator, model, os.path.splitext(file_name)[0])

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % int(save_steps) != 0:
            file_name = f"step-{self.num_steps}.safetensors"
            self.save_model(accelerator, model, file_name)
            self._run_auto_eval(accelerator, model, os.path.splitext(file_name)[0])

    def _eval_module(self, name: str):
        if name not in self._eval_modules:
            self._eval_modules[name] = importlib.import_module(name)
        return self._eval_modules[name]

    def _eval_variants(self, unwrapped_model: UNIVERSENavsimTrainingModule) -> list[tuple[str, bool]]:
        if self.auto_eval_ckpt_kind == "raw":
            return [("", False)]
        if self.auto_eval_ckpt_kind == "ema":
            if hasattr(unwrapped_model, "has_ema") and unwrapped_model.has_ema():
                return [("-ema", True)]
            return []
        variants = [("", False)]
        if hasattr(unwrapped_model, "has_ema") and unwrapped_model.has_ema():
            variants.append(("-ema", True))
        return variants

    def _append_focal_unification_args(self, argv: list[str]) -> None:
        if bool(getattr(self.train_args, "use_focal_unification", False)):
            argv.append("--use_focal_unification")
            _add_optional_arg(argv, "--focal_target_px", getattr(self.train_args, "focal_target_px", None))

    def _build_navsim_args(self, tag: str, ckpt_path: str) -> argparse.Namespace:
        mod = self._eval_module("examples.wanvideo.UNIVERSE_infer.eval_navsim_pdm")
        out_dir = os.path.join(self.infer_all_output_root, tag, "navsim_v1")
        argv = [
            "--repo_root", self.train_args.repo_root,
            "--navsim_log_path", str(_env_first("EVAL_V1_NAVSIM_LOG_PATH", "EVAL_NAVSIM_LOG_PATH", default=_NAVSIM_EVAL_LOG_DEFAULT)),
            "--sensor_blobs_path", str(_env_first("EVAL_V1_SENSOR_BLOBS_PATH", "EVAL_SENSOR_BLOBS_PATH", default=_NAVSIM_EVAL_SENSOR_DEFAULT)),
            "--metric_cache_path", str(_env_first("EVAL_V1_METRIC_CACHE_PATH", "EVAL_METRIC_CACHE_PATH", default=_NAVSIM_EVAL_CACHE_DEFAULT)),
            "--output_dir", out_dir,
            "--local_model_path", str(self.train_args.local_model_path),
            "--full_ckpt", ckpt_path,
            "--num_inference_steps", str(_env_first("NAVSIM_NUM_INFERENCE_STEPS", "EVAL_NUM_INFERENCE_STEPS", "NUM_INFERENCE_STEPS", default=3)),
            "--cfg_scale", str(_env_first("NAVSIM_CFG_SCALE", "EVAL_CFG_SCALE", "CFG_SCALE", default=1.0)),
            "--seed", str(_env_first("NAVSIM_SEED", "EVAL_SEED", "SEED", default=0)),
            "--num_history_frames", str(_env_first("EVAL_NUM_HISTORY_FRAMES", default=self.train_args.num_history_frames)),
            "--num_future_frames", str(_env_first("EVAL_NUM_FUTURE_FRAMES", default=10)),
            "--model_future_frames", str(_env_first("EVAL_MODEL_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--target_fps", str(_env_first("EVAL_TARGET_FPS", default=self.train_args.target_fps)),
            "--height", str(_env_first("EVAL_HEIGHT", default=self.train_args.height)),
            "--width", str(_env_first("EVAL_WIDTH", default=self.train_args.width)),
            "--trajectory_condition_mode", str(_env_first("EVAL_TRAJECTORY_CONDITION_MODE", default=self.train_args.trajectory_condition_mode)),
            "--scene_future_extra_seconds", str(_env_first("EVAL_V1_SCENE_FUTURE_EXTRA_SECONDS", "EVAL_SCENE_FUTURE_EXTRA_SECONDS", default=1.0)),
            "--pdm_num_poses", str(_env_first("EVAL_PDM_NUM_POSES", "PDM_NUM_POSES", default=40)),
            "--pdm_interval_length", str(_env_first("EVAL_PDM_INTERVAL_LENGTH", "PDM_INTERVAL_LENGTH", default=0.1)),
            "--traffic_agents_policy", str(_env_first("EVAL_TRAFFIC_AGENTS_POLICY", "TRAFFIC_AGENTS_POLICY", default="non_reactive")),
            "--viz_total_tokens", str(_env_first("EVAL_VIZ_TOTAL_TOKENS", default=100)),
            "--debug_prompt_steps", str(_env_first("EVAL_DEBUG_PROMPT_STEPS", default=0)),
            "--no_show_eval_progress",
        ]
        _add_optional_arg(argv, "--scene_filter_yaml", _env_first("EVAL_SCENE_FILTER_YAML", default=os.path.join(self.train_args.repo_root, "examples/wanvideo/driveva_infer/navsim_scene_filters/navtest.yaml")))
        if _env_bool("EVAL_SCENE_FILTER_YAML_FILTER_ONLY", default=True):
            argv.extend(["--scene_filter_yaml_filter_only", "1"])
        _add_optional_arg(argv, "--log_names", _env_first("EVAL_LOG_NAMES", "EVAL_V1_LOG_NAMES"))
        _add_optional_arg(argv, "--max_scenes", _env_first("EVAL_MAX_SCENES", "EVAL_V1_MAX_SCENES"))
        _add_optional_arg(argv, "--max_eval_tokens", _env_first("EVAL_MAX_EVAL_TOKENS", "MAX_EVAL_TOKENS"))
        _add_optional_arg(argv, "--num_eval_shards", _env_first("EVAL_NUM_SHARDS", "NUM_EVAL_SHARDS"))
        if _env_bool("EVAL_SAVE_VIZ", default=False):
            argv.append("--save_viz")
        if _env_bool("EVAL_USE_MIXED_LATENT_ATTENTION_MASK", "USE_MIXED_LATENT_ATTENTION_MASK", default=self.train_args.use_mixed_latent_attention_mask):
            argv.extend(["--use_mixed_latent_attention_mask", "1"])
        else:
            argv.append("--no_use_mixed_latent_attention_mask")
        _add_optional_arg(argv, "--infer_output_mode", _env_first("EVAL_INFER_OUTPUT_MODE", "INFER_OUTPUT_MODE"))
        if _env_bool("EVAL_INFER_TRAJECTORY_ONLY", default=True):
            argv.append("--infer_trajectory_only")
        else:
            argv.append("--no_infer_trajectory_only")
        if _env_bool("AUTO_EVAL_NUSCENES_METRICS", "ENABLE_NUSCENES_METRICS", default=False):
            argv.append("--enable_nuscenes_metrics")
            _add_optional_arg(argv, "--nuscenes_metric_horizons_s", _env_first("EVAL_NUSCENES_METRIC_HORIZONS_S", "NUSCENES_METRIC_HORIZONS_S", default="1,2,3"))
        self._append_focal_unification_args(argv)
        return mod.parse_args(argv)

    def _build_nuscenes_args(self, tag: str, ckpt_path: str) -> Optional[argparse.Namespace]:
        dataroot = _env_first("NUSCENES_EVAL_DATAROOT", "NUSCENES_DATAROOT")
        if not dataroot:
            return None
        mod = self._eval_module("examples.wanvideo.UNIVERSE_infer.infer_nuscenes")
        out_dir = os.path.join(self.infer_all_output_root, tag, "nuscenes")
        argv = [
            "--nuscenes_dataroot", str(dataroot),
            "--nuscenes_version", str(_env_first("NUSCENES_EVAL_VERSION", "NUSCENES_VERSION", default="v1.0-trainval")),
            "--split", str(_env_first("NUSCENES_EVAL_SPLIT", "NUSCENES_SPLIT", default="val")),
            "--camera_name", str(_env_first("NUSCENES_EVAL_CAMERA_NAME", "CAMERA_NAME", default="CAM_FRONT")),
            "--output_dir", out_dir,
            "--local_model_path", str(self.train_args.local_model_path),
            "--full_ckpt", ckpt_path,
            "--num_inference_steps", str(_env_first("NUSCENES_EVAL_NUM_INFERENCE_STEPS", "NUM_INFERENCE_STEPS", default=3)),
            "--cfg_scale", str(_env_first("NUSCENES_EVAL_CFG_SCALE", "CFG_SCALE", default=1.0)),
            "--seed", str(_env_first("NUSCENES_EVAL_SEED", "SEED", default=0)),
            "--debug_prompt_steps", str(_env_first("NUSCENES_EVAL_DEBUG_PROMPT_STEPS", default=0)),
            "--trajectory_condition_mode", str(_env_first("NUSCENES_EVAL_TRAJECTORY_CONDITION_MODE", default=self.train_args.trajectory_condition_mode)),
            "--num_history_frames", str(_env_first("NUSCENES_EVAL_NUM_HISTORY_FRAMES", default=self.train_args.num_history_frames)),
            "--num_future_frames", str(_env_first("NUSCENES_EVAL_NUM_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--model_future_frames", str(_env_first("NUSCENES_EVAL_MODEL_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--scene_future_extra_seconds", str(_env_first("NUSCENES_EVAL_SCENE_FUTURE_EXTRA_SECONDS", default=0.0)),
            "--target_fps", str(_env_first("NUSCENES_EVAL_TARGET_FPS", default=self.train_args.target_fps)),
            "--metric_horizons_s", str(_env_first("NUSCENES_EVAL_METRIC_HORIZONS_S", default="1,2,3")),
            "--ego_box_length_m", str(_env_first("NUSCENES_EVAL_EGO_BOX_LENGTH_M", default=4.084)),
            "--ego_box_width_m", str(_env_first("NUSCENES_EVAL_EGO_BOX_WIDTH_M", default=1.85)),
            "--height", str(_env_first("NUSCENES_EVAL_HEIGHT", default=self.train_args.height)),
            "--width", str(_env_first("NUSCENES_EVAL_WIDTH", default=self.train_args.width)),
            "--no_show_eval_progress",
        ]
        _add_optional_arg(argv, "--policy_anno_json", _env_first("NUSCENES_EVAL_POLICY_ANNO_JSON", "POLICY_ANNO_JSON"))
        _add_optional_arg(argv, "--max_scenes", _env_first("NUSCENES_EVAL_MAX_SCENES"))
        _add_optional_arg(argv, "--max_eval_tokens", _env_first("NUSCENES_EVAL_MAX_EVAL_TOKENS", "MAX_EVAL_TOKENS"))
        if _env_bool("NUSCENES_EVAL_SAVE_VIZ", default=False):
            argv.append("--save_viz")
            _add_optional_arg(argv, "--viz_max_tokens", _env_first("NUSCENES_EVAL_VIZ_MAX_TOKENS", default=20))
        if _env_bool("NUSCENES_EVAL_USE_MIXED_LATENT_ATTENTION_MASK", "EVAL_USE_MIXED_LATENT_ATTENTION_MASK", "USE_MIXED_LATENT_ATTENTION_MASK", default=self.train_args.use_mixed_latent_attention_mask):
            argv.append("--use_mixed_latent_attention_mask")
        else:
            argv.append("--no_use_mixed_latent_attention_mask")
        _add_optional_arg(argv, "--infer_output_mode", _env_first("NUSCENES_EVAL_INFER_OUTPUT_MODE", "EVAL_INFER_OUTPUT_MODE", "INFER_OUTPUT_MODE"))
        if _env_bool("NUSCENES_EVAL_INFER_TRAJECTORY_ONLY", "EVAL_INFER_TRAJECTORY_ONLY", default=self.train_args.infer_trajectory_only):
            argv.append("--infer_trajectory_only")
        else:
            argv.append("--no_infer_trajectory_only")
        self._append_focal_unification_args(argv)
        return mod.parse_args(argv)

    def _build_b2d_args(self, tag: str, ckpt_path: str) -> Optional[argparse.Namespace]:
        data_root = _env_first("B2D_DATA_ROOT", "BENCH2DRIVE_DATA_ROOT")
        ann_file = _env_first("B2D_EVAL_ANN_FILE", "B2D_ANN_FILE", "BENCH2DRIVE_ANN_FILE")
        if not data_root or not ann_file:
            return None
        mod = self._eval_module("examples.wanvideo.UNIVERSE_infer.infer_bench2drive")
        out_dir = os.path.join(self.infer_all_output_root, tag, "bench2drive")
        argv = [
            "--data_root", str(data_root),
            "--ann_file", str(ann_file),
            "--output_dir", out_dir,
            "--local_model_path", str(self.train_args.local_model_path),
            "--full_ckpt", ckpt_path,
            "--num_inference_steps", str(_env_first("B2D_EVAL_NUM_INFERENCE_STEPS", "NUM_INFERENCE_STEPS", default=3)),
            "--cfg_scale", str(_env_first("B2D_EVAL_CFG_SCALE", "CFG_SCALE", default=1.0)),
            "--seed", str(_env_first("B2D_EVAL_SEED", "SEED", default=0)),
            "--num_history_frames", str(_env_first("B2D_EVAL_NUM_HISTORY_FRAMES", default=self.train_args.num_history_frames)),
            "--num_future_frames", str(_env_first("B2D_EVAL_NUM_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--model_future_frames", str(_env_first("B2D_EVAL_MODEL_FUTURE_FRAMES", default=self.train_args.num_future_frames)),
            "--target_fps", str(_env_first("B2D_EVAL_TARGET_FPS", default=self.train_args.target_fps)),
            "--original_fps", str(_env_first("B2D_EVAL_ORIGINAL_FPS", "B2D_ORIGINAL_FPS", default=10)),
            "--frame_interval", str(_env_first("B2D_EVAL_FRAME_INTERVAL", "B2D_FRAME_INTERVAL", default=2)),
            "--height", str(_env_first("B2D_EVAL_HEIGHT", default=self.train_args.height)),
            "--width", str(_env_first("B2D_EVAL_WIDTH", default=self.train_args.width)),
            "--command_yaw_threshold_deg", str(_env_first("B2D_EVAL_COMMAND_YAW_THRESHOLD_DEG", default=8.0)),
            "--viz_max_scenes", str(_env_first("B2D_EVAL_VIZ_MAX_SCENES", default=30)),
            "--viz_plot_height", str(_env_first("B2D_EVAL_VIZ_PLOT_HEIGHT", default=420)),
            "--debug_prompt_steps", str(_env_first("B2D_EVAL_DEBUG_PROMPT_STEPS", default=0)),
            "--projection_debug_steps", str(_env_first("B2D_EVAL_PROJECTION_DEBUG_STEPS", default=0)),
            "--projection_debug_token", str(_env_first("B2D_EVAL_PROJECTION_DEBUG_TOKEN", default="")),
            "--distributed",
            "--no_show_eval_progress",
        ]
        _add_optional_arg(argv, "--max_scenes", _env_first("B2D_EVAL_MAX_SCENES"))
        if _env_bool("B2D_EVAL_SAVE_VIZ", default=False):
            argv.append("--save_viz")
        else:
            argv.append("--no_save_viz")
        if _env_bool("B2D_EVAL_USE_MIXED_LATENT_ATTENTION_MASK", "EVAL_USE_MIXED_LATENT_ATTENTION_MASK", "USE_MIXED_LATENT_ATTENTION_MASK", default=self.train_args.use_mixed_latent_attention_mask):
            argv.append("--use_mixed_latent_attention_mask")
        else:
            argv.append("--no_use_mixed_latent_attention_mask")
        _add_optional_arg(argv, "--infer_output_mode", _env_first("B2D_EVAL_INFER_OUTPUT_MODE", "EVAL_INFER_OUTPUT_MODE", "INFER_OUTPUT_MODE"))
        if _env_bool("B2D_EVAL_INFER_TRAJECTORY_ONLY", "EVAL_INFER_TRAJECTORY_ONLY", default=self.train_args.infer_trajectory_only):
            argv.append("--infer_trajectory_only")
        else:
            argv.append("--no_infer_trajectory_only")
        if _env_bool("B2D_EVAL_COMPUTE_PLANNING_METRICS", default=True):
            argv.append("--compute_planning_metrics")
        else:
            argv.append("--no_compute_planning_metrics")
        self._append_focal_unification_args(argv)
        return mod.parse_args(argv)

    def _run_auto_eval(self, accelerator, model, tag_base: str):
        if not self.auto_eval:
            return
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        pipe = unwrapped_model.pipe
        raw_ckpt_path = os.path.join(self.output_path, f"{tag_base}.safetensors")
        variants = self._eval_variants(unwrapped_model)
        if not variants:
            if accelerator.is_main_process:
                print("[train][eval][warn] auto eval requested but no matching checkpoint variant is available.")
            return

        module_states = _module_training_states(pipe)
        scheduler = getattr(pipe, "scheduler", None)
        scheduler_state = None
        if scheduler is not None:
            try:
                scheduler_state = {
                    "num_inference_steps": int(len(scheduler.timesteps)),
                    "training": bool(getattr(scheduler, "training", False)),
                    "shift": float(getattr(scheduler, "shift", 5.0)),
                }
            except Exception:
                scheduler_state = None
        attrs = {
            name: getattr(pipe, name)
            for name in (
                "target_fps",
                "num_history_frames",
                "trajectory_norm_mode",
                "trajectory_use_relative",
                "trajectory_condition_mode",
                "infer_replace_history_latents_before_decode",
            )
            if hasattr(pipe, name)
        }
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_cpu_rng_state = torch.random.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        vae_tqdm = None
        try:
            import diffsynth.models.wan_video_vae as wan_video_vae_mod

            vae_tqdm = getattr(wan_video_vae_mod, "tqdm", None)
        except Exception:
            vae_tqdm = None
        try:
            for param in unwrapped_model.parameters():
                param.grad = None
            torch.cuda.empty_cache()
            pipe.eval()
            for suffix, use_ema in variants:
                tag = f"{tag_base}{suffix}"
                ckpt_path = os.path.join(self.output_path, f"{tag}.safetensors") if suffix else raw_ckpt_path
                if accelerator.is_main_process:
                    print(
                        "[train][eval] start:",
                        f"tag={tag}",
                        f"world_size={accelerator.num_processes}",
                        f"ema={use_ema}",
                        f"output={os.path.join(self.infer_all_output_root, tag)}",
                    )
                try:
                    with _use_ema_trainable_weights(unwrapped_model, enabled=use_ema):
                        with torch.inference_mode():
                            if _env_bool("RUN_NAVSIM", "AUTO_EVAL_NAVSIM_V1", default=True):
                                navsim_mod = self._eval_module("examples.wanvideo.UNIVERSE_infer.eval_navsim_pdm")
                                navsim_mod.run_eval(self._build_navsim_args(tag, ckpt_path), external_pipe=pipe)
                                accelerator.wait_for_everyone()
                            if _env_bool("RUN_NUSCENES", "AUTO_EVAL_NUSCENES", default=True):
                                nuscenes_args = self._build_nuscenes_args(tag, ckpt_path)
                                if nuscenes_args is not None:
                                    nusc_mod = self._eval_module("examples.wanvideo.UNIVERSE_infer.infer_nuscenes")
                                    nusc_mod.run_eval(nuscenes_args, external_pipe=pipe)
                                    accelerator.wait_for_everyone()
                                elif accelerator.is_main_process:
                                    print("[train][eval][nuscenes][warn] dataroot not set; skip.")
                            if _env_bool("RUN_B2D", "AUTO_EVAL_B2D_VIZ", default=True):
                                b2d_args = self._build_b2d_args(tag, ckpt_path)
                                if b2d_args is not None:
                                    b2d_mod = self._eval_module("examples.wanvideo.UNIVERSE_infer.infer_bench2drive")
                                    b2d_mod.run_eval(b2d_args, external_pipe=pipe)
                                    accelerator.wait_for_everyone()
                                elif accelerator.is_main_process:
                                    print("[train][eval][b2d][warn] data root or ann file not set; skip.")
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f"[train][eval][error] tag={tag} failed: {exc}")
                    if self.auto_eval_strict:
                        raise
                finally:
                    torch.cuda.empty_cache()
                    accelerator.wait_for_everyone()
        finally:
            for name, value in attrs.items():
                setattr(pipe, name, value)
            if vae_tqdm is not None:
                try:
                    import diffsynth.models.wan_video_vae as wan_video_vae_mod

                    wan_video_vae_mod.tqdm = vae_tqdm
                except Exception:
                    pass
            try:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.random.set_rng_state(torch_cpu_rng_state)
                if torch_cuda_rng_state is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(torch_cuda_rng_state)
            except Exception as exc:
                if accelerator.is_main_process:
                    print(f"[train][eval][warn] failed to restore RNG states: {exc}")
            _restore_module_training_states(module_states)
            if scheduler is not None and scheduler_state is not None:
                try:
                    scheduler.set_timesteps(
                        scheduler_state["num_inference_steps"],
                        training=scheduler_state["training"],
                        shift=scheduler_state["shift"],
                    )
                except Exception as exc:
                    if accelerator.is_main_process:
                        print(f"[train][eval][warn] failed to restore scheduler state: {exc}")
            accelerator.wait_for_everyone()


class UNIVERSENavsimTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        *,
        local_model_path: Optional[str],
        trainable_models: Optional[str],
        lora_base_model: Optional[str],
        lora_target_modules: str,
        lora_rank: int,
        lora_checkpoint: Optional[str],
        use_gradient_checkpointing: bool,
        use_gradient_checkpointing_offload: bool,
        extra_inputs: Optional[str],
        max_timestep_boundary: float,
        min_timestep_boundary: float,
        target_fps: int,
        negative_prompt: str,
        use_trajectory: bool,
        train_future_video_noise_only: bool,
        infer_replace_history_latents_before_decode: bool,
        use_mixed_latent_attention_mask: bool,
        infer_trajectory_only: bool,
        trajectory_condition_mode: str,
        num_history_frames: int,
    ):
        super().__init__()
        self.negative_prompt = negative_prompt
        self.target_fps = target_fps
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.use_gradient_checkpointing_offload = bool(use_gradient_checkpointing_offload)
        self.extra_inputs = [x.strip() for x in extra_inputs.split(",") if x.strip()] if extra_inputs else []
        self.max_timestep_boundary = float(max_timestep_boundary)
        self.min_timestep_boundary = float(min_timestep_boundary)

        model_configs = [
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
                local_model_path=local_model_path,
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
                local_model_path=local_model_path,
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device="cpu",
                local_model_path=local_model_path,
                skip_download=True,
            ),
        ]
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.2-TI2V-5B",
            origin_file_pattern="google/*",
            local_model_path=local_model_path,
            skip_download=True,
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            use_trajectory=use_trajectory,
        )
        self.pipe.target_fps = target_fps
        self.pipe.num_history_frames = max(0, int(num_history_frames))
        self.pipe.train_future_video_noise_only = bool(train_future_video_noise_only)
        self.pipe.infer_replace_history_latents_before_decode = bool(infer_replace_history_latents_before_decode)
        self.pipe.use_mixed_latent_attention_mask = bool(use_mixed_latent_attention_mask)
        self.pipe.infer_trajectory_only = bool(infer_trajectory_only)
        self.pipe.trajectory_norm_mode = "UNIVERSE_odo"
        self.pipe.trajectory_use_relative = False
        self.pipe.trajectory_condition_mode = str(trajectory_condition_mode).lower()

        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
        )

    def forward_preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": self.negative_prompt}
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": data["num_frames"],
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "target_fps": self.target_fps,
        }

        if "trajectory" in data:
            inputs_shared["trajectory"] = data["trajectory"]
        if "ego_vel" in data:
            inputs_shared["ego_vel"] = data["ego_vel"]
        if self.pipe.trajectory_condition_mode in {"auto", "history"} and "history_positions" in data:
            inputs_shared["history_positions"] = data["history_positions"]

        for extra_input in self.extra_inputs:
            if extra_input == "longcat_video":
                inputs_shared["longcat_video"] = data.get("longcat_video")
            elif extra_input == "trajectory":
                inputs_shared["trajectory"] = data.get("trajectory")
            elif extra_input == "ego_vel":
                inputs_shared["ego_vel"] = data.get("ego_vel")
            elif extra_input == "history_positions":
                inputs_shared["history_positions"] = data.get("history_positions")
            else:
                inputs_shared[extra_input] = data[extra_input]

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit,
                self.pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )
        return {**inputs_shared, **inputs_posi}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs, return_loss_breakdown=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNIVERSE on NAVSIM v1.")

    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--navsim_log_path", type=str, required=True)
    parser.add_argument("--sensor_blobs_path", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--use_cache_only", action="store_true")
    parser.add_argument("--force_cache_computation", action="store_true")
    parser.add_argument("--train_log_names", type=str, default=None)
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=None)
    parser.add_argument("--skip_missing_files", action="store_true")
    parser.add_argument("--print_navsim_tokens", action="store_true")

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_history_frames", type=int, default=5)
    parser.add_argument("--num_future_frames", type=int, default=8)
    parser.add_argument("--surround_view", action="store_true")
    parser.add_argument("--target_fps", type=int, default=2)
    parser.add_argument("--use_focal_unification", action="store_true")
    parser.add_argument("--focal_target_px", type=str, default="")
    parser.add_argument("--focal_stats_samples", type=int, default=0)

    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--full_ckpt", type=str, default=None)
    parser.add_argument("--trainable_models", type=str, default=None)
    parser.add_argument("--lora_base_model", type=str, default=None)
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_checkpoint", type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataset_num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", choices=["cosine", "linear", "constant"])
    parser.add_argument("--find_unused_parameters", dest="find_unused_parameters", action="store_true")
    parser.add_argument("--no_find_unused_parameters", dest="find_unused_parameters", action="store_false")
    parser.set_defaults(find_unused_parameters=True)
    parser.add_argument("--gradient_clip_norm", type=float, default=None)
    parser.add_argument("--ddp_timeout_seconds", type=int, default=1800)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_start_factor", type=float, default=0.01)
    parser.add_argument("--log_every_steps", type=int, default=50)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_update_after_step", type=int, default=0)
    parser.add_argument("--ema_update_every", type=int, default=1)
    parser.add_argument("--ema_on_cpu", action="store_true")
    parser.add_argument("--save_ema", action="store_true")
    parser.add_argument("--save_raw_ckpt", dest="save_raw_ckpt", action="store_true")
    parser.add_argument("--no_save_raw_ckpt", dest="save_raw_ckpt", action="store_false")
    parser.set_defaults(save_raw_ckpt=True)

    parser.add_argument("--train_future_video_noise_only", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--no_train_future_video_noise_only", dest="train_future_video_noise_only", action="store_false")
    parser.add_argument("--infer_replace_history_latents_before_decode", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument(
        "--no_infer_replace_history_latents_before_decode",
        dest="infer_replace_history_latents_before_decode",
        action="store_false",
    )
    parser.add_argument("--use_mixed_latent_attention_mask", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument(
        "--no_use_mixed_latent_attention_mask",
        dest="use_mixed_latent_attention_mask",
        action="store_false",
    )
    parser.add_argument("--infer_trajectory_only", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--no_infer_trajectory_only", dest="infer_trajectory_only", action="store_false")
    parser.add_argument("--use_gradient_checkpointing", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--no_use_gradient_checkpointing", dest="use_gradient_checkpointing", action="store_false")
    parser.add_argument("--use_gradient_checkpointing_offload", action="store_true")

    parser.add_argument("--output_path", type=str, default="./outputs/universe/train_navsim_v1")
    parser.add_argument("--train_log_file", type=str, default="train.log")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--auto_eval", dest="auto_eval", action="store_true")
    parser.add_argument("--no_auto_eval", dest="auto_eval", action="store_false")
    parser.set_defaults(auto_eval=False)
    parser.add_argument("--auto_eval_ckpt_kind", type=str, default="ema", choices=["raw", "ema", "all"])
    parser.add_argument("--auto_eval_strict", action="store_true")
    parser.add_argument("--infer_all_output_root", type=str, default=None)

    parser.add_argument("--extra_inputs", default="longcat_video,trajectory,ego_vel")
    parser.add_argument("--use_trajectory", action="store_true")
    parser.add_argument(
        "--trajectory_condition_mode",
        type=str,
        default="velocity",
        choices=["auto", "history", "velocity"],
    )
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.lora_base_model is not None and args.lora_base_model.strip().lower() in {"none", "null", ""}:
        args.lora_base_model = None
    if int(args.target_fps) <= 0:
        raise ValueError(f"--target_fps must be > 0, got {args.target_fps}")
    if args.use_trajectory and args.trainable_models is None:
        args.trainable_models = "trajectory_encoder,trajectory_head"

    extra_inputs = [x.strip() for x in args.extra_inputs.split(",") if x.strip()] if args.extra_inputs else []
    if args.use_trajectory and "trajectory" not in extra_inputs:
        extra_inputs.append("trajectory")
    if "ego_vel" not in extra_inputs:
        extra_inputs.append("ego_vel")
    if args.trajectory_condition_mode in {"auto", "history"} and "history_positions" not in extra_inputs:
        extra_inputs.append("history_positions")
    if args.trajectory_condition_mode == "velocity":
        extra_inputs = [x for x in extra_inputs if x != "history_positions"]
    if "longcat_video" not in extra_inputs:
        extra_inputs.append("longcat_video")
    args.extra_inputs = ",".join(extra_inputs) if extra_inputs else None

    _enable_train_log_capture(args.output_path, args.train_log_file)
    os.makedirs(args.output_path, exist_ok=True)

    train_logs = [x.strip() for x in args.train_log_names.split(",") if x.strip()] if args.train_log_names else None
    dataset = NavsimUNIVERSEDataset(
        NavsimUNIVERSEConfig(
            repo_root=args.repo_root,
            navsim_log_path=args.navsim_log_path,
            sensor_blobs_path=args.sensor_blobs_path,
            cache_path=args.cache_path,
            use_cache_only=args.use_cache_only,
            force_cache_computation=args.force_cache_computation,
            num_history_frames=args.num_history_frames,
            num_future_frames=args.num_future_frames,
            frame_interval=args.frame_interval,
            train_log_names=train_logs,
            max_scenes=args.max_scenes,
            image_height=args.height,
            image_width=args.width,
            surround_view=args.surround_view,
            use_focal_unification=args.use_focal_unification,
            focal_target_px=args.focal_target_px,
            focal_stats_samples=args.focal_stats_samples,
            skip_missing_files=args.skip_missing_files,
            quiet_scene_loader=not args.print_navsim_tokens,
        ),
        split="train",
    )
    if bool(args.use_focal_unification):
        if bool(args.surround_view):
            if _is_rank0():
                print("[train][focal][warn] focal unification is disabled for surround_view mosaics.")
            args.use_focal_unification = False
        else:
            target_raw = str(args.focal_target_px or "").strip()
            target_is_auto = target_raw == "" or target_raw.lower() == "auto"
            stats: Dict[str, float] = {}
            if target_is_auto or int(args.focal_stats_samples) > 0:
                stats_samples = int(args.focal_stats_samples) if int(args.focal_stats_samples) > 0 else 256
                stats = dataset.collect_focal_stats(stats_samples)
                if _is_rank0():
                    print(f"[train][focal] stats samples={stats_samples} {_format_focal_stats(stats)}")
            if target_is_auto:
                target = float(stats.get("effective_focal_mean_median", 0.0))
                if target <= 0.0:
                    raise RuntimeError(
                        "Cannot resolve --focal_target_px auto: no valid camera intrinsics were found. "
                        "Pass a numeric --focal_target_px or disable --use_focal_unification."
                    )
                args.focal_target_px = f"{target:.6g}"
            dataset.set_focal_target_px(args.focal_target_px)
            if _is_rank0():
                print(
                    "[train][focal] enabled "
                    f"target_focal_px={args.focal_target_px} output={args.width}x{args.height}"
                )

    with open(os.path.join(args.output_path, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model = UNIVERSENavsimTrainingModule(
        local_model_path=args.local_model_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        target_fps=args.target_fps,
        negative_prompt=args.negative_prompt,
        use_trajectory=args.use_trajectory,
        train_future_video_noise_only=args.train_future_video_noise_only,
        infer_replace_history_latents_before_decode=args.infer_replace_history_latents_before_decode,
        use_mixed_latent_attention_mask=args.use_mixed_latent_attention_mask,
        infer_trajectory_only=args.infer_trajectory_only,
        trajectory_condition_mode=args.trajectory_condition_mode,
        num_history_frames=args.num_history_frames,
    )

    if args.full_ckpt:
        if not os.path.exists(args.full_ckpt):
            raise FileNotFoundError(f"full_ckpt not found: {args.full_ckpt}")
        print(f"[train] loading full checkpoint: {args.full_ckpt}")
        state_dict = _normalize_train_ckpt_keys(load_state_dict(args.full_ckpt))
        missing, unexpected = model.pipe.load_state_dict(state_dict, strict=False)
        print(f"[train] full_ckpt loaded: keys={len(state_dict)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print(f"[train] full_ckpt missing summary: {_summarize_ckpt_keys(list(missing))}")
        if len(unexpected) > 0:
            print(f"[train] full_ckpt unexpected summary: {_summarize_ckpt_keys(list(unexpected))}")

    remove_prefix = "pipe.dit." if args.lora_base_model is not None else "pipe."
    logger_cls = InProcessAutoEvalModelLogger if args.auto_eval else ModelLogger
    logger_kwargs = {}
    if args.auto_eval:
        logger_kwargs.update(
            train_args=args,
            auto_eval=True,
            auto_eval_ckpt_kind=args.auto_eval_ckpt_kind,
            auto_eval_strict=args.auto_eval_strict,
            infer_all_output_root=args.infer_all_output_root,
        )
    model_logger = logger_cls(
        args.output_path,
        remove_prefix_in_ckpt=remove_prefix,
        save_raw_ckpt=args.save_raw_ckpt,
        save_ema_ckpt=args.save_ema,
        **logger_kwargs,
    )

    print(
        "[train] navsim:",
        f"samples={len(dataset)}",
        f"history={args.num_history_frames}",
        f"future={args.num_future_frames}",
        f"target_fps={args.target_fps}",
        f"trajectory={bool(args.use_trajectory)}",
        f"condition_mode={args.trajectory_condition_mode}",
        f"future_video_loss_only={bool(args.train_future_video_noise_only)}",
        f"mixed_attention_mask={bool(args.use_mixed_latent_attention_mask)}",
        f"infer_trajectory_only={bool(args.infer_trajectory_only)}",
        f"auto_eval={bool(args.auto_eval)}",
    )
    launch_training_task(dataset, model, model_logger, args=args, save_steps=args.save_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
