"""
NavSIM PDM evaluation for Wan video pipeline.
Reads NavSIM data directly (SceneLoader) and computes PDM scores from predicted trajectories.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from examples.wanvideo.driveva_infer.navsim_eval_viz import (
    format_viz_index_prefix,
    save_bev_with_agent_artifacts,
    save_viz_video,
)
try:
    from examples.wanvideo.UNIVERSE_infer.navsim_dataset import (
        FocalUnificationConfig,
        _build_prompt_fixed,
        _resolve_video_pil,
        _ensure_navsim_importable,
        DEFAULT_NEGATIVE_PROMPT,
        one_hot_to_cmd,
    )
except ImportError:
    from navsim_dataset import (
        FocalUnificationConfig,
        _build_prompt_fixed,
        _resolve_video_pil,
        _ensure_navsim_importable,
        DEFAULT_NEGATIVE_PROMPT,
        one_hot_to_cmd,
    )

DEFAULT_VIZ_PLOT_HEIGHT = 420


@dataclass
class _PDMResultCompat:
    """Normalized PDM result used by this infer script across NavSIM versions."""

    no_at_fault_collisions: float = float("nan")
    drivable_area_compliance: float = float("nan")
    driving_direction_compliance: float = float("nan")
    traffic_light_compliance: float = float("nan")
    ego_progress: float = float("nan")
    time_to_collision_within_bound: float = float("nan")
    lane_keeping: float = float("nan")
    history_comfort: float = float("nan")
    comfort: float = float("nan")
    multiplicative_metrics_prod: float = float("nan")
    weighted_metrics: Any = None
    weighted_metrics_array: Any = None
    pdm_score: float = float("nan")
    score: float = float("nan")


def _ensure_navsim_scene_frame_type_compat() -> bool:
    """Inject SceneFrameType into navsim.common.enums when it is missing."""
    try:
        import navsim.common.enums as navsim_enums
    except Exception:
        return False
    if hasattr(navsim_enums, "SceneFrameType"):
        return False

    class SceneFrameType(IntEnum):
        ORIGINAL = 0
        SYNTHETIC = 1

    navsim_enums.SceneFrameType = SceneFrameType
    return True


def _ensure_navsim_map_parameters_compat() -> bool:
    """Inject MapParameters into navsim.planning.metric_caching.metric_cache when missing."""
    try:
        import navsim.planning.metric_caching.metric_cache as metric_cache_mod
    except Exception:
        return False
    if hasattr(metric_cache_mod, "MapParameters"):
        return False

    @dataclass
    class MapParameters:
        map_root: str
        map_version: str
        map_name: str

    metric_cache_mod.MapParameters = MapParameters
    return True


def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wan video model on NavSIM with PDM scoring.")

    # Data args
    parser.add_argument("--repo_root", type=str, required=True)
    parser.add_argument("--navsim_log_path", type=str, required=True)
    parser.add_argument("--sensor_blobs_path", type=str, required=True)
    parser.add_argument("--metric_cache_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_names", type=str, default=None, help="comma separated")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument(
        "--scene_filter_yaml",
        type=str,
        default=None,
        help="Optional UNIVERSE SceneFilter yaml to align log_names/tokens and filter params.",
    )
    parser.add_argument(
        "--scene_filter_yaml_filter_only",
        nargs="?",
        const=True,
        default=False,
        type=_str2bool,
        help=(
            "Only use scene_filter_yaml for log_names/tokens filtering; do not override "
            "frame/route parameters. Accepts optional bool value (e.g. 1/0, true/false)."
        ),
    )
    parser.add_argument(
        "--no_scene_filter_yaml_filter_only",
        dest="scene_filter_yaml_filter_only",
        action="store_false",
        help="Disable scene_filter_yaml filter-only mode.",
    )
    parser.add_argument("--token_offset", type=int, default=0, help="Skip first N tokens after rank split.")
    parser.add_argument("--max_eval_tokens", type=int, default=None, help="Evaluate at most this many tokens after offset.")
    parser.add_argument(
        "--num_eval_shards",
        type=int,
        default=None,
        help=(
            "Optional number of eval shards inside current distributed world. "
            "Default uses WORLD_SIZE. Supports up to WORLD_SIZE*32; when > WORLD_SIZE, each rank handles "
            "multiple virtual shards."
        ),
    )
    parser.add_argument("--resume_csv", type=str, default=None, help="CSV path to skip already evaluated tokens in column `token`.")
    parser.add_argument("--print_tokens", action="store_true", help="Print token during evaluation loop.")
    parser.add_argument("--show_eval_progress", dest="show_eval_progress", action="store_true", help="Show eval progress with elapsed/ETA on rank0.")
    parser.add_argument("--no_show_eval_progress", dest="show_eval_progress", action="store_false", help="Disable eval progress bar.")
    parser.add_argument("--show_denoise_progress", action="store_true", help="Show per-token denoise tqdm.")
    parser.add_argument("--show_vae_progress", action="store_true", help="Show VAE tiled decode/encode tqdm.")
    parser.add_argument(
        "--use_mixed_latent_attention_mask",
        dest="use_mixed_latent_attention_mask",
        nargs="?",
        const=True,
        default=True,
        type=_str2bool,
        help="Enable mixed visibility masking between video and trajectory latent groups.",
    )
    parser.add_argument(
        "--no_use_mixed_latent_attention_mask",
        dest="use_mixed_latent_attention_mask",
        action="store_false",
        help="Disable mixed visibility masking between video and trajectory latent groups.",
    )
    parser.add_argument(
        "--infer_output_mode",
        type=str,
        default=None,
        choices=["trajectory", "video", "both"],
        help="Pipeline return mode. Defaults to the legacy video+trajectory tuple.",
    )
    parser.add_argument(
        "--infer_trajectory_only",
        dest="infer_trajectory_only",
        action="store_true",
        help="Skip video decode when visualization is disabled; NavSIM PDM scoring only needs trajectory output.",
    )
    parser.add_argument(
        "--no_infer_trajectory_only",
        dest="infer_trajectory_only",
        action="store_false",
        help="Decode video during NavSIM eval even when visualization is disabled.",
    )
    parser.add_argument("--debug_prompt_steps", type=int, default=30, help="Print positive/negative prompt for first N inference steps on rank0.")
    parser.add_argument("--print_errors", dest="print_errors", action="store_true", help="Print token-level exceptions during evaluation.")
    parser.add_argument("--no_print_errors", dest="print_errors", action="store_false", help="Disable token-level exception prints.")
    parser.add_argument(
        "--print_ckpt_missing",
        action="store_true",
        help="Print missing/unexpected keys when loading full_ckpt.",
    )
    parser.add_argument(
        "--max_print_ckpt_keys",
        type=int,
        default=300,
        help="Maximum keys to print for missing/unexpected when --print_ckpt_missing is set. <=0 prints all.",
    )
    parser.set_defaults(print_errors=True, show_eval_progress=True, infer_trajectory_only=False)
    parser.add_argument("--save_viz", action="store_true", help="Save visualization video per token.")
    parser.add_argument("--viz_dir", type=str, default=None, help="Directory for visualization outputs. Default: <output_dir>/viz")
    parser.add_argument(
        "--viz_total_tokens",
        type=int,
        default=100,
        help="Total number of visualization tokens across all active ranks. Set <=0 to fallback to --viz_max_tokens per rank.",
    )
    parser.add_argument("--viz_max_tokens", type=int, default=20, help="Max number of tokens to visualize per rank.")

    # Video args
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_history_frames", type=int, default=5)
    parser.add_argument("--num_future_frames", type=int, default=10)
    parser.add_argument(
        "--model_future_frames",
        type=int,
        default=8,
        help="Future trajectory points produced by model (e.g., 8 at 2Hz => 4s).",
    )
    parser.add_argument(
        "--scene_future_extra_seconds",
        type=float,
        default=1.0,
        help="Ensure scene future horizon is at least model horizon + this many seconds (for TTC look-ahead).",
    )
    parser.add_argument(
        "--pdm_num_poses",
        type=int,
        default=40,
        help="PDM proposal sampling num_poses (e.g., 40).",
    )
    parser.add_argument(
        "--pdm_interval_length",
        type=float,
        default=0.1,
        help="PDM proposal sampling interval length in seconds (e.g., 0.1).",
    )
    parser.add_argument(
        "--traffic_agents_policy",
        type=str,
        default="non_reactive",
        choices=["non_reactive", "log_replay", "constant_velocity"],
        help="Background traffic policy for NavSIM PDM scoring.",
    )
    parser.add_argument(
        "--legacy_simulate_traffic_agents",
        dest="legacy_simulate_traffic_agents",
        action="store_true",
        help=(
            "Replay a traffic-agent policy before scoring legacy NavSIM v1 metric caches. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--no_legacy_simulate_traffic_agents",
        dest="legacy_simulate_traffic_agents",
        action="store_false",
        help="Use the metric-cache observation directly for legacy NavSIM v1 caches.",
    )
    parser.add_argument("--surround_view", action="store_true")
    parser.add_argument("--target_fps", type=int, default=2)
    parser.add_argument("--use_focal_unification", action="store_true")
    parser.add_argument("--focal_target_px", type=str, default="")
    parser.add_argument("--print_alignment_params", dest="print_alignment_params", action="store_true")
    parser.add_argument("--no_print_alignment_params", dest="print_alignment_params", action="store_false")
    parser.add_argument(
        "--enable_nuscenes_metrics",
        action="store_true",
        help="Compute nuScenes-style L2 (m) and Collision (%%) metrics at requested horizons.",
    )
    parser.add_argument(
        "--nuscenes_metric_horizons_s",
        type=str,
        default="1,2,3",
        help="Comma/space-separated horizon seconds for nuScenes metrics, e.g. '1,2,3'.",
    )
    # Model args
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--full_ckpt", type=str, required=True, help="Path to full checkpoint (.safetensors).")
    parser.add_argument("--num_inference_steps", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--infer_replace_history_latents_before_decode",
        dest="infer_replace_history_latents_before_decode",
        nargs="?",
        const=True,
        default=True,
        type=_str2bool,
        help=(
            "Before VAE decode, replace history latent slots with clean longcat history latents. "
            "Accepts optional bool value (e.g. 1/0, true/false)."
        ),
    )
    parser.add_argument(
        "--no_infer_replace_history_latents_before_decode",
        dest="infer_replace_history_latents_before_decode",
        action="store_false",
        help="Disable clean-history latent replacement before decode.",
    )
    parser.add_argument(
        "--trajectory_condition_mode",
        type=str,
        default="velocity",
        choices=["auto", "history", "velocity"],
        help="Trajectory prefix conditioning mode: auto (prefer history), history-only, or velocity-only.",
    )
    parser.set_defaults(print_alignment_params=True, legacy_simulate_traffic_agents=False)

    return parser.parse_args(argv)


def _init_distributed() -> Dict[str, int]:
    if torch.distributed.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        rank = int(torch.distributed.get_rank())
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(torch.distributed.get_world_size())
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _gather_results(results: List[Dict[str, Any]], device: torch.device, rank: int, world_size: int) -> Optional[List[Dict[str, Any]]]:
    results = [_sanitize_result_row(row) for row in results]
    if world_size == 1:
        return results
    if hasattr(torch.distributed, "all_gather_object"):
        gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, results)
        if rank == 0:
            final: List[Dict[str, Any]] = []
            for part in gathered:
                final.extend(part)
            return final
        return None

    payload = pickle.dumps(results)
    tensor = torch.ByteTensor(list(payload)).to(device)
    size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.zeros_like(size) for _ in range(world_size)]
    torch.distributed.all_gather(size_list, size)
    max_size = int(max([s.item() for s in size_list]))
    if tensor.numel() < max_size:
        tensor = torch.cat([tensor, torch.zeros(max_size - tensor.numel(), dtype=torch.uint8, device=device)])
    gathered = [torch.empty(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    if rank == 0:
        final: List[Dict[str, Any]] = []
        for t, s in zip(gathered, size_list):
            buf = t[: int(s.item())].cpu().numpy().tobytes()
            final.extend(pickle.loads(buf))
        return final
    return None


def _resolve_eval_shards(*, rank: int, world_size: int, num_eval_shards: Optional[int]) -> Dict[str, Any]:
    if int(world_size) <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if num_eval_shards is None:
        active_shards = int(world_size)
    else:
        requested = int(num_eval_shards)
        if requested <= 0:
            raise ValueError(f"num_eval_shards must be > 0 when set, got {num_eval_shards}")
        max_supported = int(world_size) * 32
        if requested > max_supported:
            raise ValueError(
                f"num_eval_shards must be <= world_size*32 ({max_supported}), got {requested}"
            )
        active_shards = requested
    local_shard_ids = list(range(int(rank), int(active_shards), int(world_size)))
    return {
        "active_shards": int(active_shards),
        "local_shard_ids": local_shard_ids,
        "idle_rank": len(local_shard_ids) == 0,
    }


def _safe_traj_np(traj: np.ndarray, expected_len: int) -> np.ndarray:
    if traj.shape[0] == expected_len:
        return traj
    if traj.shape[0] > expected_len:
        return traj[:expected_len]
    # pad with last pose
    pad = np.repeat(traj[-1:], expected_len - traj.shape[0], axis=0)
    return np.concatenate([traj, pad], axis=0)


def _parse_metric_horizons_s(raw: str) -> List[float]:
    chunks = re.split(r"[,\s]+", str(raw).strip())
    horizons: List[float] = []
    for chunk in chunks:
        if not chunk:
            continue
        horizon_s = float(chunk)
        if horizon_s <= 0:
            raise ValueError(f"metric horizon must be > 0, got {horizon_s}")
        horizons.append(horizon_s)
    if not horizons:
        raise ValueError(f"invalid --nuscenes_metric_horizons_s: {raw!r}")
    horizons = sorted(horizons)
    deduped: List[float] = []
    for horizon_s in horizons:
        if not deduped or abs(deduped[-1] - horizon_s) > 1e-6:
            deduped.append(horizon_s)
    return deduped


def _format_horizon_label(horizon_s: float) -> str:
    rounded = int(round(float(horizon_s)))
    if abs(float(horizon_s) - float(rounded)) <= 1e-6:
        return f"{rounded}s"
    label = f"{float(horizon_s):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{label}s"


def _load_nuscenes_final_distance() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    devkit_sdk = Path(__file__).resolve().parents[3] / "third_party" / "nuscenes-devkit" / "python-sdk"
    if devkit_sdk.exists():
        sdk_path = str(devkit_sdk)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
    try:
        from nuscenes.eval.prediction.metrics import final_distances
    except Exception as exc:
        raise RuntimeError(
            "Failed to import nuscenes-devkit prediction metrics. "
            "Please add <repo>/third_party/nuscenes-devkit/python-sdk to PYTHONPATH."
        ) from exc
    return final_distances


def _interp_xy_at_horizon(traj_xy: np.ndarray, dt_s: float, horizon_s: float) -> Optional[np.ndarray]:
    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    if traj_xy.ndim != 2 or traj_xy.shape[0] == 0 or traj_xy.shape[1] < 2:
        return None
    dt_s = float(dt_s)
    horizon_s = float(horizon_s)
    if dt_s <= 0.0 or horizon_s <= 0.0:
        return None
    times = np.arange(1, traj_xy.shape[0] + 1, dtype=np.float64) * dt_s
    if horizon_s > float(times[-1]) + 1e-6:
        return None
    x = float(np.interp(horizon_s, times, traj_xy[:, 0]))
    y = float(np.interp(horizon_s, times, traj_xy[:, 1]))
    return np.array([x, y], dtype=np.float32)


def _compute_nuscenes_l2_and_collision_metrics(
    *,
    pred_traj_xy: np.ndarray,
    pred_dt_s: float,
    gt_traj_xy: np.ndarray,
    gt_dt_s: float,
    horizons_s: List[float],
    final_distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    pred_collision_flags: Optional[np.ndarray],
    gt_collision_flags: Optional[np.ndarray],
    collision_dt_s: Optional[float],
    collision_s: Optional[float] = None,
) -> Dict[str, float]:
    metric_row: Dict[str, float] = {}
    pred_collision_flags_np = (
        np.asarray(pred_collision_flags, dtype=np.float32).reshape(-1)
        if pred_collision_flags is not None
        else None
    )
    gt_collision_flags_np = (
        np.asarray(gt_collision_flags, dtype=np.float32).reshape(-1)
        if gt_collision_flags is not None
        else None
    )
    collision_dt = float(collision_dt_s) if collision_dt_s is not None else float("nan")

    for horizon_s in horizons_s:
        label = _format_horizon_label(horizon_s)
        l2_key = f"nuscenes_l2_{label}_m"
        collision_key = f"nuscenes_collision_{label}_pct"

        pred_xy = _interp_xy_at_horizon(pred_traj_xy, pred_dt_s, horizon_s)
        gt_xy = _interp_xy_at_horizon(gt_traj_xy, gt_dt_s, horizon_s)
        if pred_xy is None or gt_xy is None:
            metric_row[l2_key] = float("nan")
            metric_row[collision_key] = float("nan")
            continue
        else:
            pred_stack = np.asarray([[pred_xy]], dtype=np.float32)
            gt_stack = np.asarray([[gt_xy]], dtype=np.float32)
            l2_val = np.asarray(final_distance_fn(pred_stack, gt_stack)).reshape(-1)
            metric_row[l2_key] = float(l2_val[0]) if l2_val.size > 0 else float("nan")

        collision_written = False
        if (
            pred_collision_flags_np is not None
            and pred_collision_flags_np.size > 1
            and np.isfinite(collision_dt)
            and collision_dt > 0.0
        ):
            times_s = np.arange(pred_collision_flags_np.size, dtype=np.float64) * collision_dt
            horizon_indices = np.where((times_s > 1e-6) & (times_s <= float(horizon_s) + 1e-6))[0]
            collision_vals: List[float] = []
            for step_idx in horizon_indices.tolist():
                pred_coll = float(pred_collision_flags_np[step_idx])
                if not np.isfinite(pred_coll):
                    continue
                if (
                    gt_collision_flags_np is not None
                    and step_idx < gt_collision_flags_np.size
                    and np.isfinite(gt_collision_flags_np[step_idx])
                    and float(gt_collision_flags_np[step_idx]) > 0.5
                ):
                    pred_coll = 0.0
                collision_vals.append(pred_coll)
            if len(collision_vals) > 0:
                metric_row[collision_key] = float(
                    np.mean(np.asarray(collision_vals, dtype=np.float32)) * 100.0
                )
            else:
                metric_row[collision_key] = float("nan")
            collision_written = True

        if not collision_written:
            metric_row[collision_key] = (
                100.0
                if (
                    collision_s is not None
                    and np.isfinite(float(collision_s))
                    and float(collision_s) <= float(horizon_s) + 1e-6
                )
                else 0.0
            )
    return metric_row


def _extract_pdm_collision_flags(
    *,
    scorer: Any,
    proposal_idx: int,
) -> Optional[np.ndarray]:
    try:
        observation = scorer._observation
        ego_polygons = scorer._ego_polygons
        proposal_sampling = scorer.proposal_sampling
    except Exception:
        return None

    if (
        observation is None
        or ego_polygons is None
    ):
        return None
    if proposal_idx < 0 or proposal_idx >= int(ego_polygons.shape[0]):
        return None

    n_steps = int(
        min(
            int(proposal_sampling.num_poses) + 1,
            int(ego_polygons.shape[1]),
        )
    )
    if n_steps <= 0:
        return None

    collision_flags = np.zeros(n_steps, dtype=np.float32)
    red_light_token = str(getattr(observation, "red_light_token", ""))

    for time_idx in range(n_steps):
        try:
            ego_polygon = ego_polygons[proposal_idx, time_idx]
            intersecting = observation[time_idx].query(
                np.asarray([ego_polygon], dtype=object), predicate="intersects"
            )
        except Exception:
            continue

        if (
            not isinstance(intersecting, (list, tuple))
            or len(intersecting) < 2
            or len(intersecting[1]) == 0
        ):
            continue

        for geometry_idx in np.asarray(intersecting[1], dtype=np.int64).reshape(-1).tolist():
            try:
                token = observation[time_idx].tokens[geometry_idx]
            except Exception:
                continue

            if red_light_token and red_light_token in token:
                continue

            collision_flags[time_idx] = 1.0
            break

    return collision_flags


def _load_scene_filter_yaml(path: Optional[str], *, filter_only: bool = False) -> Dict[str, Any]:
    if not path:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"scene_filter_yaml not found: {yaml_path}")
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required when --scene_filter_yaml is set.") from exc
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if filter_only:
        keys = ("log_names", "tokens")
    else:
        keys = (
            "num_history_frames",
            "num_future_frames",
            "frame_interval",
            "has_route",
            "log_names",
            "tokens",
        )
    return {k: raw[k] for k in keys if k in raw}


def _sampling_horizon_s(sampling: Any) -> float:
    num_poses = getattr(sampling, "num_poses", None)
    interval = getattr(sampling, "interval_length", None)
    if num_poses is None or interval is None:
        return float("nan")
    return float(num_poses) * float(interval)


def _fmt_float(x: float) -> str:
    if np.isfinite(x):
        return f"{x:.6f}"
    if np.isinf(x):
        return "inf"
    return "nan"


def _result_value_to_python(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, memoryview):
        return value.tobytes().hex()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _result_value_to_python(asdict(value))
    if isinstance(value, dict):
        return {str(k): _result_value_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_result_value_to_python(item) for item in value]
    if isinstance(value, set):
        return [_result_value_to_python(item) for item in sorted(value, key=str)]
    try:
        pickle.dumps(value)
        return value
    except Exception:
        return str(value)


def _sanitize_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(key): _result_value_to_python(value) for key, value in row.items()}


def _metric_value_to_python(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, memoryview):
        return value.tobytes().hex()
    return value


def _as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return default
            value = value.reshape(-1)[0]
        return float(value)
    except Exception:
        return default


def _as_finite_float(value: Any, default: float = float("nan")) -> float:
    value_f = _as_float(value, default=default)
    return value_f if np.isfinite(value_f) else default


def _get_metric_float(result: Any, *names: str, default: float = float("nan")) -> float:
    for name in names:
        if hasattr(result, name):
            value = _as_float(getattr(result, name), default=default)
            if not np.isnan(value):
                return value
    return default


def _row_from_pdm_score_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, np.generic):
        return {"score": output.item()}
    if isinstance(output, (int, float)):
        return {"score": float(output)}
    if isinstance(output, np.ndarray) and output.ndim == 0:
        return {"score": output.item()}
    if is_dataclass(output):
        return asdict(output)
    if isinstance(output, dict):
        return dict(output)
    if hasattr(output, "iloc") and hasattr(output, "columns"):
        if len(output) <= 0:
            return {}
        return dict(output.iloc[0].to_dict())
    return {
        name: getattr(output, name)
        for name in dir(output)
        if not name.startswith("_") and not callable(getattr(output, name, None))
    }


def _normalize_pdm_result(output: Any) -> Tuple[_PDMResultCompat, Dict[str, Any]]:
    row = {k: _metric_value_to_python(v) for k, v in _row_from_pdm_score_output(output).items()}

    has_traffic_light = "traffic_light_compliance" in row
    if "pdm_score" not in row and "score" in row:
        row["pdm_score"] = row["score"]
    if "score" not in row and "pdm_score" in row:
        row["score"] = row["pdm_score"]
    if "history_comfort" not in row and "comfort" in row:
        row["history_comfort"] = row["comfort"]
    if "comfort" not in row and "history_comfort" in row:
        row["comfort"] = row["history_comfort"]
    if "traffic_light_compliance" not in row:
        row["traffic_light_compliance"] = float("nan")
    if "lane_keeping" not in row:
        row["lane_keeping"] = float("nan")

    if "multiplicative_metrics_prod" not in row:
        no_collision = _as_finite_float(row.get("no_at_fault_collisions"), default=1.0)
        drivable = _as_finite_float(row.get("drivable_area_compliance"), default=1.0)
        traffic_light = _as_finite_float(row.get("traffic_light_compliance"), default=1.0) if has_traffic_light else 1.0
        driving_direction = _as_finite_float(row.get("driving_direction_compliance"), default=1.0)
        row["multiplicative_metrics_prod"] = no_collision * drivable * traffic_light * driving_direction

    result = _PDMResultCompat(
        no_at_fault_collisions=_as_float(row.get("no_at_fault_collisions")),
        drivable_area_compliance=_as_float(row.get("drivable_area_compliance")),
        driving_direction_compliance=_as_float(row.get("driving_direction_compliance")),
        traffic_light_compliance=_as_float(row.get("traffic_light_compliance")),
        ego_progress=_as_float(row.get("ego_progress")),
        time_to_collision_within_bound=_as_float(row.get("time_to_collision_within_bound")),
        lane_keeping=_as_float(row.get("lane_keeping")),
        history_comfort=_as_float(row.get("history_comfort")),
        comfort=_as_float(row.get("comfort")),
        multiplicative_metrics_prod=_as_float(row.get("multiplicative_metrics_prod")),
        weighted_metrics=row.get("weighted_metrics"),
        weighted_metrics_array=row.get("weighted_metrics_array"),
        pdm_score=_as_float(row.get("pdm_score")),
        score=_as_float(row.get("score")),
    )

    normalized = asdict(result)
    for key, value in row.items():
        normalized.setdefault(key, value)
    return result, normalized


def _build_traffic_agents_policy(policy_name: str, proposal_sampling: Any) -> Any:
    normalized = str(policy_name).strip().lower()
    if normalized in {"non_reactive", "log_replay"}:
        from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents

        return LogReplayTrafficAgents(proposal_sampling)
    if normalized == "constant_velocity":
        from navsim.traffic_agents_policies.constant_velocity_traffic_agents import ConstantVelocityTrafficAgents

        return ConstantVelocityTrafficAgents(proposal_sampling)
    raise ValueError(f"Unsupported traffic_agents_policy={policy_name!r}")


def _as_detection_track_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [value]


def _iter_detection_track_objects(detection_track: Any) -> List[Any]:
    tracked_objects = getattr(detection_track, "tracked_objects", None)
    if tracked_objects is None:
        return []
    objects = getattr(tracked_objects, "tracked_objects", tracked_objects)
    try:
        return list(objects)
    except TypeError:
        return []


def _tracked_object_token(tracked_object: Any) -> Optional[str]:
    token = getattr(tracked_object, "track_token", None)
    if token is None:
        metadata = getattr(tracked_object, "metadata", None)
        token = getattr(metadata, "track_token", None)
    return str(token) if token is not None else None


def _rebuild_observation_traffic_light_maps(observation: Any) -> Optional[List[Tuple[List[str], np.ndarray]]]:
    occupancy_maps = getattr(observation, "_occupancy_maps", None)
    if occupancy_maps is None:
        return None
    try:
        if len(occupancy_maps) == 0:
            return None
    except TypeError:
        return None

    red_light_token = str(getattr(observation, "_red_light_token", "red_light"))
    occupancy_maps_tl: List[Tuple[List[str], np.ndarray]] = []
    for occupancy_map in occupancy_maps:
        raw_tokens = getattr(occupancy_map, "tokens", None)
        tokens = list(raw_tokens) if raw_tokens is not None else []
        geometries = getattr(occupancy_map, "_geometries", None)
        if geometries is None:
            occupancy_maps_tl.append(([], np.array([], dtype=np.object_)))
            continue

        tl_indices = [idx for idx, token in enumerate(tokens) if str(token).startswith(red_light_token)]
        tl_tokens = [tokens[idx] for idx in tl_indices]
        tl_polygons = np.asarray([geometries[idx] for idx in tl_indices], dtype=np.object_)
        occupancy_maps_tl.append((tl_tokens, tl_polygons))

    return occupancy_maps_tl


def _rebuild_detection_tracks_from_observation(observation: Any) -> List[Any]:
    occupancy_maps = getattr(observation, "_occupancy_maps", None)
    unique_objects = getattr(observation, "_unique_objects", None)
    if occupancy_maps is None or not isinstance(unique_objects, dict):
        return []
    try:
        if len(occupancy_maps) == 0:
            return []
    except TypeError:
        return []

    try:
        from nuplan.common.actor_state.tracked_objects import TrackedObjects
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    except Exception:
        return []

    red_light_token = str(getattr(observation, "_red_light_token", "red_light"))
    detection_tracks: List[Any] = []
    for occupancy_map in occupancy_maps:
        raw_tokens = getattr(occupancy_map, "tokens", None)
        tokens = list(raw_tokens) if raw_tokens is not None else []
        tracked_objects = []
        for token in tokens:
            token_str = str(token)
            if token_str.startswith(red_light_token):
                continue
            tracked_object = unique_objects.get(token)
            if tracked_object is None:
                tracked_object = unique_objects.get(token_str)
            if tracked_object is not None:
                tracked_objects.append(tracked_object)
        detection_tracks.append(DetectionsTracks(TrackedObjects(tracked_objects)))

    return detection_tracks


def _ensure_metric_cache_pdm_observation_compat(metric_cache: Any) -> None:
    """Patch legacy NavSIM metric-cache pickles for PDM scoring."""
    observation = getattr(metric_cache, "observation", None)
    if observation is None:
        return

    detection_tracks = _as_detection_track_list(getattr(observation, "_detections_tracks", None))
    if not detection_tracks:
        detection_tracks = []
        detection_tracks.extend(_as_detection_track_list(getattr(metric_cache, "current_tracked_objects", None)))
        detection_tracks.extend(_as_detection_track_list(getattr(metric_cache, "future_tracked_objects", None)))
        if not detection_tracks:
            detection_tracks.extend(_as_detection_track_list(getattr(metric_cache, "past_detections_tracks", None)))
        if not detection_tracks:
            detection_tracks.extend(_rebuild_detection_tracks_from_observation(observation))
        if detection_tracks:
            setattr(observation, "_detections_tracks", detection_tracks)
    if detection_tracks:
        if not hasattr(metric_cache, "current_tracked_objects"):
            setattr(metric_cache, "current_tracked_objects", detection_tracks[:1])
        if not hasattr(metric_cache, "future_tracked_objects"):
            setattr(metric_cache, "future_tracked_objects", detection_tracks[1:])

    if detection_tracks and (
        not hasattr(observation, "_unique_objects") or getattr(observation, "_unique_objects", None) is None
    ):
        unique_objects: Dict[str, Any] = {}
        for detection_track in detection_tracks:
            for tracked_object in _iter_detection_track_objects(detection_track):
                token = _tracked_object_token(tracked_object)
                if token is not None and token not in unique_objects:
                    unique_objects[token] = tracked_object
        setattr(observation, "_unique_objects", unique_objects)

    if not hasattr(observation, "_red_light_token"):
        setattr(observation, "_red_light_token", "red_light")
    if not hasattr(observation, "_collided_track_ids"):
        setattr(observation, "_collided_track_ids", [])
    if not hasattr(observation, "_occupancy_maps_tl") or getattr(observation, "_occupancy_maps_tl", None) is None:
        setattr(observation, "_occupancy_maps_tl", _rebuild_observation_traffic_light_maps(observation))
    if detection_tracks and not getattr(observation, "_initialized", False):
        setattr(observation, "_initialized", True)


def _call_pdm_score_legacy_metric_cache(
    *,
    pdm_score_fn: Callable[..., Any],
    metric_cache: Any,
    model_trajectory: Any,
    future_sampling: Any,
    simulator: Any,
    scorer: Any,
    traffic_agents_policy: Any,
    legacy_simulate_traffic_agents: bool,
) -> Tuple[Any, Optional[np.ndarray]]:
    pdm_score_globals = getattr(pdm_score_fn, "__globals__", {})
    transform_trajectory = pdm_score_globals.get("transform_trajectory")
    get_trajectory_as_array = pdm_score_globals.get("get_trajectory_as_array")
    if transform_trajectory is None or get_trajectory_as_array is None:
        raise RuntimeError("NavSIM pdm_score helpers are unavailable for legacy metric cache compatibility.")

    initial_ego_state = metric_cache.ego_state
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        future_sampling,
        initial_ego_state.time_point,
    )
    pred_states = get_trajectory_as_array(
        pred_trajectory,
        future_sampling,
        initial_ego_state.time_point,
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    simulated_agent_detections_tracks = None
    if legacy_simulate_traffic_agents:
        simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(simulated_states[1], metric_cache)
        if len(simulated_agent_detections_tracks) != trajectory_states.shape[1]:
            raise ValueError(
                "Traffic agents policy returned trajectories of invalid length: "
                f"{len(simulated_agent_detections_tracks)} != {trajectory_states.shape[1]}"
            )
        expected_observation_len = int(getattr(metric_cache.observation, "_observation_samples", 0)) + 1
        if expected_observation_len > len(simulated_agent_detections_tracks):
            simulated_agent_detections_tracks = list(simulated_agent_detections_tracks)
            simulated_agent_detections_tracks.extend(
                [simulated_agent_detections_tracks[-1]]
                * (expected_observation_len - len(simulated_agent_detections_tracks))
            )

    score_kwargs: Dict[str, Any] = {
        "states": simulated_states,
        "observation": metric_cache.observation,
        "centerline": metric_cache.centerline,
        "route_lane_ids": metric_cache.route_lane_ids,
        "drivable_area_map": metric_cache.drivable_area_map,
    }
    scorer_params = inspect.signature(scorer.score_proposals).parameters
    if "map_parameters" in scorer_params:
        score_kwargs["map_parameters"] = getattr(metric_cache, "map_parameters", None)
    if simulated_agent_detections_tracks is not None and "simulated_agent_detections_tracks" in scorer_params:
        score_kwargs["simulated_agent_detections_tracks"] = simulated_agent_detections_tracks
    if "human_past_trajectory" in scorer_params:
        score_kwargs["human_past_trajectory"] = getattr(metric_cache, "past_human_trajectory", None)

    pred_idx = 1
    pdm_result = scorer.score_proposals(**score_kwargs)[pred_idx]
    return pdm_result, simulated_states[pred_idx]


def _score_navsim_pdm(
    *,
    pdm_score_fn: Callable[..., Any],
    metric_cache: Any,
    model_trajectory: Any,
    future_sampling: Any,
    simulator: Any,
    scorer: Any,
    traffic_agents_policy: Any,
    legacy_simulate_traffic_agents: bool,
) -> Tuple[_PDMResultCompat, Dict[str, Any], Optional[np.ndarray]]:
    _ensure_metric_cache_pdm_observation_compat(metric_cache)
    kwargs: Dict[str, Any] = {
        "metric_cache": metric_cache,
        "model_trajectory": model_trajectory,
        "future_sampling": future_sampling,
        "simulator": simulator,
        "scorer": scorer,
    }
    if "traffic_agents_policy" in inspect.signature(pdm_score_fn).parameters:
        kwargs["traffic_agents_policy"] = traffic_agents_policy

    try:
        raw = pdm_score_fn(**kwargs)
    except AttributeError as exc:
        msg = str(exc)
        fallback_markers = (
            "map_parameters",
            "past_human_trajectory",
            "_detections_tracks",
            "simulated_agent_detections_tracks",
        )
        if not any(marker in msg for marker in fallback_markers):
            raise
        raw = _call_pdm_score_legacy_metric_cache(
            pdm_score_fn=pdm_score_fn,
            metric_cache=metric_cache,
            model_trajectory=model_trajectory,
            future_sampling=future_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy,
            legacy_simulate_traffic_agents=legacy_simulate_traffic_agents,
        )
    simulated_states = None
    raw_result = raw
    if isinstance(raw, tuple):
        raw_result = raw[0]
        if len(raw) > 1:
            simulated_states = raw[1]

    result, row = _normalize_pdm_result(raw_result)
    return result, row, simulated_states


def _reconstruct_pdm_score_from_result(result: _PDMResultCompat, scorer: Any) -> float:
    reported = _get_metric_float(result, "pdm_score", "score")
    if np.isfinite(reported):
        return reported

    scorer_cfg = getattr(scorer, "_config", None)
    metric_weight_pairs = (
        ("ego_progress", "progress_weight"),
        ("time_to_collision_within_bound", "ttc_weight"),
        ("lane_keeping", "lane_keeping_weight"),
        ("history_comfort", "history_comfort_weight"),
        ("comfort", "comfortable_weight"),
        ("driving_direction_compliance", "driving_direction_weight"),
    )
    weighted_sum = 0.0
    weight_sum = 0.0
    for metric_name, weight_name in metric_weight_pairs:
        weight = _as_float(getattr(scorer_cfg, weight_name, None), default=0.0)
        metric = _get_metric_float(result, metric_name)
        if weight > 0.0 and np.isfinite(metric):
            weighted_sum += metric * weight
            weight_sum += weight

    multiplicative = _get_metric_float(result, "multiplicative_metrics_prod")
    if not np.isfinite(multiplicative):
        multiplicative = (
            _get_metric_float(result, "no_at_fault_collisions", default=1.0)
            * _get_metric_float(result, "drivable_area_compliance", default=1.0)
            * _get_metric_float(result, "traffic_light_compliance", default=1.0)
            * _get_metric_float(result, "driving_direction_compliance", default=1.0)
        )
    if weight_sum <= 0.0:
        return multiplicative
    return multiplicative * (weighted_sum / weight_sum)


def _min_scene_future_frames(
    *,
    model_future_frames: int,
    model_fps: float,
    scene_interval_s: float,
    extra_seconds: float,
) -> int:
    if model_future_frames <= 0:
        raise ValueError(f"model_future_frames must be > 0, got {model_future_frames}")
    if model_fps <= 0:
        raise ValueError(f"model_fps must be > 0, got {model_fps}")
    if scene_interval_s <= 0:
        raise ValueError(f"scene_interval_s must be > 0, got {scene_interval_s}")
    model_horizon_s = float(model_future_frames) / float(model_fps)
    required_horizon_s = model_horizon_s + max(0.0, float(extra_seconds))
    return max(1, int(math.ceil(required_horizon_s / float(scene_interval_s))))


def _print_alignment_summary(
    *,
    rank: int,
    scene_filter: Any,
    model_sampling: Any,
    proposal_sampling: Any,
    scorer: Any,
    navsim_interval_length: float,
    target_fps: int,
) -> None:
    if rank != 0:
        return
    model_h = _sampling_horizon_s(model_sampling)
    pdm_h = _sampling_horizon_s(proposal_sampling)
    navsim_h = float(scene_filter.num_future_frames) * float(navsim_interval_length)
    scorer_cfg = getattr(scorer, "_config", None)
    w_progress = float(getattr(scorer_cfg, "progress_weight", float("nan")))
    w_ttc = float(getattr(scorer_cfg, "ttc_weight", float("nan")))
    w_lane = float(getattr(scorer_cfg, "lane_keeping_weight", float("nan")))
    w_history_comfort = float(getattr(scorer_cfg, "history_comfort_weight", float("nan")))
    w_two_frame_comfort = float(getattr(scorer_cfg, "two_frame_extended_comfort_weight", float("nan")))
    w_comfort = float(getattr(scorer_cfg, "comfortable_weight", float("nan")))
    w_dir = float(getattr(scorer_cfg, "driving_direction_weight", float("nan")))
    print(
        "[eval][align] scene_filter:",
        f"history={scene_filter.num_history_frames}",
        f"future={scene_filter.num_future_frames}",
        f"frame_interval={scene_filter.frame_interval}",
        f"has_route={scene_filter.has_route}",
        f"log_names={0 if scene_filter.log_names is None else len(scene_filter.log_names)}",
        f"tokens={0 if scene_filter.tokens is None else len(scene_filter.tokens)}",
    )
    print(
        "[eval][align] model_sampling:",
        f"num_poses={getattr(model_sampling, 'num_poses', 'na')}",
        f"interval={getattr(model_sampling, 'interval_length', 'na')}",
        f"horizon_s={_fmt_float(model_h)}",
        f"target_fps={target_fps}",
    )
    print(
        "[eval][align] pdm_sampling:",
        f"num_poses={getattr(proposal_sampling, 'num_poses', 'na')}",
        f"interval={getattr(proposal_sampling, 'interval_length', 'na')}",
        f"horizon_s={_fmt_float(pdm_h)}",
    )
    print(
        "[eval][align] scorer_weights:",
        f"progress={_fmt_float(w_progress)}",
        f"ttc={_fmt_float(w_ttc)}",
        f"lane_keeping={_fmt_float(w_lane)}",
        f"history_comfort={_fmt_float(w_history_comfort)}",
        f"two_frame_comfort={_fmt_float(w_two_frame_comfort)}",
        f"comfort={_fmt_float(w_comfort)}",
        f"driving_direction={_fmt_float(w_dir)}",
    )
    print(
        "[eval][align] horizons:",
        f"navsim_scene={_fmt_float(navsim_h)}s",
        f"model={_fmt_float(model_h)}s",
        f"pdm={_fmt_float(pdm_h)}s",
    )


def _print_requested_vs_effective(
    *,
    rank: int,
    args: argparse.Namespace,
    scene_filter: Any,
    model_sampling: Any,
    proposal_sampling: Any,
) -> None:
    if rank != 0:
        return
    print(
        "[eval][align] requested:",
        f"cli_history={args.num_history_frames}",
        f"cli_scene_future={args.num_future_frames}",
        f"cli_model_future={args.model_future_frames}",
        f"cli_pdm={args.pdm_num_poses}x{args.pdm_interval_length}",
        f"scene_filter_yaml={'none' if not args.scene_filter_yaml else args.scene_filter_yaml}",
        f"scene_filter_yaml_filter_only={args.scene_filter_yaml_filter_only}",
    )
    print(
        "[eval][align] effective:",
        f"history={scene_filter.num_history_frames}",
        f"scene_future={scene_filter.num_future_frames}",
        f"model_future={getattr(model_sampling, 'num_poses', 'na')}",
        f"pdm={getattr(proposal_sampling, 'num_poses', 'na')}x{getattr(proposal_sampling, 'interval_length', 'na')}",
    )
    if (
        args.scene_filter_yaml
        and not args.scene_filter_yaml_filter_only
        and (
        int(scene_filter.num_history_frames) != int(args.num_history_frames)
        or int(scene_filter.num_future_frames) != int(args.num_future_frames)
        )
    ):
        print(
            "[eval][align][warn] scene_filter effective values differ from CLI values "
            "(likely overridden by scene_filter_yaml)."
        )


def _normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint keys to pipeline-style names.

    Supports both:
    - old mixed format: `blocks.*` (DiT bare keys) + `pipe.trajectory_*.*`
    - new format: `dit.*` + `trajectory_*.*` (remove_prefix=`pipe.`)
    """
    normalized: Dict[str, torch.Tensor] = {}
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
    for key, value in state_dict.items():
        norm_key = key[5:] if key.startswith("pipe.") else key
        if not norm_key.startswith(known_pipe_prefixes):
            # Old full ckpt format stores DiT params as bare module keys (e.g., "blocks.0...").
            norm_key = f"dit.{norm_key}"
        normalized[norm_key] = value
    return normalized


def _print_ckpt_load_report(
    missing: List[str],
    unexpected: List[str],
    *,
    rank: int,
    print_keys: bool,
    max_keys: int,
) -> None:
    if rank != 0:
        return

    def _prefix(key: str) -> str:
        if "." not in key:
            return key
        return key.split(".", 1)[0]

    if missing:
        missing_counter = Counter(_prefix(k) for k in missing)
        summary = ", ".join(f"{k}:{v}" for k, v in missing_counter.most_common())
        print(f"[eval] ckpt missing summary by module: {summary}")
    if unexpected:
        unexpected_counter = Counter(_prefix(k) for k in unexpected)
        summary = ", ".join(f"{k}:{v}" for k, v in unexpected_counter.most_common())
        print(f"[eval] ckpt unexpected summary by module: {summary}")

    if not print_keys:
        return

    def _print_key_list(title: str, keys: List[str]) -> None:
        if not keys:
            return
        if max_keys <= 0:
            to_print = keys
            suffix = ""
        else:
            to_print = keys[:max_keys]
            suffix = f" (showing {len(to_print)}/{len(keys)})"
        print(f"[eval] {title}{suffix}:")
        for k in to_print:
            print(f"[eval][{title}] {k}")

    _print_key_list("missing", missing)
    _print_key_list("unexpected", unexpected)


def _no_progress_bar(iterable, *args, **kwargs):
    return iterable


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _sync_device_for_timing(device: torch.device) -> None:
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _cuda_memory_current_mb(device: torch.device) -> Dict[str, float]:
    if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
        return {}
    mib = 1024.0 * 1024.0
    return {
        "allocated_mb": float(torch.cuda.memory_allocated(device) / mib),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / mib),
    }


def _cuda_memory_peak_mb(device: torch.device) -> Dict[str, float]:
    if not torch.cuda.is_available() or getattr(device, "type", None) != "cuda":
        return {}
    mib = 1024.0 * 1024.0
    return {
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / mib),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / mib),
    }


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _make_empty_video_decode(device: torch.device) -> Callable[..., torch.Tensor]:
    def _decode_without_video(latents: Any = None, *args: Any, **kwargs: Any) -> torch.Tensor:
        out_device = latents.device if torch.is_tensor(latents) else device
        return torch.empty((1, 3, 0, 1, 1), device=out_device, dtype=torch.float32)

    return _decode_without_video


def _set_vae_progress(show_vae_progress: bool) -> None:
    try:
        import diffsynth.models.wan_video_vae as wan_video_vae_mod
    except Exception:
        return
    if not hasattr(wan_video_vae_mod, "_original_tqdm_for_eval"):
        wan_video_vae_mod._original_tqdm_for_eval = wan_video_vae_mod.tqdm
    wan_video_vae_mod.tqdm = wan_video_vae_mod._original_tqdm_for_eval if show_vae_progress else _no_progress_bar


def _is_path_like(x: Any) -> bool:
    return isinstance(x, (str, Path))


def _camera_image_or_path(camera_obj: Any) -> Any:
    image = getattr(camera_obj, "image", None)
    if image is not None:
        return image
    camera_path = getattr(camera_obj, "camera_path", None)
    if camera_path is not None:
        return camera_path
    return None


def _to_rgb_uint8(image_like: Any) -> np.ndarray:
    if _is_path_like(image_like):
        return np.asarray(Image.open(image_like).convert("RGB"), dtype=np.uint8)
    if isinstance(image_like, Image.Image):
        return np.asarray(image_like.convert("RGB"), dtype=np.uint8)
    if torch.is_tensor(image_like):
        arr = image_like.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in {1, 3}:
            arr = arr.permute(1, 2, 0)
        arr = arr.float().numpy()
        if arr.ndim != 3:
            raise ValueError(f"Unsupported tensor image shape: {tuple(image_like.shape)}")
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.dtype != np.uint8:
            if arr.min() >= -0.1 and arr.max() <= 1.1:
                arr = np.clip(arr * 255.0, 0.0, 255.0)
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        return arr
    arr = np.asarray(image_like)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.min() >= -0.1 and arr.max() <= 1.1:
            arr = np.clip(arr * 255.0, 0.0, 255.0)
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return arr


def _to_chw_tensor(image_like: Any, normalize: str) -> torch.Tensor:
    arr = _to_rgb_uint8(image_like).astype(np.float32) / 255.0
    if normalize == "[-1,1]":
        arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize_to_hw(img: torch.Tensor, height: int = 704, width: int = 1280) -> torch.Tensor:
    if img.dim() == 3:  # (C,H,W)
        _, h, w = img.shape
        if (h, w) == (height, width):
            return img
        return F.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)
    if img.dim() == 4:  # (T,C,H,W)
        _, _, h, w = img.shape
        if (h, w) == (height, width):
            return img
        return F.interpolate(
            img, size=(height, width), mode="bilinear", align_corners=False
        )
    raise ValueError(f"Unexpected img shape {tuple(img.shape)}")


def _declared_color_space(camera_obj: Any) -> Optional[str]:
    if camera_obj is None:
        return None
    keys = ("color_space", "image_color_space", "channel_order", "pixel_format")
    for key in keys:
        raw = camera_obj.get(key) if isinstance(camera_obj, dict) else getattr(camera_obj, key, None)
        if raw is None and not isinstance(camera_obj, dict):
            meta = getattr(camera_obj, "metadata", None)
            if isinstance(meta, dict):
                raw = meta.get(key)
        if raw is None:
            continue
        token = str(raw).strip().upper()
        if len(token) == 0:
            continue
        if "BGR" in token:
            return "BGR"
        if "RGB" in token:
            return "RGB"
    return None


def _maybe_bgr_to_rgb(arr: np.ndarray, camera_obj: Any) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    # Only convert when the source explicitly declares BGR.
    # Do not use pixel-statistics heuristics (e.g., mean(R) > mean(B)),
    # because red-dominant scenes can be falsely swapped.
    if _declared_color_space(camera_obj) == "BGR":
        return arr[..., [2, 1, 0]]
    return arr


def _grab_front_camera(cameras: Any) -> tuple[str, Any]:
    cam_f0 = getattr(cameras, "cam_f0", None)
    if cam_f0 is None:
        raise ValueError("Missing camera view: cam_f0")
    item = _camera_image_or_path(cam_f0)
    if item is None:
        raise ValueError("Missing camera content in cam_f0")
    if _is_path_like(item):
        return "path", str(item)
    return "rgb", _maybe_bgr_to_rgb(_to_rgb_uint8(item), cam_f0)


def _front_camera_intrinsics(cameras: Any) -> Optional[np.ndarray]:
    cam_f0 = getattr(cameras, "cam_f0", None)
    if cam_f0 is None:
        return None
    intrinsics = getattr(cam_f0, "intrinsics", None)
    if intrinsics is None:
        return None
    try:
        arr = np.asarray(intrinsics, dtype=np.float32)
    except Exception:
        return None
    if arr.shape != (3, 3) or not np.isfinite(arr).all():
        return None
    return arr


def _resize_rgb_uint8(arr: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr
    return np.asarray(Image.fromarray(arr).resize((width, height), Image.BILINEAR), dtype=np.uint8)


def _build_surround_mosaic(cameras: Any, surround_keys: tuple[str, ...]) -> Any:
    camera_items = []
    for key in surround_keys:
        cam = getattr(cameras, key, None)
        if cam is None:
            raise ValueError(f"Missing camera view: {key}")
        item = _camera_image_or_path(cam)
        if item is None:
            raise ValueError(f"Missing camera content in view: {key}")
        if _is_path_like(item):
            camera_items.append(str(item))
        else:
            camera_items.append(_maybe_bgr_to_rgb(_to_rgb_uint8(item), cam))
    if all(_is_path_like(item) for item in camera_items):
        return [str(item) for item in camera_items]
    rgb_items = [_to_rgb_uint8(item) for item in camera_items]
    h0, w0 = rgb_items[0].shape[:2]
    rgb_items = [_resize_rgb_uint8(arr, height=h0, width=w0) for arr in rgb_items]
    row1 = np.concatenate(rgb_items[0:3], axis=1)
    row2 = np.concatenate(rgb_items[3:6], axis=1)
    return np.concatenate([row1, row2], axis=0)


def _extract_future_cameras(scene: Any, num_trajectory_frames: int) -> List[Any]:
    if hasattr(scene, "get_future_frames"):
        try:
            cams = scene.get_future_frames(num_trajectory_frames=num_trajectory_frames)
            return list(cams) if cams is not None else []
        except TypeError:
            cams = scene.get_future_frames(num_trajectory_frames)
            return list(cams) if cams is not None else []
    frames = getattr(scene, "frames", None)
    scene_metadata = getattr(scene, "scene_metadata", None)
    if frames is None or scene_metadata is None:
        return []
    start_idx = int(getattr(scene_metadata, "num_history_frames", 0))
    end_idx = min(len(frames), start_idx + int(num_trajectory_frames))
    future_cameras: List[Any] = []
    for frame_idx in range(start_idx, end_idx):
        frame = frames[frame_idx]
        cams = getattr(frame, "cameras", None)
        if cams is not None:
            future_cameras.append(cams)
    return future_cameras


class _CompatUNIVERSEFeatureBuilder:
    def __init__(
        self,
        prompt_frames: int = 4,
        normalize: str = "[-1,1]",
        ltx_min_prompt_frames: int = 8,
        view_mode: str = "front",
        surround_keys: tuple[str, ...] = (
            "cam_l0",
            "cam_f0",
            "cam_r0",
            "cam_l2",
            "cam_b0",
            "cam_r2",
        ),
    ) -> None:
        self.prompt_frames = int(prompt_frames)
        self.normalize = str(normalize)
        self.ltx_min_prompt_frames = int(ltx_min_prompt_frames)
        self.view_mode = str(view_mode)
        self.surround_keys = tuple(surround_keys)

    def compute_features(self, agent_input: Any) -> Dict[str, Any]:
        hist_candidates = [len(agent_input.ego_statuses), len(agent_input.cameras)]
        ego2global_t = getattr(agent_input, "ego2global_T", None)
        if ego2global_t is not None:
            try:
                hist_candidates.append(int(ego2global_t.shape[0]))
            except Exception:
                pass
        num_hist = min(hist_candidates)
        if num_hist <= 0:
            raise ValueError("No history frames available in AgentInput.")
        t_raw = min(max(1, self.prompt_frames), num_hist)
        start_idx = num_hist - t_raw
        idx_range = range(start_idx, num_hist)

        paths_front: List[str] = []
        paths_surround: List[List[str]] = []
        intrinsics_front: List[Any] = []
        imgs_tensor: List[torch.Tensor] = []
        saw_path = False

        if self.view_mode == "front":
            for i in idx_range:
                intrinsics_front.append(_front_camera_intrinsics(agent_input.cameras[i]))
                mode, item = _grab_front_camera(agent_input.cameras[i])
                if mode == "path":
                    saw_path = True
                    paths_front.append(item)
                else:
                    imgs_tensor.append(_to_chw_tensor(item, normalize=self.normalize))
        else:
            for i in idx_range:
                item = _build_surround_mosaic(agent_input.cameras[i], self.surround_keys)
                if isinstance(item, list):
                    saw_path = True
                    paths_surround.append([str(p) for p in item])
                else:
                    imgs_tensor.append(_to_chw_tensor(item, normalize=self.normalize))

        output: Dict[str, Any] = {}
        if saw_path:
            output["image_paths"] = paths_front if self.view_mode == "front" else paths_surround
        else:
            images = torch.stack(imgs_tensor, dim=0)  # (T,C,H,W)
            images = _resize_to_hw(images, 768, 1344)
            output["images"] = images
        if self.view_mode == "front":
            output["camera_intrinsics"] = intrinsics_front

        def _ego_pose_triplet(ego_status: Any) -> List[float]:
            ego_pose = np.asarray(getattr(ego_status, "ego_pose", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
            return [
                float(ego_pose[0]) if ego_pose.size > 0 else 0.0,
                float(ego_pose[1]) if ego_pose.size > 1 else 0.0,
                float(ego_pose[2]) if ego_pose.size > 2 else 0.0,
            ]

        ego_statuses = agent_input.ego_statuses
        output["history_trajectory"] = torch.tensor(
            [_ego_pose_triplet(e) for e in ego_statuses[:4]], dtype=torch.float32
        )

        last_status = ego_statuses[-1]
        output["vel"] = torch.as_tensor(getattr(last_status, "ego_velocity", [0.0, 0.0]), dtype=torch.float32)
        output["acc"] = torch.as_tensor(getattr(last_status, "ego_acceleration", [0.0, 0.0]), dtype=torch.float32)
        output["driving_command"] = torch.as_tensor(
            getattr(last_status, "driving_command", [0.0, 1.0, 0.0]), dtype=torch.float32
        )
        return output


class _CompatTrajectoryTargetBuilder:
    def __init__(
        self,
        trajectory_sampling: Any,
        normalize: str = "[-1,1]",
        view_mode: str = "front",
        surround_keys: tuple[str, ...] = (
            "cam_l0",
            "cam_f0",
            "cam_r0",
            "cam_l2",
            "cam_b0",
            "cam_r2",
        ),
    ) -> None:
        self._trajectory_sampling = trajectory_sampling
        self.normalize = str(normalize)
        self.view_mode = str(view_mode)
        self.surround_keys = tuple(surround_keys)

    def compute_targets(self, scene: Any) -> Dict[str, Any]:
        future = scene.get_future_trajectory(num_trajectory_frames=int(self._trajectory_sampling.num_poses))
        output: Dict[str, Any] = {
            "trajectory": torch.as_tensor(np.asarray(future.poses), dtype=torch.float32)
        }

        cameras = _extract_future_cameras(scene, num_trajectory_frames=int(self._trajectory_sampling.num_poses))
        if len(cameras) == 0:
            return output

        got_path = False
        frames_tensor: List[torch.Tensor] = []
        future_paths_front: List[str] = []
        future_paths_surround: List[List[str]] = []
        future_intrinsics_front: List[Any] = []

        if self.view_mode == "front":
            for cams in cameras:
                future_intrinsics_front.append(_front_camera_intrinsics(cams))
                mode, item = _grab_front_camera(cams)
                if mode == "path":
                    got_path = True
                    future_paths_front.append(item)
                else:
                    frames_tensor.append(_to_chw_tensor(item, normalize=self.normalize))
        else:
            for cams in cameras:
                item = _build_surround_mosaic(cams, self.surround_keys)
                if isinstance(item, list):
                    got_path = True
                    future_paths_surround.append([str(p) for p in item])
                else:
                    frames_tensor.append(_to_chw_tensor(item, normalize=self.normalize))

        if got_path:
            if self.view_mode == "front":
                output["future_image_paths"] = future_paths_front
                output["future_camera_intrinsics"] = future_intrinsics_front
            else:
                output["future_image_paths"] = future_paths_surround
            return output

        if len(frames_tensor) == 0:
            return output
        output["future_frames"] = torch.stack(frames_tensor, dim=0)
        if self.view_mode == "front":
            output["future_camera_intrinsics"] = future_intrinsics_front
        return output


def _resolve_UNIVERSE_feature_builders():
    try:
        from importlib import import_module

        feature_module = import_module("navsim.agents.videodrive.videodrive_features")
        UNIVERSE_feature_builder_cls = getattr(feature_module, "Video" "DriveFeatureBuilder")
        trajectory_target_builder_cls = feature_module.TrajectoryTargetBuilder
        return UNIVERSE_feature_builder_cls, trajectory_target_builder_cls, "navsim.agents.videodrive.videodrive_features"
    except Exception as exc:
        source = f"local_compat ({exc.__class__.__name__})"
        return _CompatUNIVERSEFeatureBuilder, _CompatTrajectoryTargetBuilder, source


def _build_scene_loader(SceneLoader: Any, SensorConfig: Any, args: argparse.Namespace, scene_filter: Any):
    sensor_config = SensorConfig.build_all_sensors(include=True)
    init_sig = inspect.signature(SceneLoader.__init__)
    if "sensor_blobs_path" in init_sig.parameters:
        loader = SceneLoader(
            data_path=Path(args.navsim_log_path),
            sensor_blobs_path=Path(args.sensor_blobs_path),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
            load_image_path=True,
        )
    else:
        loader = SceneLoader(
            data_path=Path(args.navsim_log_path),
            original_sensor_path=Path(args.sensor_blobs_path),
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )
    loader._UNIVERSE_sensor_blobs_path = Path(args.sensor_blobs_path)
    loader._UNIVERSE_load_image_path = bool("load_image_path" in init_sig.parameters)
    return loader


def _build_scene_without_print(scene_loader, scene_cls, token: str):
    """Build Scene directly to avoid SceneLoader.get_scene_from_token token printing."""
    assert token in scene_loader.tokens
    sensor_root = (
        getattr(scene_loader, "_sensor_blobs_path", None)
        or getattr(scene_loader, "_original_sensor_path", None)
        or getattr(scene_loader, "_UNIVERSE_sensor_blobs_path", None)
    )
    if sensor_root is None:
        raise AttributeError("SceneLoader does not expose a sensor root path.")

    kwargs: Dict[str, Any] = {
        "scene_dict_list": scene_loader.scene_frames_dicts[token],
        "sensor_blobs_path": sensor_root,
        "num_history_frames": scene_loader._scene_filter.num_history_frames,
        "num_future_frames": scene_loader._scene_filter.num_future_frames,
        "sensor_config": scene_loader._sensor_config,
    }
    from_scene_sig = inspect.signature(scene_cls.from_scene_dict_list)
    if "load_image_path" in from_scene_sig.parameters:
        kwargs["load_image_path"] = bool(
            getattr(scene_loader, "load_image_path", getattr(scene_loader, "_UNIVERSE_load_image_path", False))
        )
    return scene_cls.from_scene_dict_list(**kwargs)


def run_eval(args: argparse.Namespace, external_pipe: Optional[WanVideoPipeline] = None) -> None:
    _ensure_navsim_importable(Path(args.repo_root))

    # Lazy imports: navsim/nuplan are only importable after _ensure_navsim_importable adds the path
    from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig, Trajectory, NAVSIM_INTERVAL_LENGTH
    from navsim.common.dataloader import SceneLoader, MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    dist_info = _init_distributed()
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]
    UNIVERSE_feature_builder_cls, TrajectoryTargetBuilder, builder_source = _resolve_UNIVERSE_feature_builders()
    if rank == 0:
        print(f"[eval] feature_builder_source={builder_source}")
    focal_config = FocalUnificationConfig.build(
        enabled=bool(getattr(args, "use_focal_unification", False)) and not bool(args.surround_view),
        target_focal_px=getattr(args, "focal_target_px", ""),
        output_width=int(args.width),
        output_height=int(args.height),
    )
    if rank == 0:
        print(
            "[eval][focal]",
            f"enabled={focal_config.enabled}",
            f"target_focal_px={focal_config.target_focal_px}",
            f"output={args.width}x{args.height}",
        )
    compat_items: List[str] = []
    if _ensure_navsim_scene_frame_type_compat():
        compat_items.append("navsim.common.enums.SceneFrameType")
    if _ensure_navsim_map_parameters_compat():
        compat_items.append("navsim.planning.metric_caching.metric_cache.MapParameters")
    if compat_items and rank == 0:
        print("[eval][compat] injected for v2 metric-cache pickle compatibility:", ", ".join(compat_items))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if external_pipe is None:
        # Keep model params on the same device as pipeline compute device to avoid
        # cuda index_select against cpu weights.
        model_offload_device = str(device)
        # eval_navsim_v1.sh passes LOCAL_MODEL_PATH, so all Wan base components
        # are resolved from local disk and no runtime download is attempted.
        model_configs = [
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device=model_offload_device,
                local_model_path=args.local_model_path,
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device=model_offload_device,
                local_model_path=args.local_model_path,
                skip_download=True,
            ),
            ModelConfig(
                model_id="Wan-AI/Wan2.2-TI2V-5B",
                origin_file_pattern="Wan2.2_VAE.pth",
                offload_device=model_offload_device,
                local_model_path=args.local_model_path,
                skip_download=True,
            ),
        ]

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=str(device),
            model_configs=model_configs,
            use_trajectory=True,
        )
    else:
        pipe = external_pipe
        if rank == 0:
            print("[eval] using external pipeline")
    pipe.eval()
    _set_vae_progress(args.show_vae_progress)
    pipe.target_fps = args.target_fps
    pipe.num_history_frames = int(args.num_history_frames)
    # Keep trajectory settings aligned with the released UNIVERSE checkpoint:
    # local-frame x/y/heading points are denoised directly, not as deltas.
    pipe.trajectory_norm_mode = "UNIVERSE_odo"
    pipe.trajectory_use_relative = False
    pipe.infer_replace_history_latents_before_decode = bool(args.infer_replace_history_latents_before_decode)
    pipe.trajectory_condition_mode = args.trajectory_condition_mode
    pipe.use_mixed_latent_attention_mask = bool(args.use_mixed_latent_attention_mask)
    infer_output_mode = getattr(args, "infer_output_mode", None)
    infer_trajectory_only = bool(getattr(args, "infer_trajectory_only", False))
    if infer_output_mode == "trajectory":
        infer_trajectory_only = True
    elif infer_output_mode in {"video", "both"}:
        infer_trajectory_only = False
    pipe.infer_trajectory_only = infer_trajectory_only
    skip_video_decode_for_timing = bool(
        infer_trajectory_only
        and not args.save_viz
        and infer_output_mode is None
    )
    if rank == 0:
        print(
            "[eval] infer timing mode:",
            f"infer_trajectory_only={infer_trajectory_only}",
            f"output_mode={infer_output_mode or 'legacy'}",
            f"mixed_attention_mask={bool(args.use_mixed_latent_attention_mask)}",
            f"skip_video_decode={skip_video_decode_for_timing}",
            f"save_viz={bool(args.save_viz)}",
        )

    if external_pipe is None:
        if not os.path.exists(args.full_ckpt):
            raise FileNotFoundError(f"full_ckpt not found: {args.full_ckpt}")
        if rank == 0:
            print(f"[eval] loading full checkpoint: {args.full_ckpt}")
        state_dict_raw = load_state_dict(args.full_ckpt)
        state_dict = _normalize_checkpoint_keys(state_dict_raw)
        # Newer UNIVERSE checkpoints include DiT plus trajectory heads; older
        # DiT-only weights still run but cannot produce learned trajectories.
        has_traj_keys = any(
            k.startswith("trajectory_encoder.") or k.startswith("trajectory_head.")
            for k in state_dict.keys()
        )
        if has_traj_keys:
            missing, unexpected = pipe.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(
                    f"[eval] full_ckpt loaded into pipeline "
                    f"(dit+trajectory), missing={len(missing)} unexpected={len(unexpected)}"
                )
            _print_ckpt_load_report(
                list(missing),
                list(unexpected),
                rank=rank,
                print_keys=args.print_ckpt_missing,
                max_keys=args.max_print_ckpt_keys,
            )
        else:
            missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(
                    f"[eval][warn] full_ckpt has no trajectory_* keys; "
                    f"loading DiT only. missing={len(missing)} unexpected={len(unexpected)}"
                )
            _print_ckpt_load_report(
                list(missing),
                list(unexpected),
                rank=rank,
                print_keys=args.print_ckpt_missing,
                max_keys=args.max_print_ckpt_keys,
            )
    if rank == 0:
        def _dev_of(m):
            if m is None:
                return "none"
            try:
                return str(next(m.parameters()).device)
            except StopIteration:
                return "no-params"
        print(
            "[eval] module devices:",
            f"text_encoder={_dev_of(pipe.text_encoder)}",
            f"dit={_dev_of(pipe.dit)}",
            f"vae={_dev_of(pipe.vae)}",
            f"trajectory_encoder={_dev_of(pipe.trajectory_encoder)}",
            f"trajectory_head={_dev_of(pipe.trajectory_head)}",
        )
        if torch.cuda.is_available() and device.type == "cuda":
            mem_now = _cuda_memory_current_mb(device)
            total_mb = float(torch.cuda.get_device_properties(device).total_memory / (1024.0 * 1024.0))
            print(
                "[eval] cuda memory after model load:",
                f"allocated={mem_now.get('allocated_mb', float('nan')):.1f}MB",
                f"reserved={mem_now.get('reserved_mb', float('nan')):.1f}MB",
                f"total={total_mb:.1f}MB",
            )
        print(f"[eval] vae_progress={'on' if args.show_vae_progress else 'off'}")

    log_names = [x.strip() for x in args.log_names.split(",") if x.strip()] if args.log_names else None
    scene_filter_overrides = _load_scene_filter_yaml(
        args.scene_filter_yaml,
        filter_only=args.scene_filter_yaml_filter_only,
    )
    scene_filter_kwargs: Dict[str, Any] = {
        "num_history_frames": args.num_history_frames,
        "num_future_frames": args.num_future_frames,
        "frame_interval": args.frame_interval,
        "has_route": True,
        "max_scenes": args.max_scenes,
        "log_names": log_names,
    }
    for key, value in scene_filter_overrides.items():
        scene_filter_kwargs[key] = value
    if log_names is not None:
        # Explicit CLI list has priority over yaml log_names.
        scene_filter_kwargs["log_names"] = log_names
    scene_filter = SceneFilter(**scene_filter_kwargs)
    min_scene_future_frames = _min_scene_future_frames(
        model_future_frames=int(args.model_future_frames),
        model_fps=float(args.target_fps),
        scene_interval_s=float(NAVSIM_INTERVAL_LENGTH),
        extra_seconds=float(args.scene_future_extra_seconds),
    )
    if int(scene_filter.num_future_frames) < int(min_scene_future_frames):
        if rank == 0:
            print(
                "[eval][align][warn] scene future horizon too short for TTC look-ahead;",
                f"bump {scene_filter.num_future_frames} -> {min_scene_future_frames}",
                f"(model_future={args.model_future_frames}, target_fps={args.target_fps}, +{args.scene_future_extra_seconds}s)",
            )
        scene_filter_kwargs["num_future_frames"] = int(min_scene_future_frames)
        scene_filter = SceneFilter(**scene_filter_kwargs)
    scene_loader = _build_scene_loader(
        SceneLoader=SceneLoader,
        SensorConfig=SensorConfig,
        args=args,
        scene_filter=scene_filter,
    )

    metric_cache_loader = MetricCacheLoader(Path(args.metric_cache_path))

    # Fix relative paths in metric cache: the metadata CSV may contain relative
    # paths (e.g. "exp/metric_cache/<scene>/…") that don't resolve from CWD.
    # Re-root them so they become absolute under the supplied cache directory.
    cache_path_abs = Path(args.metric_cache_path).resolve()
    fixed_paths: Dict[str, str] = {}
    for tok, p in metric_cache_loader.metric_cache_paths.items():
        p_path = Path(p)
        if p_path.exists():
            fixed_paths[tok] = p
            continue
        # Try to locate the file relative to cache_path by stripping the
        # leading directories up to and including "metric_cache/"
        parts = p_path.parts
        for i, part in enumerate(parts):
            if part == "metric_cache" and i + 1 < len(parts):
                candidate = cache_path_abs.joinpath(*parts[i + 1:])
                if candidate.exists():
                    fixed_paths[tok] = str(candidate)
                    break
        else:
            # Last resort: keep the original (will fail later with a clear message)
            fixed_paths[tok] = p
    metric_cache_loader.metric_cache_paths = fixed_paths
    if rank == 0:
        print(f"[eval] metric_cache_path={args.metric_cache_path} cache_tokens={len(metric_cache_loader.tokens)}")

    tokens = sorted(list(set(scene_loader.tokens) & set(metric_cache_loader.tokens)))
    shard_cfg = _resolve_eval_shards(
        rank=rank,
        world_size=world_size,
        num_eval_shards=args.num_eval_shards,
    )
    active_shards = int(shard_cfg["active_shards"])
    local_shard_ids: List[int] = list(shard_cfg["local_shard_ids"])
    if len(local_shard_ids) == 0:
        local_tokens: List[str] = []
    else:
        local_shard_id_set = set(local_shard_ids)
        local_tokens = [
            tok
            for idx, tok in enumerate(tokens)
            if (idx % active_shards) in local_shard_id_set
        ]
    if args.token_offset > 0:
        local_tokens = local_tokens[args.token_offset :]
    if args.max_eval_tokens is not None:
        local_tokens = local_tokens[: max(args.max_eval_tokens, 0)]
    if args.resume_csv:
        resume_path = Path(args.resume_csv)
        if resume_path.exists():
            import pandas as pd
            done_df = pd.read_csv(resume_path)
            done_tokens = set(done_df.get("token", []).tolist())
            done_tokens.discard("average")
            before = len(local_tokens)
            local_tokens = [t for t in local_tokens if t not in done_tokens]
            if rank == 0:
                print(f"[eval] resume skip: {before - len(local_tokens)} tokens already in {resume_path}")
        elif rank == 0:
            print(f"[eval] resume_csv not found, ignore: {resume_path}")
    if rank == 0:
        idle_ranks = max(0, int(world_size) - int(active_shards))
        print(
            "[eval] shard config:",
            f"world_size={world_size}",
            f"active_shards={active_shards}",
            f"idle_ranks={idle_ranks}",
            f"virtual_shards_per_rank~={(active_shards + world_size - 1) // max(1, world_size)}",
            f"requested={args.num_eval_shards if args.num_eval_shards is not None else 'auto'}",
        )
        print(f"[eval] total tokens={len(tokens)} local_tokens(rank0)={len(local_tokens)}")
    local_viz_quota = int(args.viz_max_tokens)
    if int(args.viz_total_tokens) > 0:
        active_ranks = min(int(world_size), int(active_shards))
        if active_ranks > 0 and rank < active_ranks:
            base = int(args.viz_total_tokens) // active_ranks
            rem = int(args.viz_total_tokens) % active_ranks
            local_viz_quota = base + (1 if rank < rem else 0)
        else:
            local_viz_quota = 0

    if args.target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {args.target_fps}")
    model_sampling = TrajectorySampling(
        num_poses=args.model_future_frames,
        interval_length=1.0 / float(args.target_fps),
    )
    proposal_sampling = TrajectorySampling(
        num_poses=args.pdm_num_poses,
        interval_length=args.pdm_interval_length,
    )
    legacy_simulate_traffic_agents = bool(args.legacy_simulate_traffic_agents)
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(proposal_sampling=proposal_sampling)
    pdm_score_accepts_traffic_policy = "traffic_agents_policy" in inspect.signature(pdm_score).parameters
    traffic_agents_policy = None
    if pdm_score_accepts_traffic_policy or legacy_simulate_traffic_agents:
        traffic_agents_policy = _build_traffic_agents_policy(args.traffic_agents_policy, proposal_sampling)
    if rank == 0:
        policy_label = args.traffic_agents_policy if traffic_agents_policy is not None else "unused"
        print(f"[eval] traffic_agents_policy={policy_label}")
        print(f"[eval] legacy_simulate_traffic_agents={legacy_simulate_traffic_agents}")
    nuscenes_metric_horizons_s: List[float] = []
    nuscenes_final_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None
    if args.enable_nuscenes_metrics:
        nuscenes_metric_horizons_s = _parse_metric_horizons_s(args.nuscenes_metric_horizons_s)
        nuscenes_final_distance_fn = _load_nuscenes_final_distance()
        if rank == 0:
            horizon_text = ", ".join(_format_horizon_label(h) for h in nuscenes_metric_horizons_s)
            print(f"[eval][nuscenes] enabled horizons={horizon_text}")
            print("[eval][nuscenes] metrics: L2 (m), Collision (%)")
    if args.print_alignment_params:
        _print_requested_vs_effective(
            rank=rank,
            args=args,
            scene_filter=scene_filter,
            model_sampling=model_sampling,
            proposal_sampling=proposal_sampling,
        )
        _print_alignment_summary(
            rank=rank,
            scene_filter=scene_filter,
            model_sampling=model_sampling,
            proposal_sampling=proposal_sampling,
            scorer=scorer,
            navsim_interval_length=NAVSIM_INTERVAL_LENGTH,
            target_fps=args.target_fps,
        )

    feature_builder = UNIVERSE_feature_builder_cls(
        prompt_frames=scene_filter.num_history_frames,
        normalize="[-1,1]",
        view_mode="surround6" if args.surround_view else "front",
    )
    need_future_targets = bool(args.save_viz) or bool(args.enable_nuscenes_metrics)
    target_builder = None
    viz_dir = None
    if need_future_targets:
        target_builder = TrajectoryTargetBuilder(
            trajectory_sampling=TrajectorySampling(
                num_poses=scene_filter.num_future_frames,
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
            normalize="[-1,1]",
            view_mode="surround6" if args.surround_view else "front",
        )
    if args.save_viz:
        viz_dir = Path(args.viz_dir) if args.viz_dir is not None else (Path(args.output_dir) / "viz")
        viz_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print(
                f"[eval] save_viz enabled: dir={viz_dir} local_quota(rank0)={local_viz_quota} "
                f"total_tokens={args.viz_total_tokens} fallback_per_rank={args.viz_max_tokens} "
                f"fps={args.target_fps} plot_height={DEFAULT_VIZ_PLOT_HEIGHT}"
            )

    results: List[Dict[str, Any]] = []
    viz_saved = 0
    eval_start = time.perf_counter()
    local_success = 0
    local_failed = 0
    local_infer_time_sum_s = 0.0
    local_infer_time_count = 0
    local_last_infer_time_s = float("nan")
    printed_runtime_alignment = False
    printed_prompt_steps = 0
    if args.show_eval_progress and rank == 0:
        token_iter = tqdm(local_tokens, desc="Eval", total=len(local_tokens), dynamic_ncols=True)
    else:
        token_iter = local_tokens
    for step_idx, token in enumerate(token_iter, start=1):
        if args.print_tokens:
            print(token, flush=True)
        score_row: Dict[str, Any] = {"token": token, "valid": True, "rank": rank}
        try:
            metric_cache = metric_cache_loader.get_from_token(token)

            scene = None
            if target_builder is not None:
                scene = _build_scene_without_print(scene_loader, Scene, token)
                agent_input = scene.get_agent_input()
            else:
                agent_input = scene_loader.get_agent_input_from_token(token)

            features = feature_builder.compute_features(agent_input)
            # Feature builders normalize NavSIM into the model contract:
            # history frames, route command, ego speed, and optional history XYH.
            history_frames = features.get("image_paths") or features.get("images")
            if history_frames is None:
                raise ValueError("Missing history frames from feature builder.")
            history_video = _resolve_video_pil(
                history_frames,
                height=args.height,
                width=args.width,
                surround_view=args.surround_view,
                normalize_mode="[-1,1]",
                camera_intrinsics=features.get("camera_intrinsics"),
                focal_config=focal_config,
            )

            history_traj = features.get("history_trajectory")
            driving_command = features.get("driving_command")
            vel = features.get("vel")
            speed = float(torch.linalg.norm(vel).item()) if vel is not None else 0.0

            prompt = _build_prompt_fixed(driving_command, speed)
            if rank == 0 and printed_prompt_steps < max(0, int(args.debug_prompt_steps)):
                idx = printed_prompt_steps + 1
                print(f"[eval][prompt][{idx}/{args.debug_prompt_steps}] positive: {prompt}")
                print(f"[eval][prompt][{idx}/{args.debug_prompt_steps}] negative: {args.negative_prompt}")
                printed_prompt_steps += 1
            total_frames = scene_filter.num_history_frames + args.model_future_frames
            history_cond = None
            if args.trajectory_condition_mode in {"auto", "history"}:
                history_cond = history_traj
            # The pipeline conditions on clean history frames and denoises only
            # the requested future trajectory length.
            _sync_device_for_timing(device)
            mem_before = _cuda_memory_current_mb(device)
            _reset_cuda_peak_memory(device)
            infer_start_s = time.perf_counter()
            original_vae_decode = None
            with torch.no_grad():
                if skip_video_decode_for_timing and pipe.vae is not None:
                    original_vae_decode = pipe.vae.decode
                    pipe.vae.decode = _make_empty_video_decode(device)
                try:
                    out = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        input_video=None,
                        longcat_video=history_video,
                        height=args.height,
                        width=args.width,
                        num_frames=total_frames,
                        cfg_scale=args.cfg_scale,
                        num_inference_steps=args.num_inference_steps,
                        trajectory_len=args.model_future_frames,
                        ego_vel=vel,
                        history_positions=history_cond,
                        seed=args.seed,
                        rand_device=str(device),
                        tiled=True,
                        progress_bar_cmd=tqdm if args.show_denoise_progress else _no_progress_bar,
                        output_mode=infer_output_mode,
                    )
                finally:
                    if original_vae_decode is not None and pipe.vae is not None:
                        pipe.vae.decode = original_vae_decode
            _sync_device_for_timing(device)
            infer_time_s = time.perf_counter() - infer_start_s
            mem_after = _cuda_memory_current_mb(device)
            mem_peak = _cuda_memory_peak_mb(device)
            score_row["infer_time_s"] = float(infer_time_s)
            score_row["infer_time_ms"] = float(infer_time_s * 1000.0)
            score_row["infer_mode"] = (
                "trajectory_only" if infer_trajectory_only else "video_and_trajectory"
            )
            score_row["skip_video_decode"] = bool(skip_video_decode_for_timing)
            if mem_before and mem_after and mem_peak:
                before_alloc = float(mem_before["allocated_mb"])
                before_reserved = float(mem_before["reserved_mb"])
                peak_alloc = float(mem_peak["peak_allocated_mb"])
                peak_reserved = float(mem_peak["peak_reserved_mb"])
                score_row["gpu_mem_alloc_before_mb"] = before_alloc
                score_row["gpu_mem_alloc_after_mb"] = float(mem_after["allocated_mb"])
                score_row["gpu_mem_alloc_peak_mb"] = peak_alloc
                score_row["gpu_mem_alloc_peak_delta_mb"] = max(0.0, peak_alloc - before_alloc)
                score_row["gpu_mem_reserved_before_mb"] = before_reserved
                score_row["gpu_mem_reserved_after_mb"] = float(mem_after["reserved_mb"])
                score_row["gpu_mem_reserved_peak_mb"] = peak_reserved
                score_row["gpu_mem_reserved_peak_delta_mb"] = max(0.0, peak_reserved - before_reserved)
            if infer_output_mode == "trajectory":
                pred_video = []
                traj_pred = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
            elif infer_output_mode == "video":
                pred_video = out
                traj_pred = None
            elif isinstance(out, (list, tuple)) and len(out) >= 2:
                pred_video, traj_pred, _vel = out
            else:
                raise ValueError("Pipeline did not return trajectory output.")
            if traj_pred is None:
                raise ValueError("Pipeline video-only output has no trajectory for NavSIM PDM scoring.")

            traj_np = traj_pred[0].detach().float().cpu().numpy()
            traj_np = _safe_traj_np(traj_np, args.model_future_frames)
            targets: Optional[Dict[str, Any]] = None
            gt_traj: Optional[np.ndarray] = None
            gt_future_video: List[Image.Image] = []
            if target_builder is not None:
                if scene is None:
                    scene = _build_scene_without_print(scene_loader, Scene, token)
                targets = target_builder.compute_targets(scene)
                gt_traj = np.asarray(targets["trajectory"], dtype=np.float32)
                gt_traj = _safe_traj_np(gt_traj, scene_filter.num_future_frames)
                if bool(args.save_viz):
                    future_frames = targets.get("future_image_paths") or targets.get("future_frames")
                    if future_frames is None:
                        raise ValueError("Missing future frames from target builder.")
                    gt_future_video = _resolve_video_pil(
                        future_frames,
                        height=args.height,
                        width=args.width,
                        surround_view=args.surround_view,
                        normalize_mode="[-1,1]",
                        camera_intrinsics=targets.get("future_camera_intrinsics"),
                        focal_config=focal_config,
                    )
            if history_traj is None:
                hist_np = np.zeros((0, 3), dtype=np.float32)
            else:
                hist_np = np.asarray(history_traj, dtype=np.float32)
                if hist_np.ndim == 1:
                    hist_np = hist_np[None, :]

            trajectory = Trajectory(
                poses=traj_np.astype(np.float32),
                trajectory_sampling=model_sampling,
            )
            if args.print_alignment_params and (rank == 0) and (not printed_runtime_alignment):
                print(
                    "[eval][align][runtime-before-pdm]",
                    f"token={token}",
                    f"pred_len={traj_np.shape[0]}",
                    f"pred_sampling={trajectory.trajectory_sampling.num_poses}x{trajectory.trajectory_sampling.interval_length}",
                    f"pdm_sampling={proposal_sampling.num_poses}x{proposal_sampling.interval_length}",
                )

            pdm_result, pdm_dict, _ego_simulated_states = _score_navsim_pdm(
                pdm_score_fn=pdm_score,
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_agents_policy,
                legacy_simulate_traffic_agents=legacy_simulate_traffic_agents,
            )
            score_row.update(pdm_dict)
            collision_s = float("inf")
            try:
                collision_s = float(scorer.time_to_at_fault_collision(1))
            except Exception:
                pass

            if args.enable_nuscenes_metrics:
                if gt_traj is None:
                    raise ValueError("Missing GT trajectory for nuScenes metrics.")
                if nuscenes_final_distance_fn is None:
                    raise RuntimeError("nuScenes metric function was not initialized.")
                pred_collision_flags = _extract_pdm_collision_flags(
                    scorer=scorer,
                    proposal_idx=1,
                )
                gt_collision_flags = _extract_pdm_collision_flags(
                    scorer=scorer,
                    proposal_idx=0,
                )
                nuscenes_row = _compute_nuscenes_l2_and_collision_metrics(
                    pred_traj_xy=traj_np[..., :2],
                    pred_dt_s=1.0 / float(args.target_fps),
                    gt_traj_xy=gt_traj[..., :2],
                    gt_dt_s=float(NAVSIM_INTERVAL_LENGTH),
                    horizons_s=nuscenes_metric_horizons_s,
                    final_distance_fn=nuscenes_final_distance_fn,
                    pred_collision_flags=pred_collision_flags,
                    gt_collision_flags=gt_collision_flags,
                    collision_dt_s=float(proposal_sampling.interval_length),
                    collision_s=collision_s,
                )
                score_row.update(nuscenes_row)
            if args.print_alignment_params and (rank == 0) and (not printed_runtime_alignment):
                reconstructed_score = _reconstruct_pdm_score_from_result(pdm_result, scorer)
                ttc_s = float("inf")
                try:
                    ttc_s = float(scorer.time_to_ttc_infraction(1))
                except Exception:
                    pass
                print(
                    "[eval][align][runtime-after-pdm]",
                    f"token={token}",
                    f"no_collision={_fmt_float(float(pdm_result.no_at_fault_collisions))}",
                    f"drivable={_fmt_float(float(pdm_result.drivable_area_compliance))}",
                    f"traffic_light={_fmt_float(float(pdm_result.traffic_light_compliance))}",
                    f"driving_direction={_fmt_float(float(pdm_result.driving_direction_compliance))}",
                    f"progress={_fmt_float(float(pdm_result.ego_progress))}",
                    f"ttc={_fmt_float(float(pdm_result.time_to_collision_within_bound))}",
                    f"lane_keeping={_fmt_float(float(pdm_result.lane_keeping))}",
                    f"history_comfort={_fmt_float(float(pdm_result.history_comfort))}",
                    f"comfort={_fmt_float(float(pdm_result.comfort))}",
                    f"reported_score={_fmt_float(_get_metric_float(pdm_result, 'pdm_score', 'score'))}",
                    f"reconstructed_score={_fmt_float(float(reconstructed_score))}",
                    f"collision_s={_fmt_float(collision_s)}",
                    f"ttc_s={_fmt_float(ttc_s)}",
                )
                printed_runtime_alignment = True

            if args.save_viz and target_builder is not None and viz_saved < local_viz_quota:
                try:
                    if targets is None:
                        if scene is None:
                            scene = _build_scene_without_print(scene_loader, Scene, token)
                        targets = target_builder.compute_targets(scene)
                    if len(gt_future_video) <= 0:
                        future_frames = targets.get("future_image_paths") or targets.get("future_frames")
                        if future_frames is None:
                            raise ValueError("Missing future frames from target builder.")
                        gt_future_video = _resolve_video_pil(
                            future_frames,
                            height=args.height,
                            width=args.width,
                            surround_view=args.surround_view,
                            normalize_mode="[-1,1]",
                            camera_intrinsics=targets.get("future_camera_intrinsics"),
                            focal_config=focal_config,
                        )
                    gt_video = history_video + gt_future_video
                    if gt_traj is None:
                        gt_traj = np.asarray(targets["trajectory"], dtype=np.float32)
                        gt_traj = _safe_traj_np(gt_traj, scene_filter.num_future_frames)
                    cmd_text = one_hot_to_cmd(driving_command)
                    yaw_deg = 0.0
                    if hist_np.ndim == 2 and hist_np.shape[0] > 0 and hist_np.shape[1] > 2:
                        yaw_deg = float(np.degrees(float(hist_np[-1, 2])))
                    extra_info = {
                        "nav_cmd": cmd_text,
                        "speed_mps": float(speed),
                        "yaw_deg": float(yaw_deg),
                    }
                    camera_for_projection = None
                    num_history_frames = None
                    projection_image_path = None
                    viz_index = viz_saved + 1
                    viz_index_prefix = format_viz_index_prefix(viz_index)
                    if scene is not None:
                        try:
                            num_history_frames = int(scene.scene_metadata.num_history_frames)
                            cur_idx = max(0, num_history_frames - 1)
                            camera_for_projection = getattr(scene.frames[cur_idx].cameras, "cam_f0", None)
                            if camera_for_projection is not None:
                                projection_dir = viz_dir / "projected_traj"
                                projection_dir.mkdir(parents=True, exist_ok=True)
                                projection_image_path = projection_dir / f"{viz_index_prefix}{token}_rank{rank}.png"
                        except Exception:
                            camera_for_projection = None
                            num_history_frames = None
                            projection_image_path = None
                    viz_path = viz_dir / f"{viz_index_prefix}{token}_rank{rank}.mp4"
                    save_viz_video(
                        out_path=viz_path,
                        gt_video=gt_video,
                        pred_video=pred_video,
                        history_traj=hist_np,
                        gt_future_traj=gt_traj,
                        pred_future_traj=traj_np,
                        width=args.width,
                        height=args.height,
                        plot_height=DEFAULT_VIZ_PLOT_HEIGHT,
                        fps=int(args.target_fps),
                        score_dict=pdm_dict,
                        extra_info=extra_info,
                        camera_for_projection=camera_for_projection,
                        num_history_frames=num_history_frames,
                        gt_future_max_steps=(
                            int(args.model_future_frames)
                            if float(getattr(args, "scene_future_extra_seconds", 0.0)) > 0.0
                            else None
                        ),
                        projection_image_path=projection_image_path,
                        projection_overlay_on_video=False,
                    )
                    score_row["viz_path"] = str(viz_path)
                    if projection_image_path is not None:
                        score_row["viz_projection_path"] = str(projection_image_path)
                    if scene is not None:
                        try:
                            bev_dir = viz_dir / "bev_with_agent"
                            bev_paths = save_bev_with_agent_artifacts(
                                out_dir=bev_dir,
                                token=f"{viz_index_prefix}{token}",
                                scene=scene,
                                pred_future_traj=traj_np,
                                rank=rank,
                            )
                            score_row["viz_bev_image"] = str(bev_paths.get("image_path", ""))
                            score_row["viz_bev_gif"] = str(bev_paths.get("gif_path", ""))
                        except Exception as bev_exc:
                            score_row["viz_bev_error"] = str(bev_exc)
                            if args.print_errors or rank == 0:
                                print(f"[eval][viz_bev_error] token={token} err={bev_exc}", flush=True)
                    viz_saved += 1
                except Exception as viz_exc:
                    score_row["viz_error"] = str(viz_exc)
                    if args.print_errors:
                        print(f"[eval][viz_error] token={token} err={viz_exc}", flush=True)
        except Exception as exc:
            score_row["valid"] = False
            score_row["error"] = str(exc)
            if args.print_errors:
                print(f"[eval][error] token={token} err={exc}", flush=True)
        results.append(score_row)
        if score_row["valid"]:
            local_success += 1
            infer_time_value = score_row.get("infer_time_s")
            if infer_time_value is not None:
                local_last_infer_time_s = float(infer_time_value)
                local_infer_time_sum_s += local_last_infer_time_s
                local_infer_time_count += 1
        else:
            local_failed += 1
        if args.show_eval_progress and rank == 0:
            elapsed_s = time.perf_counter() - eval_start
            avg_s = elapsed_s / max(1, step_idx)
            eta_s = avg_s * max(0, len(local_tokens) - step_idx)
            infer_avg_s = local_infer_time_sum_s / max(1, local_infer_time_count)
            infer_part = (
                f" infer_last={local_last_infer_time_s:.3f}s infer_avg={infer_avg_s:.3f}s"
                if local_infer_time_count > 0
                else ""
            )
            token_iter.set_postfix_str(
                f"ok={local_success} fail={local_failed} elapsed={_format_duration(elapsed_s)} eta={_format_duration(eta_s)}{infer_part}"
            )
    if args.show_eval_progress and rank == 0 and hasattr(token_iter, "close"):
        token_iter.close()

    final_results = _gather_results(results, device=device, rank=rank, world_size=world_size)
    if rank == 0 and final_results is not None:
        import pandas as pd

        elapsed_s = time.perf_counter() - eval_start
        total_scenes = len(final_results)
        avg_scene_s = elapsed_s / max(1, total_scenes)
        print(f"[eval] elapsed={elapsed_s:.2f}s avg_per_scene={avg_scene_s:.2f}s scenes={total_scenes}")

        os.makedirs(args.output_dir, exist_ok=True)
        df = pd.DataFrame(final_results)
        num_success = df["valid"].sum()
        num_failed = len(df) - num_success
        valid_rows = df[df["valid"] == True] if "valid" in df.columns else df
        # Compute averages only on numeric metric columns.
        metric_df = df.drop(columns=["token", "valid", "rank", "error"], errors="ignore")
        metric_df = metric_df.select_dtypes(include=[np.number, "bool"])
        avg_row = metric_df.mean(skipna=True)
        avg_row["token"] = "average"
        avg_row["valid"] = bool(df["valid"].all())
        avg_row["rank"] = "0"
        df.loc[len(df)] = avg_row

        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        out_path = Path(args.output_dir) / f"pdm_score_{timestamp}.csv"
        df.to_csv(out_path, index=False)
        def _summary_dict_from_series(
            metric_series: "pd.Series",
            *,
            include_null_prefixes: Optional[tuple[str, ...]] = None,
        ) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for metric_name in metric_df.columns:
                metric_value = metric_series.get(metric_name, np.nan)
                if pd.isna(metric_value):
                    if include_null_prefixes and any(str(metric_name).startswith(prefix) for prefix in include_null_prefixes):
                        out[str(metric_name)] = None
                    continue
                if isinstance(metric_value, (bool, np.bool_)):
                    out[str(metric_name)] = bool(metric_value)
                else:
                    out[str(metric_name)] = float(metric_value)
            return out

        summary_metrics = {
            "all": _summary_dict_from_series(avg_row),
        }
        summary_out_path = Path(args.output_dir) / f"pdm_score_{timestamp}.summary.json"
        with summary_out_path.open("w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
        if len(metric_df.columns) > 0:
            print("[eval] average metrics:")
            for metric_name in metric_df.columns:
                metric_value = avg_row.get(metric_name, np.nan)
                if pd.isna(metric_value):
                    continue
                print(f"[eval][avg] {metric_name}={float(metric_value):.6f}")
        if "infer_time_s" in valid_rows.columns:
            infer_times = pd.to_numeric(valid_rows["infer_time_s"], errors="coerce").dropna()
            if len(infer_times) > 0:
                print(
                    "[eval][infer_time]",
                    f"count={len(infer_times)}",
                    f"mean={float(infer_times.mean()):.6f}s",
                    f"median={float(infer_times.median()):.6f}s",
                    f"p90={float(infer_times.quantile(0.90)):.6f}s",
                    f"min={float(infer_times.min()):.6f}s",
                    f"max={float(infer_times.max()):.6f}s",
                    f"mode={str(valid_rows.get('infer_mode', 'unknown').dropna().iloc[0]) if 'infer_mode' in valid_rows.columns and valid_rows['infer_mode'].dropna().shape[0] > 0 else 'unknown'}",
                )
        if "gpu_mem_alloc_peak_mb" in valid_rows.columns:
            peak_alloc = pd.to_numeric(valid_rows["gpu_mem_alloc_peak_mb"], errors="coerce").dropna()
            delta_alloc = pd.to_numeric(valid_rows.get("gpu_mem_alloc_peak_delta_mb"), errors="coerce").dropna()
            peak_reserved = pd.to_numeric(valid_rows.get("gpu_mem_reserved_peak_mb"), errors="coerce").dropna()
            delta_reserved = pd.to_numeric(valid_rows.get("gpu_mem_reserved_peak_delta_mb"), errors="coerce").dropna()
            if len(peak_alloc) > 0:
                parts = [
                    f"count={len(peak_alloc)}",
                    f"peak_alloc_mean={float(peak_alloc.mean()):.1f}MB",
                    f"peak_alloc_max={float(peak_alloc.max()):.1f}MB",
                ]
                if len(delta_alloc) > 0:
                    parts.extend(
                        [
                            f"peak_alloc_delta_mean={float(delta_alloc.mean()):.1f}MB",
                            f"peak_alloc_delta_max={float(delta_alloc.max()):.1f}MB",
                        ]
                    )
                if len(peak_reserved) > 0:
                    parts.extend(
                        [
                            f"peak_reserved_mean={float(peak_reserved.mean()):.1f}MB",
                            f"peak_reserved_max={float(peak_reserved.max()):.1f}MB",
                        ]
                    )
                if len(delta_reserved) > 0:
                    parts.extend(
                        [
                            f"peak_reserved_delta_mean={float(delta_reserved.mean()):.1f}MB",
                            f"peak_reserved_delta_max={float(delta_reserved.max()):.1f}MB",
                        ]
                    )
                print("[eval][infer_memory] " + " ".join(parts))
        if args.enable_nuscenes_metrics and len(nuscenes_metric_horizons_s) > 0:
            l2_parts: List[str] = []
            collision_parts: List[str] = []
            for horizon_s in nuscenes_metric_horizons_s:
                label = _format_horizon_label(horizon_s)
                l2_key = f"nuscenes_l2_{label}_m"
                collision_key = f"nuscenes_collision_{label}_pct"
                l2_val = avg_row.get(l2_key, np.nan)
                collision_val = avg_row.get(collision_key, np.nan)
                if not pd.isna(l2_val):
                    l2_parts.append(f"{label}={float(l2_val):.4f}")
                if not pd.isna(collision_val):
                    collision_parts.append(f"{label}={float(collision_val):.2f}")
            if l2_parts:
                print("[eval][nuscenes][avg] L2 (m): " + ", ".join(l2_parts))
            if collision_parts:
                print("[eval][nuscenes][avg] Collision (%): " + ", ".join(collision_parts))
        if args.save_viz:
            viz_count = int((df.get("viz_path").notna().sum()) if "viz_path" in df.columns else 0)
            print(f"[eval] viz_saved={viz_count}")
        print(f"[eval] Done. success={num_success} failed={num_failed} saved={out_path}")

def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
