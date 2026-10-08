"""
nuScenes inference/evaluation script for Wan video pipeline.

Focus metrics:
- L2 (m) @ requested horizons (default: 1s/2s/3s)
- Collision (%) @ requested horizons (default: 1s/2s/3s)
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from examples.wanvideo.driveva_infer.navsim_eval_viz import format_viz_index_prefix, save_viz_video

try:
    from .navsim_dataset import (
        DEFAULT_NEGATIVE_PROMPT,
        FocalUnificationConfig,
        _build_prompt_fixed,
        one_hot_to_cmd,
        preprocess_camera_image,
    )
except ImportError:
    from navsim_dataset import (
        DEFAULT_NEGATIVE_PROMPT,
        FocalUnificationConfig,
        _build_prompt_fixed,
        one_hot_to_cmd,
        preprocess_camera_image,
    )

DEFAULT_VIZ_PLOT_HEIGHT = 420


@dataclass
class _NuScenesTokenState:
    token: str
    timestamp_s: float
    ego_x: float
    ego_y: float
    ego_z: float
    ego_yaw: float
    image_path: str
    ann_boxes: List[Tuple[float, float, float, float, float, float, str]]
    camera_intrinsics: Optional[np.ndarray] = None
    camera_sensor2ego_rotation: Optional[np.ndarray] = None
    camera_sensor2ego_translation: Optional[np.ndarray] = None


@dataclass
class _NuScenesSceneCache:
    scene_name: str
    token_states: List[_NuScenesTokenState]
    timestamps_s: np.ndarray


@dataclass
class _TokenAnchorMeta:
    scene_idx: int
    scene_name: str
    anchor_idx: int
    min_anchor: int
    max_anchor: int

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer/evaluate Wan video model on nuScenes.")

    # Data args
    parser.add_argument("--nuscenes_dataroot", type=str, required=True)
    parser.add_argument("--nuscenes_version", type=str, default="v1.0-trainval")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--camera_name", type=str, default="CAM_FRONT")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument(
        "--policy_anno_json",
        type=str,
        default=None,
        help=(
            "Optional Policy-World-Model annotation json "
            "(e.g. plan_val_filter_w_ego_w_cmd_1s_to_19s.json). "
            "When set, only tokens present in this file are evaluated."
        ),
    )
    parser.add_argument("--token_offset", type=int, default=0, help="Skip first N local samples after rank split.")
    parser.add_argument("--max_eval_tokens", type=int, default=None, help="Evaluate at most this many local samples after offset.")
    parser.add_argument("--resume_csv", type=str, default=None, help="CSV path to skip already evaluated tokens in column `token`.")
    parser.add_argument("--print_tokens", action="store_true")
    parser.add_argument("--show_eval_progress", dest="show_eval_progress", action="store_true")
    parser.add_argument("--no_show_eval_progress", dest="show_eval_progress", action="store_false")
    parser.add_argument("--print_errors", dest="print_errors", action="store_true")
    parser.add_argument("--no_print_errors", dest="print_errors", action="store_false")
    parser.set_defaults(show_eval_progress=True, print_errors=True)

    # Video args
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--use_focal_unification", action="store_true")
    parser.add_argument("--focal_target_px", type=str, default="")
    parser.add_argument("--num_history_frames", type=int, default=5)
    parser.add_argument("--num_future_frames", type=int, default=8)
    parser.add_argument(
        "--model_future_frames",
        type=int,
        default=8,
        help="Future trajectory points produced by model.",
    )
    parser.add_argument(
        "--scene_future_extra_seconds",
        type=float,
        default=0.0,
        help="Ensure scene future horizon is at least model horizon + this many seconds.",
    )
    parser.add_argument("--target_fps", type=int, default=2)
    parser.add_argument("--command_yaw_threshold_deg", type=float, default=8.0)
    parser.add_argument(
        "--metric_horizons_s",
        type=str,
        default="1,2,3",
        help="Comma/space-separated horizon seconds, e.g. '1,2,3'.",
    )
    parser.add_argument("--ego_box_length_m", type=float, default=4.084, help="Ego box length used for collision check.")
    parser.add_argument("--ego_box_width_m", type=float, default=1.85, help="Ego box width used for collision check.")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--viz_dir", type=str, default=None, help="Directory for visualization outputs. Default: <output_dir>/viz")
    parser.add_argument("--viz_max_tokens", type=int, default=20, help="Global max number of tokens to visualize.")
    parser.add_argument(
        "--save_projected_traj_image",
        dest="save_projected_traj_image",
        action="store_true",
        help="Save current-frame trajectory projection image under <viz_dir>/projected_traj.",
    )
    parser.add_argument(
        "--no_save_projected_traj_image",
        dest="save_projected_traj_image",
        action="store_false",
        help="Disable current-frame trajectory projection image export.",
    )
    parser.set_defaults(save_projected_traj_image=True)
    parser.add_argument("--show_denoise_progress", action="store_true", help="Show per-token denoise tqdm.")
    parser.add_argument("--show_vae_progress", action="store_true", help="Show VAE tiled decode/encode tqdm.")
    parser.add_argument("--debug_prompt_steps", type=int, default=30, help="Print positive/negative prompt for first N steps on rank0.")
    parser.add_argument("--use_mixed_latent_attention_mask", dest="use_mixed_latent_attention_mask", action="store_true")
    parser.add_argument("--no_use_mixed_latent_attention_mask", dest="use_mixed_latent_attention_mask", action="store_false")
    parser.add_argument("--infer_trajectory_only", dest="infer_trajectory_only", action="store_true")
    parser.add_argument("--no_infer_trajectory_only", dest="infer_trajectory_only", action="store_false")
    parser.add_argument("--infer_output_mode", type=str, default=None, choices=["trajectory", "video", "both"])
    parser.set_defaults(use_mixed_latent_attention_mask=True, infer_trajectory_only=False)
    # Model args
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--full_ckpt", type=str, required=True, help="Path to full checkpoint (.safetensors).")
    parser.add_argument("--num_inference_steps", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--trajectory_condition_mode",
        type=str,
        default="velocity",
        choices=["auto", "history", "velocity"],
        help="Trajectory prefix conditioning mode. Defaults to velocity mode.",
    )
    # Output
    parser.add_argument("--output_dir", type=str, required=True)

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


def _gather_results(
    results: List[Dict[str, Any]],
    device: torch.device,
    rank: int,
    world_size: int,
) -> Optional[List[Dict[str, Any]]]:
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


def _normalize_checkpoint_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            norm_key = f"dit.{norm_key}"
        normalized[norm_key] = value
    return normalized


def _print_ckpt_load_report(
    missing: List[str],
    unexpected: List[str],
    *,
    rank: int,
) -> None:
    if rank != 0:
        return

    def _prefix(key: str) -> str:
        return key.split(".", 1)[0] if "." in key else key

    if missing:
        miss_counter = Counter(_prefix(k) for k in missing)
        miss_summary = ", ".join(f"{k}:{v}" for k, v in miss_counter.most_common())
        print(f"[infer] ckpt missing summary by module: {miss_summary}")
    if unexpected:
        unexp_counter = Counter(_prefix(k) for k in unexpected)
        unexp_summary = ", ".join(f"{k}:{v}" for k, v in unexp_counter.most_common())
        print(f"[infer] ckpt unexpected summary by module: {unexp_summary}")


def _safe_traj_np(traj: np.ndarray, expected_len: int) -> np.ndarray:
    if traj.shape[0] == expected_len:
        return traj
    if traj.shape[0] > expected_len:
        return traj[:expected_len]
    pad = np.repeat(traj[-1:], expected_len - traj.shape[0], axis=0)
    return np.concatenate([traj, pad], axis=0)


def _cmd_from_future_yaw(future_xyh: np.ndarray, yaw_thresh_deg: float) -> np.ndarray:
    if future_xyh.ndim != 2 or future_xyh.shape[0] == 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    yaw_deg = float(np.degrees(float(future_xyh[-1, 2])))
    if yaw_deg > yaw_thresh_deg:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if yaw_deg < -yaw_thresh_deg:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _parse_horizons_s(raw: str) -> List[float]:
    chunks = re.split(r"[,\s]+", str(raw).strip())
    horizons: List[float] = []
    for chunk in chunks:
        if not chunk:
            continue
        value = float(chunk)
        if value <= 0:
            raise ValueError(f"horizon must be > 0, got {value}")
        horizons.append(value)
    if not horizons:
        raise ValueError(f"Invalid --metric_horizons_s: {raw!r}")
    horizons = sorted(horizons)
    deduped: List[float] = []
    for h in horizons:
        if not deduped or abs(deduped[-1] - h) > 1e-6:
            deduped.append(h)
    return deduped


def _extract_policy_token(entry: Any) -> Optional[str]:
    # Match Policy DatasetNuScenes logic:
    # omini_anno = self.omini_annos[idx][0]; token = omini_anno[-1]["token"].
    try:
        if isinstance(entry, list) and len(entry) > 0 and isinstance(entry[0], list):
            omini_anno = entry[0]
        else:
            omini_anno = entry
        if isinstance(omini_anno, list) and len(omini_anno) > 0 and isinstance(omini_anno[-1], dict):
            tok = omini_anno[-1].get("token")
            if isinstance(tok, str) and tok:
                return tok
    except Exception:
        pass

    # Fallback for slightly different packing formats.
    candidates: List[str] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            tok = obj.get("token")
            if isinstance(tok, str) and tok:
                candidates.append(tok)
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(entry)
    if len(candidates) == 0:
        return None
    for tok in candidates:
        if re.fullmatch(r"[0-9a-fA-F]{32}", tok):
            return tok
    return candidates[0]


def _load_policy_filter_tokens(path: str, *, verbose: bool = True) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Policy annotation json must be a list, got {type(raw).__name__}")
    tokens: List[str] = []
    missing = 0
    for item in raw:
        token = _extract_policy_token(item)
        if token is None:
            missing += 1
            continue
        tokens.append(str(token))
    if len(tokens) == 0:
        raise ValueError(
            f"No valid tokens found in Policy annotation json: {path}"
        )
    if missing > 0 and verbose:
        print(
            "[infer][policy_filter][warn]",
            f"ignored_entries_without_token={missing}",
            f"total_entries={len(raw)}",
            f"usable_tokens={len(tokens)}",
        )
    return tokens


def _unique_preserve_order(items: List[str]) -> Tuple[List[str], int]:
    seen = set()
    out: List[str] = []
    dup = 0
    for item in items:
        if item in seen:
            dup += 1
            continue
        seen.add(item)
        out.append(item)
    return out, dup


def _format_horizon_label(horizon_s: float) -> str:
    rounded = int(round(float(horizon_s)))
    if abs(float(horizon_s) - float(rounded)) <= 1e-6:
        return f"{rounded}s"
    label = f"{float(horizon_s):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{label}s"


def _interp_xyh_at_horizon(
    traj_xyh: np.ndarray, times_s: np.ndarray, horizon_s: float
) -> Optional[np.ndarray]:
    traj_xyh = np.asarray(traj_xyh, dtype=np.float32)
    times_s = np.asarray(times_s, dtype=np.float64)
    if traj_xyh.ndim != 2 or traj_xyh.shape[0] == 0 or traj_xyh.shape[1] < 3:
        return None
    if times_s.ndim != 1 or times_s.shape[0] != traj_xyh.shape[0]:
        return None
    if horizon_s < float(times_s[0]) - 1e-6 or horizon_s > float(times_s[-1]) + 1e-6:
        return None
    x = float(np.interp(horizon_s, times_s, traj_xyh[:, 0]))
    y = float(np.interp(horizon_s, times_s, traj_xyh[:, 1]))
    yaw_unwrapped = np.unwrap(traj_xyh[:, 2].astype(np.float64))
    yaw = float(np.interp(horizon_s, times_s, yaw_unwrapped))
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    return np.array([x, y, yaw], dtype=np.float32)


def _local_xy_to_global(x_local: float, y_local: float, anchor_x: float, anchor_y: float, anchor_yaw: float) -> Tuple[float, float]:
    c = math.cos(anchor_yaw)
    s = math.sin(anchor_yaw)
    x_global = anchor_x + c * float(x_local) - s * float(y_local)
    y_global = anchor_y + s * float(x_local) + c * float(y_local)
    return float(x_global), float(y_global)


def _obb_intersects(
    center_a: Tuple[float, float],
    length_a: float,
    width_a: float,
    yaw_a: float,
    center_b: Tuple[float, float],
    length_b: float,
    width_b: float,
    yaw_b: float,
) -> bool:
    ax = np.array([math.cos(yaw_a), math.sin(yaw_a)], dtype=np.float64)
    ay = np.array([-math.sin(yaw_a), math.cos(yaw_a)], dtype=np.float64)
    bx = np.array([math.cos(yaw_b), math.sin(yaw_b)], dtype=np.float64)
    by = np.array([-math.sin(yaw_b), math.cos(yaw_b)], dtype=np.float64)

    d = np.array(center_b, dtype=np.float64) - np.array(center_a, dtype=np.float64)
    axes = (ax, ay, bx, by)
    ha_l = max(float(length_a) * 0.5, 1e-3)
    ha_w = max(float(width_a) * 0.5, 1e-3)
    hb_l = max(float(length_b) * 0.5, 1e-3)
    hb_w = max(float(width_b) * 0.5, 1e-3)

    for axis in axes:
        proj_dist = abs(float(np.dot(d, axis)))
        proj_a = ha_l * abs(float(np.dot(ax, axis))) + ha_w * abs(float(np.dot(ay, axis)))
        proj_b = hb_l * abs(float(np.dot(bx, axis))) + hb_w * abs(float(np.dot(by, axis)))
        if proj_dist > (proj_a + proj_b + 1e-8):
            return False
    return True


_BEV_X_MIN_M = -50.0
_BEV_X_MAX_M = 50.0
_BEV_Y_MIN_M = -50.0
_BEV_Y_MAX_M = 50.0
_BEV_LEGACY_RES_M = 0.1
_LEGACY_EGO_FRONT_SHIFT_M = 0.5 + 0.985793


def _bev_dims(bev_res_m: float) -> Tuple[int, int]:
    bev_w = int(round((_BEV_X_MAX_M - _BEV_X_MIN_M) / float(bev_res_m)))
    bev_h = int(round((_BEV_Y_MAX_M - _BEV_Y_MIN_M) / float(bev_res_m)))
    return bev_w, bev_h


def _resolve_annotation_category_name(
    *,
    nusc: Any,
    ann: Dict[str, Any],
    instance_category_cache: Dict[str, str],
) -> str:
    category_name = str(ann.get("category_name", "") or "")
    if category_name:
        return category_name
    instance_token = str(ann.get("instance_token", "") or "")
    if not instance_token:
        return ""
    if instance_token in instance_category_cache:
        return instance_category_cache[instance_token]
    resolved = ""
    try:
        instance_rec = nusc.get("instance", instance_token)
        category_token = str(instance_rec.get("category_token", "") or "")
        if category_token:
            category_rec = nusc.get("category", category_token)
            resolved = str(category_rec.get("name", "") or "")
    except Exception:
        resolved = ""
    instance_category_cache[instance_token] = resolved
    return resolved


def _global_xy_yaw_to_local(
    x_global: float,
    y_global: float,
    yaw_global: float,
    anchor_x: float,
    anchor_y: float,
    anchor_yaw: float,
) -> Tuple[float, float, float]:
    c = math.cos(-float(anchor_yaw))
    s = math.sin(-float(anchor_yaw))
    dx = float(x_global) - float(anchor_x)
    dy = float(y_global) - float(anchor_y)
    x_local = c * dx - s * dy
    y_local = s * dx + c * dy
    yaw_local = float(
        np.arctan2(
            np.sin(float(yaw_global) - float(anchor_yaw)),
            np.cos(float(yaw_global) - float(anchor_yaw)),
        )
    )
    return float(x_local), float(y_local), float(yaw_local)


def _oriented_box_corners(
    center_x: float,
    center_y: float,
    yaw: float,
    length: float,
    width: float,
) -> List[Tuple[float, float]]:
    half_l = float(length) * 0.5
    half_w = float(width) * 0.5
    corners_local = np.asarray(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    corners = corners_local @ rot.T
    corners[:, 0] += float(center_x)
    corners[:, 1] += float(center_y)
    return [(float(x), float(y)) for x, y in corners]


def _ego_box_corners_from_rear_axle(
    rear_x: float,
    rear_y: float,
    yaw: float,
    length: float,
    width: float,
    *,
    front_shift_m: float,
) -> List[Tuple[float, float]]:
    cx = float(rear_x) + float(front_shift_m) * math.cos(float(yaw))
    cy = float(rear_y) + float(front_shift_m) * math.sin(float(yaw))
    return _oriented_box_corners(cx, cy, yaw, length, width)


def _to_bev_pixels(
    poly_xy: List[Tuple[float, float]],
    *,
    bev_res_m: float,
) -> List[Tuple[float, float]]:
    pixels: List[Tuple[float, float]] = []
    for x_m, y_m in poly_xy:
        px = (float(x_m) - _BEV_X_MIN_M) / float(bev_res_m)
        py = (float(y_m) - _BEV_Y_MIN_M) / float(bev_res_m)
        pixels.append((float(px), float(py)))
    return pixels


def _build_bev_occupancy_from_future_states(
    future_states: List[_NuScenesTokenState],
    *,
    anchor_x: float,
    anchor_y: float,
    anchor_yaw: float,
    bev_res_m: float,
) -> np.ndarray:
    bev_w, bev_h = _bev_dims(bev_res_m)
    if len(future_states) == 0:
        return np.zeros((0, bev_h, bev_w), dtype=np.bool_)
    occupancy = np.zeros((len(future_states), bev_h, bev_w), dtype=np.bool_)
    for step_idx, state in enumerate(future_states):
        img = Image.new("1", (bev_w, bev_h), 0)
        drawer = ImageDraw.Draw(img)
        for (bx, by, _bz, byaw, bl, bw, _category_name) in state.ann_boxes:
            lx, ly, lyaw = _global_xy_yaw_to_local(
                float(bx),
                float(by),
                float(byaw),
                float(anchor_x),
                float(anchor_y),
                float(anchor_yaw),
            )
            box_poly = _oriented_box_corners(lx, ly, lyaw, float(bl), float(bw))
            drawer.polygon(_to_bev_pixels(box_poly, bev_res_m=bev_res_m), fill=1, outline=1)
        occupancy[step_idx] = np.asarray(img, dtype=np.uint8) > 0
    return occupancy


def _compute_policy_bev_collision_flags(
    traj_local_xy: np.ndarray,
    occupancy: np.ndarray,
    *,
    ego_length_m: float,
    ego_width_m: float,
    bev_res_m: float,
    ego_front_shift_m: float,
) -> np.ndarray:
    traj_local_xy = np.asarray(traj_local_xy, dtype=np.float64)
    n_steps = int(min(traj_local_xy.shape[0], occupancy.shape[0]))
    bev_w, bev_h = _bev_dims(bev_res_m)
    flags = np.full(n_steps, np.nan, dtype=np.float32)
    prev = np.zeros(2, dtype=np.float64)
    for step_idx in range(n_steps):
        curr = traj_local_xy[step_idx]
        if not np.isfinite(curr).all():
            continue
        delta = curr - prev
        if float(np.linalg.norm(delta)) < 1.0:
            yaw = 0.0
        else:
            yaw = float(np.arctan2(float(delta[1]), float(delta[0])))
        ego_poly = _ego_box_corners_from_rear_axle(
            float(curr[0]),
            float(curr[1]),
            yaw,
            float(ego_length_m),
            float(ego_width_m),
            front_shift_m=float(ego_front_shift_m),
        )
        ego_img = Image.new("1", (bev_w, bev_h), 0)
        ego_draw = ImageDraw.Draw(ego_img)
        ego_draw.polygon(_to_bev_pixels(ego_poly, bev_res_m=bev_res_m), fill=1, outline=1)
        ego_mask = np.asarray(ego_img, dtype=np.uint8) > 0
        collided = bool(np.any(np.logical_and(ego_mask, occupancy[step_idx])))
        flags[step_idx] = 1.0 if collided else 0.0
        prev = curr
    return flags


def _compute_collision_percent_with_horizon(
    *,
    pred_step_collision: np.ndarray,
    gt_step_collision: np.ndarray,
    future_times_s: np.ndarray,
    horizon_s: float,
) -> float:
    horizon_indices = np.where(future_times_s <= float(horizon_s) + 1e-6)[0]
    if horizon_indices.size <= 0:
        return float("nan")
    coll_values: List[float] = []
    for step_idx in horizon_indices.tolist():
        pred_coll = float(pred_step_collision[step_idx])
        if not np.isfinite(pred_coll):
            continue
        if float(gt_step_collision[step_idx]) > 0.5:
            pred_coll = 0.0
        coll_values.append(pred_coll)
    if len(coll_values) <= 0:
        return float("nan")
    return float(np.mean(np.asarray(coll_values, dtype=np.float32)) * 100.0)


def _compute_l2_vad_with_horizon(
    *,
    pred_step_xy_local: np.ndarray,
    gt_step_xy_local: np.ndarray,
    future_times_s: np.ndarray,
    horizon_s: float,
) -> float:
    horizon_indices = np.where(future_times_s <= float(horizon_s) + 1e-6)[0]
    if horizon_indices.size <= 0:
        return float("nan")
    l2_vals: List[float] = []
    for step_idx in horizon_indices.tolist():
        if step_idx >= int(pred_step_xy_local.shape[0]) or step_idx >= int(gt_step_xy_local.shape[0]):
            continue
        pred_xy = np.asarray(pred_step_xy_local[step_idx], dtype=np.float64)
        gt_xy = np.asarray(gt_step_xy_local[step_idx], dtype=np.float64)
        if pred_xy.shape[0] < 2 or gt_xy.shape[0] < 2:
            continue
        if not np.isfinite(pred_xy).all() or not np.isfinite(gt_xy).all():
            continue
        l2_vals.append(float(np.linalg.norm(pred_xy[:2] - gt_xy[:2])))
    if len(l2_vals) <= 0:
        return float("nan")
    return float(np.mean(np.asarray(l2_vals, dtype=np.float32)))


def _no_progress_bar(iterable, *args, **kwargs):
    return iterable


def _set_vae_progress(show_vae_progress: bool) -> None:
    try:
        import diffsynth.models.wan_video_vae as wan_video_vae_mod
    except Exception:
        return
    if not hasattr(wan_video_vae_mod, "_original_tqdm_for_eval"):
        wan_video_vae_mod._original_tqdm_for_eval = wan_video_vae_mod.tqdm
    wan_video_vae_mod.tqdm = wan_video_vae_mod._original_tqdm_for_eval if show_vae_progress else _no_progress_bar


def _load_nuscenes_devkit():
    devkit_sdk = Path(__file__).resolve().parents[3] / "third_party" / "nuscenes-devkit" / "python-sdk"
    if devkit_sdk.exists():
        sdk_path = str(devkit_sdk)
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
    try:
        from pyquaternion import Quaternion
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import get_scenes_of_split
        from nuscenes.eval.common.utils import quaternion_yaw
        from nuscenes.eval.prediction.metrics import final_distances
    except Exception as exc:
        raise ImportError(
            "Failed to import nuscenes-devkit python-sdk. "
            "Please ensure <repo>/third_party/nuscenes-devkit/python-sdk is in PYTHONPATH."
        ) from exc
    return Quaternion, NuScenes, get_scenes_of_split, quaternion_yaw, final_distances


def _build_local_xyh(global_xyh: np.ndarray, anchor_xyh: np.ndarray) -> np.ndarray:
    gx = np.asarray(global_xyh[:, 0], dtype=np.float64)
    gy = np.asarray(global_xyh[:, 1], dtype=np.float64)
    gyaw = np.asarray(global_xyh[:, 2], dtype=np.float64)
    ax = float(anchor_xyh[0])
    ay = float(anchor_xyh[1])
    ayaw = float(anchor_xyh[2])
    c = math.cos(-ayaw)
    s = math.sin(-ayaw)
    dx = gx - ax
    dy = gy - ay
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    lyaw = np.arctan2(np.sin(gyaw - ayaw), np.cos(gyaw - ayaw))
    return np.stack([lx, ly, lyaw], axis=-1).astype(np.float32)


def _build_nuscenes_camera_for_projection(
    anchor_state: _NuScenesTokenState,
    intrinsics_override: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    intrinsics = intrinsics_override if intrinsics_override is not None else anchor_state.camera_intrinsics
    s2e_r = anchor_state.camera_sensor2ego_rotation
    s2e_t = anchor_state.camera_sensor2ego_translation
    if intrinsics is None or s2e_r is None or s2e_t is None:
        return None
    try:
        kk = np.asarray(intrinsics, dtype=np.float32)
        rr = np.asarray(s2e_r, dtype=np.float32)
        tt = np.asarray(s2e_t, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if kk.ndim != 2 or kk.shape[0] < 3 or kk.shape[1] < 3:
        return None
    if rr.shape != (3, 3) or tt.size < 3:
        return None
    out: Dict[str, Any] = {
        "intrinsics": kk,
        "sensor2lidar_rotation": rr,
        "sensor2lidar_translation": tt[:3],
    }
    if anchor_state.image_path:
        out["image"] = anchor_state.image_path
    return out


def _build_nuscenes_scene_cache(
    *,
    nusc: Any,
    scene_name: str,
    camera_name: str,
    quaternion_ctor: Any,
    quaternion_yaw_fn: Callable[[Any], float],
) -> _NuScenesSceneCache:
    scene_recs = [s for s in nusc.scene if s.get("name") == scene_name]
    if len(scene_recs) == 0:
        raise RuntimeError(f"scene not found in nuScenes metadata: {scene_name}")
    scene_rec = scene_recs[0]
    instance_category_cache: Dict[str, str] = {}

    states: List[_NuScenesTokenState] = []
    sample_token = scene_rec["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        if camera_name not in sample["data"]:
            raise RuntimeError(f"camera={camera_name} missing in sample={sample_token} scene={scene_name}")
        sd = nusc.get("sample_data", sample["data"][camera_name])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        q_ego = quaternion_ctor(ego_pose["rotation"])
        ego_yaw = float(quaternion_yaw_fn(q_ego))
        ego_x, ego_y, ego_z = [float(v) for v in ego_pose["translation"]]
        camera_intrinsics: Optional[np.ndarray] = None
        camera_sensor2ego_rotation: Optional[np.ndarray] = None
        camera_sensor2ego_translation: Optional[np.ndarray] = None
        try:
            intrinsic_raw = calib.get("camera_intrinsic")
            if intrinsic_raw is not None:
                intrinsic = np.asarray(intrinsic_raw, dtype=np.float32)
                if intrinsic.ndim == 2 and intrinsic.shape[0] >= 3 and intrinsic.shape[1] >= 3:
                    camera_intrinsics = intrinsic
            q_cam = quaternion_ctor(calib["rotation"])
            cam_rot = np.asarray(q_cam.rotation_matrix, dtype=np.float32)
            cam_trans = np.asarray(calib["translation"], dtype=np.float32).reshape(-1)
            if cam_rot.shape == (3, 3) and cam_trans.size >= 3:
                camera_sensor2ego_rotation = cam_rot
                camera_sensor2ego_translation = cam_trans[:3]
        except Exception:
            camera_intrinsics = None
            camera_sensor2ego_rotation = None
            camera_sensor2ego_translation = None

        ann_boxes: List[Tuple[float, float, float, float, float, float, str]] = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category_name = _resolve_annotation_category_name(
                nusc=nusc,
                ann=ann,
                instance_category_cache=instance_category_cache,
            )
            cx, cy, cz = [float(v) for v in ann["translation"]]
            w, l, _h = [float(v) for v in ann["size"]]
            q_ann = quaternion_ctor(ann["rotation"])
            ann_yaw = float(quaternion_yaw_fn(q_ann))
            ann_boxes.append((cx, cy, cz, ann_yaw, l, w, category_name))

        image_path = str(Path(nusc.dataroot) / sd["filename"])
        state = _NuScenesTokenState(
            token=str(sample_token),
            timestamp_s=float(sample["timestamp"]) / 1e6,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_z=ego_z,
            ego_yaw=ego_yaw,
            image_path=image_path,
            ann_boxes=ann_boxes,
            camera_intrinsics=camera_intrinsics,
            camera_sensor2ego_rotation=camera_sensor2ego_rotation,
            camera_sensor2ego_translation=camera_sensor2ego_translation,
        )
        states.append(state)
        sample_token = sample["next"]

    if len(states) == 0:
        raise RuntimeError(f"empty scene: {scene_name}")
    timestamps_s = np.asarray([s.timestamp_s for s in states], dtype=np.float64)
    return _NuScenesSceneCache(
        scene_name=scene_name,
        token_states=states,
        timestamps_s=timestamps_s,
    )


def _load_resized_rgb(path: str, width: int, height: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    return img


def _load_camera_rgb(
    state: _NuScenesTokenState,
    *,
    width: int,
    height: int,
    focal_config: FocalUnificationConfig,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    img = Image.open(state.image_path).convert("RGB")
    img, processed_intrinsics, _ = preprocess_camera_image(
        img,
        width=int(width),
        height=int(height),
        intrinsics=state.camera_intrinsics,
        focal_config=focal_config,
    )
    return img, processed_intrinsics


def _min_scene_future_frames(
    *,
    model_future_frames: int,
    model_fps: float,
    scene_fps: float,
    extra_seconds: float,
    min_eval_horizon_s: float,
) -> int:
    if model_future_frames <= 0:
        raise ValueError(f"model_future_frames must be > 0, got {model_future_frames}")
    if model_fps <= 0:
        raise ValueError(f"model_fps must be > 0, got {model_fps}")
    if scene_fps <= 0:
        raise ValueError(f"scene_fps must be > 0, got {scene_fps}")
    model_horizon_s = float(model_future_frames) / float(model_fps)
    required_horizon_s = max(model_horizon_s + max(0.0, float(extra_seconds)), float(min_eval_horizon_s))
    return max(1, int(np.ceil(required_horizon_s * float(scene_fps))))


def run_eval(args: argparse.Namespace, external_pipe: Optional[WanVideoPipeline] = None) -> None:
    dist_info = _init_distributed()
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]
    focal_config = FocalUnificationConfig.build(
        enabled=bool(getattr(args, "use_focal_unification", False)),
        target_focal_px=getattr(args, "focal_target_px", ""),
        output_width=int(args.width),
        output_height=int(args.height),
    )
    if rank == 0:
        print(
            "[infer][focal]",
            f"enabled={focal_config.enabled}",
            f"target_focal_px={focal_config.target_focal_px}",
            f"output={args.width}x{args.height}",
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    Quaternion, NuScenes, get_scenes_of_split, quaternion_yaw_fn, final_distances_fn = _load_nuscenes_devkit()

    if external_pipe is None:
        model_offload_device = str(device)
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
            print("[infer] using external pipeline")
    _set_vae_progress(args.show_vae_progress)
    pipe.target_fps = args.target_fps
    pipe.num_history_frames = int(args.num_history_frames)
    # nuScenes uses the same UNIVERSE trajectory head as NavSIM: local-frame
    # x/y/heading points are predicted directly at target_fps.
    pipe.trajectory_norm_mode = "UNIVERSE_odo"
    pipe.trajectory_use_relative = False
    pipe.trajectory_condition_mode = str(args.trajectory_condition_mode).lower()
    infer_output_mode = getattr(args, "infer_output_mode", None)
    infer_trajectory_only = bool(getattr(args, "infer_trajectory_only", False))
    if infer_output_mode == "trajectory":
        infer_trajectory_only = True
    elif infer_output_mode in {"video", "both"}:
        infer_trajectory_only = False
    pipe.use_mixed_latent_attention_mask = bool(args.use_mixed_latent_attention_mask)
    pipe.infer_trajectory_only = infer_trajectory_only

    if external_pipe is None:
        if not os.path.exists(args.full_ckpt):
            raise FileNotFoundError(f"full_ckpt not found: {args.full_ckpt}")
        if rank == 0:
            print(f"[infer] loading full checkpoint: {args.full_ckpt}")
        state_dict_raw = load_state_dict(args.full_ckpt)
        state_dict = _normalize_checkpoint_keys(state_dict_raw)
        has_traj_keys = any(
            k.startswith("trajectory_encoder.") or k.startswith("trajectory_head.")
            for k in state_dict.keys()
        )
        if has_traj_keys:
            missing, unexpected = pipe.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(
                    f"[infer] full_ckpt loaded into pipeline (dit+trajectory), "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            _print_ckpt_load_report(list(missing), list(unexpected), rank=rank)
        else:
            missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(
                    f"[infer][warn] full_ckpt has no trajectory_* keys; loading DiT only. "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            _print_ckpt_load_report(list(missing), list(unexpected), rank=rank)

    horizons_s = _parse_horizons_s(args.metric_horizons_s)
    max_horizon_s = float(max(horizons_s))
    scene_fps = 2.0
    min_future_frames_for_metrics = max(1, int(np.ceil(max_horizon_s * scene_fps)))
    min_future_frames_for_3s = max(1, int(np.ceil(3.0 * scene_fps)))
    min_future_frames_for_eligibility = (
        min_future_frames_for_3s if args.policy_anno_json else min_future_frames_for_metrics
    )
    min_scene_future_frames = _min_scene_future_frames(
        model_future_frames=int(args.model_future_frames),
        model_fps=float(args.target_fps),
        scene_fps=scene_fps,
        extra_seconds=float(args.scene_future_extra_seconds),
        min_eval_horizon_s=max_horizon_s,
    )
    if int(args.num_future_frames) < int(min_scene_future_frames):
        if rank == 0:
            print(
                "[infer][align][warn] num_future_frames too small; "
                f"bump {args.num_future_frames} -> {min_scene_future_frames} "
                f"(max_horizon={max_horizon_s}s, model_future={args.model_future_frames}, target_fps={args.target_fps})"
            )
        args.num_future_frames = int(min_scene_future_frames)

    pred_horizon_s = float(args.model_future_frames) / float(max(float(args.target_fps), 1e-6))
    if pred_horizon_s + 1e-6 < max_horizon_s and rank == 0:
        print(
            "[infer][warn] model horizon shorter than requested metric horizon:",
            f"model_horizon={pred_horizon_s:.2f}s requested_max={max_horizon_s:.2f}s",
            "Some metric horizons will be NaN.",
        )

    nusc = NuScenes(
        version=str(args.nuscenes_version),
        dataroot=str(args.nuscenes_dataroot),
        verbose=False,
    )
    scene_names = list(get_scenes_of_split(str(args.split), nusc))
    if args.max_scenes is not None:
        scene_names = scene_names[: max(0, int(args.max_scenes))]
    if len(scene_names) == 0:
        raise RuntimeError(
            f"No scenes found for split={args.split} version={args.nuscenes_version}."
        )

    scene_caches: List[_NuScenesSceneCache] = []
    sample_refs: List[Tuple[int, int]] = []
    num_history = int(args.num_history_frames)
    num_future = int(args.num_future_frames)
    if num_history < 1:
        raise ValueError(f"--num_history_frames must be >= 1, got {num_history}")
    if num_future < 1:
        raise ValueError(f"--num_future_frames must be >= 1, got {num_future}")

    if args.show_eval_progress and rank == 0:
        scene_iter = tqdm(
            scene_names,
            desc="Load-nuScenes-scenes",
            total=len(scene_names),
            dynamic_ncols=True,
        )
    else:
        scene_iter = scene_names

    token_anchor_meta: Dict[str, _TokenAnchorMeta] = {}
    for scene_name in scene_iter:
        cache = _build_nuscenes_scene_cache(
            nusc=nusc,
            scene_name=scene_name,
            camera_name=args.camera_name,
            quaternion_ctor=Quaternion,
            quaternion_yaw_fn=quaternion_yaw_fn,
        )
        scene_idx = len(scene_caches)
        scene_caches.append(cache)
        n = len(cache.token_states)
        min_anchor = num_history - 1
        # In policy-filter mode, require at least 3s future GT trajectory.
        max_anchor = n - int(min_future_frames_for_eligibility) - 1
        for anchor_idx in range(n):
            token = str(cache.token_states[anchor_idx].token)
            if token not in token_anchor_meta:
                token_anchor_meta[token] = _TokenAnchorMeta(
                    scene_idx=scene_idx,
                    scene_name=str(scene_name),
                    anchor_idx=int(anchor_idx),
                    min_anchor=int(min_anchor),
                    max_anchor=int(max_anchor),
                )
        if max_anchor < min_anchor:
            continue
        for anchor_idx in range(min_anchor, max_anchor + 1):
            sample_refs.append((scene_idx, anchor_idx))
    if args.show_eval_progress and rank == 0 and hasattr(scene_iter, "close"):
        scene_iter.close()

    if args.policy_anno_json:
        policy_path = Path(str(args.policy_anno_json)).expanduser()
        if not policy_path.exists():
            raise FileNotFoundError(f"policy_anno_json not found: {policy_path}")
        policy_tokens = _load_policy_filter_tokens(str(policy_path), verbose=(rank == 0))
        policy_tokens, dup_tokens = _unique_preserve_order(policy_tokens)
        ref_by_token: Dict[str, Tuple[int, int]] = {}
        for scene_idx, anchor_idx in sample_refs:
            token = str(scene_caches[scene_idx].token_states[anchor_idx].token)
            # Keep earliest matching ref when duplicate tokens exist.
            if token not in ref_by_token:
                ref_by_token[token] = (scene_idx, anchor_idx)
        filtered_refs: List[Tuple[int, int]] = []
        missing_tokens = 0
        missing_not_in_split = 0
        missing_insufficient_history = 0
        missing_insufficient_future = 0
        missing_other = 0
        for token in policy_tokens:
            ref = ref_by_token.get(str(token))
            if ref is None:
                missing_tokens += 1
                meta = token_anchor_meta.get(str(token))
                if meta is None:
                    missing_not_in_split += 1
                elif int(meta.anchor_idx) < int(meta.min_anchor):
                    missing_insufficient_history += 1
                elif int(meta.anchor_idx) > int(meta.max_anchor):
                    missing_insufficient_future += 1
                else:
                    missing_other += 1
                continue
            filtered_refs.append(ref)
        if rank == 0:
            print(
                "[infer][policy_filter]",
                f"anno={policy_path}",
                f"tokens={len(policy_tokens)}",
                f"dup_removed={dup_tokens}",
                f"matched={len(filtered_refs)}",
                f"missing={missing_tokens}",
                f"missing_not_in_split={missing_not_in_split}",
                f"missing_history={missing_insufficient_history}",
                f"missing_future={missing_insufficient_future}",
                f"missing_other={missing_other}",
                f"before={len(sample_refs)}",
            )
        sample_refs = filtered_refs

    if len(sample_refs) == 0:
        raise RuntimeError(
            "No valid nuScenes windows after filtering. "
            f"Need history={num_history} and >= {min_future_frames_for_eligibility} future frames "
            f"(metric_horizon_requires={min_future_frames_for_metrics}, "
            f"three_seconds={min_future_frames_for_3s} @ {scene_fps:.1f}Hz)."
        )

    all_indices = list(range(len(sample_refs)))
    done_tokens: set = set()
    resume_path = Path(args.resume_csv) if args.resume_csv else None
    if resume_path is not None:
        if resume_path.exists():
            import pandas as pd

            done_df = pd.read_csv(resume_path)
            done_tokens = set(done_df.get("token", []).tolist())
            done_tokens.discard("average")
        elif rank == 0:
            print(f"[infer] resume_csv not found, ignore: {resume_path}")

    def _window_rank_indices(target_rank: int) -> List[int]:
        indices = all_indices[target_rank::world_size]
        if args.token_offset > 0:
            indices = indices[args.token_offset:]
        if args.max_eval_tokens is not None:
            indices = indices[: max(args.max_eval_tokens, 0)]
        return indices

    def _drop_done_tokens(indices: List[int]) -> List[int]:
        if not done_tokens:
            return indices
        filtered: List[int] = []
        for idx in indices:
            scene_idx, anchor_idx = sample_refs[idx]
            token = scene_caches[scene_idx].token_states[anchor_idx].token
            if token in done_tokens:
                continue
            filtered.append(idx)
        return filtered

    local_indices_pre = _window_rank_indices(rank)
    local_indices = _drop_done_tokens(local_indices_pre)
    if rank == 0 and resume_path is not None and done_tokens:
        print(
            f"[infer] resume skip: {len(local_indices_pre) - len(local_indices)} samples already in {resume_path}"
        )

    global_viz_indices: set = set()
    viz_index_by_sample: Dict[int, int] = {}
    global_viz_order: List[int] = []
    if args.save_viz:
        global_eval_indices: List[int] = []
        for target_rank in range(world_size):
            rank_indices = _drop_done_tokens(_window_rank_indices(target_rank))
            global_eval_indices.extend(rank_indices)
        global_eval_indices = sorted(global_eval_indices)
        global_cap = max(0, int(args.viz_max_tokens))
        global_viz_order = global_eval_indices[:global_cap]
        global_viz_indices = set(global_viz_order)
        for i, sample_idx in enumerate(global_viz_order, start=1):
            viz_index_by_sample[int(sample_idx)] = i
        if rank == 0:
            print(
                f"[infer][viz] global cap={global_cap} selected={len(global_viz_indices)} "
                f"from eval_indices={len(global_eval_indices)}"
            )

    viz_dir = None
    if args.save_viz:
        viz_dir = Path(args.viz_dir) if args.viz_dir is not None else (Path(args.output_dir) / "viz")
        viz_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(
            f"[infer][nuscenes] scenes={len(scene_caches)} windows={len(sample_refs)} "
            f"local={len(local_indices)} world_size={world_size} "
            f"history={num_history} future={num_future} model_future={args.model_future_frames} "
            f"min_future_for_eligibility={min_future_frames_for_eligibility} "
            f"min_future_for_metrics={min_future_frames_for_metrics} "
            f"min_future_for_3s={min_future_frames_for_3s} "
            f"horizons={','.join(_format_horizon_label(h) for h in horizons_s)}"
        )

    results: List[Dict[str, Any]] = []
    viz_saved = 0
    printed_prompt_steps = 0
    start_t = time.perf_counter()

    if args.show_eval_progress and rank == 0:
        iterator = tqdm(local_indices, desc="Eval-nuScenes", total=len(local_indices), dynamic_ncols=True)
    else:
        iterator = local_indices

    for sample_idx in iterator:
        row: Dict[str, Any] = {"dataset_idx": int(sample_idx), "valid": True, "rank": rank}
        try:
            scene_idx, anchor_idx = sample_refs[sample_idx]
            scene_cache = scene_caches[scene_idx]
            states = scene_cache.token_states

            hist_start = anchor_idx - num_history + 1
            hist_end = anchor_idx + 1
            fut_start = anchor_idx + 1
            fut_end = anchor_idx + 1 + num_future
            history_states = states[hist_start:hist_end]
            future_states = states[fut_start:fut_end]

            token = str(states[anchor_idx].token)
            row["token"] = token
            row["scene"] = str(scene_cache.scene_name)
            if args.print_tokens:
                print(token, flush=True)

            history_video: List[Image.Image] = []
            history_processed_intrinsics: List[Optional[np.ndarray]] = []
            for s in history_states:
                image, processed_intrinsics = _load_camera_rgb(
                    s,
                    width=args.width,
                    height=args.height,
                    focal_config=focal_config,
                )
                history_video.append(image)
                history_processed_intrinsics.append(processed_intrinsics)
            gt_future_video: List[Image.Image] = []
            gt_video: List[Image.Image] = []
            if args.save_viz:
                for s in future_states:
                    image, _ = _load_camera_rgb(
                        s,
                        width=args.width,
                        height=args.height,
                        focal_config=focal_config,
                    )
                    gt_future_video.append(image)
            if args.save_viz:
                gt_video = history_video + gt_future_video

            anchor = states[anchor_idx]
            anchor_xyh = np.array([anchor.ego_x, anchor.ego_y, anchor.ego_yaw], dtype=np.float64)
            history_global_xyh = np.asarray(
                [[s.ego_x, s.ego_y, s.ego_yaw] for s in history_states], dtype=np.float64
            )
            future_global_xyh = np.asarray(
                [[s.ego_x, s.ego_y, s.ego_yaw] for s in future_states], dtype=np.float64
            )
            # Anchor all trajectory labels at the current sample so metrics and
            # model outputs share the same ego-local coordinate frame.
            history_local_xyh_full = _build_local_xyh(history_global_xyh, anchor_xyh)
            # Keep history_positions aligned with the UNIVERSE feature builder:
            # it uses ego_statuses[:4] from history trajectory.
            history_local_xyh = history_local_xyh_full[: min(4, history_local_xyh_full.shape[0])]
            future_local_xyh = _build_local_xyh(future_global_xyh, anchor_xyh)

            if anchor_idx > 0:
                prev = states[anchor_idx - 1]
                dt = max(anchor.timestamp_s - prev.timestamp_s, 1e-3)
                vel_global = np.array(
                    [(anchor.ego_x - prev.ego_x) / dt, (anchor.ego_y - prev.ego_y) / dt], dtype=np.float64
                )
            elif anchor_idx + 1 < len(states):
                nxt = states[anchor_idx + 1]
                dt = max(nxt.timestamp_s - anchor.timestamp_s, 1e-3)
                vel_global = np.array(
                    [(nxt.ego_x - anchor.ego_x) / dt, (nxt.ego_y - anchor.ego_y) / dt], dtype=np.float64
                )
            else:
                vel_global = np.zeros(2, dtype=np.float64)
            c = math.cos(-anchor.ego_yaw)
            s = math.sin(-anchor.ego_yaw)
            vx_local = c * vel_global[0] - s * vel_global[1]
            vy_local = s * vel_global[0] + c * vel_global[1]
            ego_vel = np.array([vx_local, vy_local, 0.0], dtype=np.float32)
            speed_mps = float(np.linalg.norm(ego_vel[:2]))

            driving_command = _cmd_from_future_yaw(
                future_local_xyh, yaw_thresh_deg=float(args.command_yaw_threshold_deg)
            ).astype(np.float32)
            prompt = _build_prompt_fixed(torch.from_numpy(driving_command), speed_mps)
            if rank == 0 and printed_prompt_steps < max(0, int(args.debug_prompt_steps)):
                n = printed_prompt_steps + 1
                print(f"[infer][prompt][{n}/{args.debug_prompt_steps}] positive: {prompt}")
                print(f"[infer][prompt][{n}/{args.debug_prompt_steps}] negative: {args.negative_prompt}")
                printed_prompt_steps += 1

            total_frames = int(num_history + int(args.model_future_frames))
            history_cond = (
                history_local_xyh
                if str(args.trajectory_condition_mode).lower() in {"auto", "history"}
                else None
            )
            # The same call signature is used by all three infer_all datasets:
            # history video is the visual condition; velocity/history is the
            # trajectory prefix; trajectory_len controls future points only.
            with torch.no_grad():
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
                    ego_vel=ego_vel,
                    history_positions=history_cond,
                    seed=args.seed + int(sample_idx),
                    rand_device=str(device),
                    tiled=True,
                    progress_bar_cmd=tqdm if args.show_denoise_progress else _no_progress_bar,
                    output_mode=infer_output_mode,
                )

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
                raise ValueError("Pipeline video-only output has no trajectory for nuScenes metrics.")

            pred_traj = traj_pred[0].detach().float().cpu().numpy()
            pred_traj = _safe_traj_np(pred_traj, int(args.model_future_frames))

            pred_times_s = (
                np.arange(1, pred_traj.shape[0] + 1, dtype=np.float64) / float(args.target_fps)
            )
            future_times_s = np.asarray(
                [float(s.timestamp_s - anchor.timestamp_s) for s in future_states], dtype=np.float64
            )
            if future_times_s.ndim != 1 or future_times_s.shape[0] == 0:
                raise ValueError("invalid future times for nuScenes sample")

            bev_occupancy_old = _build_bev_occupancy_from_future_states(
                future_states,
                anchor_x=float(anchor.ego_x),
                anchor_y=float(anchor.ego_y),
                anchor_yaw=float(anchor.ego_yaw),
                bev_res_m=_BEV_LEGACY_RES_M,
            )
            pred_step_xy_local = np.full((future_times_s.shape[0], 2), np.nan, dtype=np.float32)
            for step_idx, step_time_s in enumerate(future_times_s.tolist()):
                pred_xyh_step = _interp_xyh_at_horizon(pred_traj, pred_times_s, float(step_time_s))
                if pred_xyh_step is None:
                    continue
                pred_step_xy_local[step_idx, :] = pred_xyh_step[:2].astype(np.float32)
            gt_step_xy_local = np.asarray(future_local_xyh[..., :2], dtype=np.float32)
            pred_step_collision_old = _compute_policy_bev_collision_flags(
                pred_step_xy_local,
                bev_occupancy_old,
                ego_length_m=float(args.ego_box_length_m),
                ego_width_m=float(args.ego_box_width_m),
                bev_res_m=_BEV_LEGACY_RES_M,
                ego_front_shift_m=_LEGACY_EGO_FRONT_SHIFT_M,
            )
            gt_step_collision_old = _compute_policy_bev_collision_flags(
                gt_step_xy_local,
                bev_occupancy_old,
                ego_length_m=float(args.ego_box_length_m),
                ego_width_m=float(args.ego_box_width_m),
                bev_res_m=_BEV_LEGACY_RES_M,
                ego_front_shift_m=_LEGACY_EGO_FRONT_SHIFT_M,
            )

            for horizon_s in horizons_s:
                label = _format_horizon_label(horizon_s)
                l2_key = f"l2_{label}_m"
                l2_vad_key = f"l2_vad_{label}_m"
                collision_old_key = f"collision_old_{label}_pct"

                pred_xyh_h = _interp_xyh_at_horizon(pred_traj, pred_times_s, float(horizon_s))
                gt_xyh_h = _interp_xyh_at_horizon(future_local_xyh.astype(np.float32), future_times_s, float(horizon_s))
                if pred_xyh_h is None or gt_xyh_h is None:
                    row[l2_key] = float("nan")
                    row[l2_vad_key] = float("nan")
                    row[collision_old_key] = float("nan")
                    continue

                pred_xy = pred_xyh_h[:2].astype(np.float32)
                gt_xy = gt_xyh_h[:2].astype(np.float32)
                l2_val = np.asarray(final_distances_fn(np.asarray([[pred_xy]]), np.asarray([[gt_xy]]))).reshape(-1)
                row[l2_key] = float(l2_val[0]) if l2_val.size > 0 else float("nan")
                row[l2_vad_key] = _compute_l2_vad_with_horizon(
                    pred_step_xy_local=pred_step_xy_local,
                    gt_step_xy_local=gt_step_xy_local,
                    future_times_s=future_times_s,
                    horizon_s=float(horizon_s),
                )
                row[collision_old_key] = _compute_collision_percent_with_horizon(
                    pred_step_collision=pred_step_collision_old,
                    gt_step_collision=gt_step_collision_old,
                    future_times_s=future_times_s,
                    horizon_s=float(horizon_s),
                )

            gt_cmd_text = one_hot_to_cmd(driving_command)
            pred_cmd_onehot = _cmd_from_future_yaw(
                pred_traj, yaw_thresh_deg=float(args.command_yaw_threshold_deg)
            )
            pred_cmd_text = one_hot_to_cmd(pred_cmd_onehot)
            row["cmd_gt"] = gt_cmd_text
            row["cmd_pred"] = pred_cmd_text
            row["cmd_match"] = bool(pred_cmd_text == gt_cmd_text)

            row["hist_frames_gt"] = int(len(history_video))
            row["future_frames_gt"] = int(len(future_states))
            row["pred_frames"] = int(len(pred_video) if isinstance(pred_video, list) else 0)
            row["hist_frames_ok"] = bool(len(history_video) == num_history)
            row["future_frames_ok"] = bool(len(future_states) >= int(min_future_frames_for_eligibility))
            row["pred_frames_ok"] = bool(row["pred_frames"] == total_frames)
            row["traj_len_gt"] = int(future_local_xyh.shape[0])
            row["traj_len_pred"] = int(pred_traj.shape[0])
            row["speed_mps"] = float(speed_mps)

            if args.save_viz and sample_idx in global_viz_indices:
                try:
                    projection_image_path = None
                    camera_for_projection = None
                    if bool(args.save_projected_traj_image):
                        anchor_processed_intrinsics = (
                            history_processed_intrinsics[-1] if len(history_processed_intrinsics) > 0 else None
                        )
                        camera_for_projection = _build_nuscenes_camera_for_projection(
                            anchor,
                            intrinsics_override=anchor_processed_intrinsics,
                        )
                    extra_info = {
                        "nav_cmd": f"gt={gt_cmd_text}, pred={pred_cmd_text}",
                        "speed_mps": row["speed_mps"],
                        "yaw_deg": float(np.degrees(float(history_local_xyh[-1, 2]))) if history_local_xyh.shape[0] > 0 else 0.0,
                    }
                    viz_index = viz_index_by_sample.get(int(sample_idx))
                    viz_index_prefix = format_viz_index_prefix(viz_index)
                    viz_path = viz_dir / f"{viz_index_prefix}{token}_rank{rank}.mp4"
                    if viz_index is not None:
                        row["viz_index"] = int(viz_index)
                    if camera_for_projection is not None:
                        projection_dir = viz_dir / "projected_traj"
                        projection_dir.mkdir(parents=True, exist_ok=True)
                        projection_image_path = projection_dir / f"{viz_index_prefix}{token}_rank{rank}.png"
                    score_dict = {
                        "cmd_match": float(row["cmd_match"]),
                    }
                    for horizon_s in horizons_s:
                        label = _format_horizon_label(horizon_s)
                        l2_key = f"l2_{label}_m"
                        l2_vad_key = f"l2_vad_{label}_m"
                        collision_old_key = f"collision_old_{label}_pct"
                        if l2_key in row:
                            score_dict[l2_key] = row[l2_key]
                        if l2_vad_key in row:
                            score_dict[l2_vad_key] = row[l2_vad_key]
                        if collision_old_key in row:
                            score_dict[collision_old_key] = row[collision_old_key]
                    save_viz_video(
                        out_path=viz_path,
                        gt_video=gt_video,
                        pred_video=pred_video,
                        history_traj=history_local_xyh,
                        gt_future_traj=future_local_xyh,
                        pred_future_traj=pred_traj,
                        width=args.width,
                        height=args.height,
                        plot_height=DEFAULT_VIZ_PLOT_HEIGHT,
                        fps=int(args.target_fps),
                        score_dict=score_dict,
                        extra_info=extra_info,
                        camera_for_projection=camera_for_projection,
                        num_history_frames=int(args.num_history_frames),
                        gt_future_max_steps=(
                            int(args.model_future_frames)
                            if float(getattr(args, "scene_future_extra_seconds", 0.0)) > 0.0
                            else None
                        ),
                        projection_image_path=projection_image_path,
                        projection_overlay_on_video=False,
                    )
                    row["viz_path"] = str(viz_path)
                    if projection_image_path is not None:
                        row["viz_projection_path"] = str(projection_image_path)
                    viz_saved += 1
                except Exception as viz_exc:
                    row["viz_error"] = str(viz_exc)
                    if args.print_errors:
                        print(f"[infer][viz_error] token={token} err={viz_exc}", flush=True)

        except Exception as exc:
            row["valid"] = False
            row["error"] = str(exc)
            if args.print_errors:
                print(f"[infer][error] idx={sample_idx} err={exc}", flush=True)
        results.append(row)

    if args.show_eval_progress and rank == 0 and hasattr(iterator, "close"):
        iterator.close()

    final_results = _gather_results(results, device=device, rank=rank, world_size=world_size)
    if rank == 0 and final_results is not None:
        import pandas as pd

        elapsed = time.perf_counter() - start_t
        os.makedirs(args.output_dir, exist_ok=True)
        df = pd.DataFrame(final_results)
        num_success = int(df["valid"].sum()) if "valid" in df.columns else 0
        num_failed = len(df) - num_success

        metric_df = df.drop(
            columns=["token", "scene", "valid", "rank", "error", "cmd_gt", "cmd_pred", "viz_path", "viz_error"],
            errors="ignore",
        )
        metric_df = metric_df.select_dtypes(include=[np.number, "bool"])
        avg_row = metric_df.mean(skipna=True)
        avg_row["token"] = "average"
        avg_row["valid"] = bool(df["valid"].all()) if "valid" in df.columns else False
        avg_row["rank"] = "0"
        df.loc[len(df)] = avg_row

        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        out_path = Path(args.output_dir) / f"nuscenes_infer_{timestamp}.csv"
        df.to_csv(out_path, index=False)
        def _summary_dict_from_series(metric_series: "pd.Series") -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for name in metric_df.columns:
                value = metric_series.get(name, np.nan)
                if pd.isna(value):
                    continue
                if isinstance(value, (bool, np.bool_)):
                    out[str(name)] = bool(value)
                else:
                    out[str(name)] = float(value)
            return out

        summary_metrics = {"all": _summary_dict_from_series(avg_row)}
        summary_out_path = Path(args.output_dir) / f"nuscenes_infer_{timestamp}.summary.json"
        with summary_out_path.open("w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, ensure_ascii=False, indent=2, sort_keys=True)

        print(
            f"[infer] elapsed={elapsed:.2f}s scenes={len(df) - 1} "
            f"avg_per_scene={elapsed / max(1, len(df) - 1):.2f}s"
        )
        if len(metric_df.columns) > 0:
            print("[infer] average metrics:")
            for name in metric_df.columns:
                value = avg_row.get(name, np.nan)
                if pd.isna(value):
                    continue
                print(f"[infer][avg] {name}={float(value):.6f}")

        l2_parts: List[str] = []
        l2_vad_parts: List[str] = []
        collision_old_parts: List[str] = []
        for horizon_s in horizons_s:
            label = _format_horizon_label(horizon_s)
            l2_key = f"l2_{label}_m"
            l2_vad_key = f"l2_vad_{label}_m"
            col_old_key = f"collision_old_{label}_pct"
            l2_val = avg_row.get(l2_key, np.nan)
            l2_vad_val = avg_row.get(l2_vad_key, np.nan)
            col_old_val = avg_row.get(col_old_key, np.nan)
            if not pd.isna(l2_val):
                l2_parts.append(f"{label}={float(l2_val):.4f}")
            if not pd.isna(l2_vad_val):
                l2_vad_parts.append(f"{label}={float(l2_vad_val):.4f}")
            if not pd.isna(col_old_val):
                collision_old_parts.append(f"{label}={float(col_old_val):.2f}")
        if l2_parts or l2_vad_parts:
            left = ", ".join(l2_parts) if l2_parts else "none"
            right = ", ".join(l2_vad_parts) if l2_vad_parts else "none"
            print(f"[infer][nuscenes][avg] L2 (m): fde=[{left}] | vad=[{right}]")
        if collision_old_parts:
            print("[infer][nuscenes][avg] Collision-old (%): " + ", ".join(collision_old_parts))

        print(f"[infer] Done. success={num_success} failed={num_failed} saved={out_path}")


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
