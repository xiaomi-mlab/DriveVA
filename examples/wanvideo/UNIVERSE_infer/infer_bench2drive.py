"""
Bench2Drive inference/evaluation script for Wan video pipeline.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from examples.wanvideo.driveva_infer.b2d_planning_metrics import (
    B2DPlanningMetricLite as _MigratedB2DPlanningMetricLite,
    compute_planning_metrics_for_scene as _migrated_compute_planning_metrics_for_scene,
)
from examples.wanvideo.driveva_infer.navsim_eval_viz import format_viz_index_prefix, save_viz_video

try:
    from .bench2drive_dataset import Bench2DriveEvalDataset
    from .navsim_dataset import DEFAULT_NEGATIVE_PROMPT, one_hot_to_cmd, _build_prompt_fixed
except ImportError:
    from bench2drive_dataset import Bench2DriveEvalDataset
    from navsim_dataset import DEFAULT_NEGATIVE_PROMPT, one_hot_to_cmd, _build_prompt_fixed


DEFAULT_VIZ_PLOT_HEIGHT = 420
_R_CUR_TO_VEHICLE = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
# Camera optical(CV) -> Unreal/Carla camera frame (row-vector convention).
_R_CAMCV_TO_CAMUE = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_R_CAMUE_TO_CAMCV = _R_CAMCV_TO_CAMUE.T.astype(np.float32)

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer/evaluate Wan video model on Bench2Drive.")

    # Data args
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ann_file", type=str, required=True)
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=None,
        help="Optional maximum number of samples to evaluate. Default: all samples.",
    )
    parser.add_argument("--dataset_stride", type=int, default=1)
    parser.add_argument("--target_fps", type=int, default=2)
    parser.add_argument("--original_fps", type=int, default=10)
    parser.add_argument("--frame_interval", type=int, default=None, help="Explicit frame interval override.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable torch.distributed inference. Default is single-process.",
    )
    parser.add_argument("--use_structured_prompt", action="store_true")

    parser.add_argument("--show_eval_progress", dest="show_eval_progress", action="store_true")
    parser.add_argument("--no_show_eval_progress", dest="show_eval_progress", action="store_false")
    parser.set_defaults(show_eval_progress=True)

    # Video args
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--use_focal_unification", action="store_true")
    parser.add_argument("--focal_target_px", type=str, default="")
    parser.add_argument("--num_history_frames", type=int, default=5)
    parser.add_argument("--num_future_frames", type=int, default=8)
    parser.add_argument("--command_yaw_threshold_deg", type=float, default=8.0)
    parser.add_argument(
        "--model_future_frames",
        type=int,
        default=None,
        help="Future trajectory points produced by model. Default: num_future_frames.",
    )
    parser.add_argument("--debug_prompt_steps", type=int, default=10)
    parser.add_argument("--show_denoise_progress", action="store_true")
    parser.add_argument("--show_vae_progress", action="store_true")

    # Model args
    parser.add_argument("--local_model_path", type=str, default=None)
    parser.add_argument("--full_ckpt", type=str, default=None, help="Path to full DiT checkpoint (.safetensors).")
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--use_mixed_latent_attention_mask", dest="use_mixed_latent_attention_mask", action="store_true")
    parser.add_argument("--no_use_mixed_latent_attention_mask", dest="use_mixed_latent_attention_mask", action="store_false")
    parser.add_argument("--infer_trajectory_only", dest="infer_trajectory_only", action="store_true")
    parser.add_argument("--no_infer_trajectory_only", dest="infer_trajectory_only", action="store_false")
    parser.add_argument("--infer_output_mode", type=str, default=None, choices=["trajectory", "video", "both"])
    parser.set_defaults(use_mixed_latent_attention_mask=True, infer_trajectory_only=False)

    # Output args
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_viz", dest="save_viz", action="store_true")
    parser.add_argument("--no_save_viz", dest="save_viz", action="store_false")
    parser.set_defaults(save_viz=True)
    parser.add_argument("--viz_max_scenes", type=int, default=30)
    parser.add_argument("--viz_plot_height", type=int, default=DEFAULT_VIZ_PLOT_HEIGHT)
    parser.add_argument(
        "--save_projected_traj_image",
        dest="save_projected_traj_image",
        action="store_true",
        help="Save current-frame trajectory projection image under <output_dir>/viz/projected_traj.",
    )
    parser.add_argument(
        "--no_save_projected_traj_image",
        dest="save_projected_traj_image",
        action="store_false",
        help="Disable current-frame trajectory projection image export.",
    )
    parser.set_defaults(save_projected_traj_image=True)
    parser.add_argument("--print_errors", dest="print_errors", action="store_true")
    parser.add_argument("--no_print_errors", dest="print_errors", action="store_false")
    parser.set_defaults(print_errors=True)
    parser.add_argument(
        "--projection_debug_steps",
        type=int,
        default=0,
        help="Print projection-candidate debug for first N visualized samples on rank0.",
    )
    parser.add_argument(
        "--projection_debug_token",
        type=str,
        default="",
        help="If set, dump full local/lidar/cam/img projection chain for this exact token.",
    )
    parser.add_argument(
        "--projection_debug_max_points",
        type=int,
        default=64,
        help="Maximum points per polyline in --projection_debug_token dump.",
    )
    parser.add_argument(
        "--compute_planning_metrics",
        dest="compute_planning_metrics",
        action="store_true",
        help="Compute Bench2Drive planning metrics (L2/collision) when annotation fields are available.",
    )
    parser.add_argument(
        "--no_compute_planning_metrics",
        dest="compute_planning_metrics",
        action="store_false",
        help="Disable Bench2Drive planning metrics.",
    )
    parser.set_defaults(compute_planning_metrics=True)

    return parser.parse_args(argv)

def _init_distributed(enable_distributed: bool) -> Dict[str, int]:
    if not enable_distributed:
        return {"rank": 0, "local_rank": 0, "world_size": 1}

    if torch.distributed.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        if not torch.distributed.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend)
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


def _normalize_train_ckpt_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


def _fit_len_2d(arr: np.ndarray, expected_len: int, channels: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.zeros((expected_len, channels), dtype=np.float32)
    if arr.shape[1] < channels:
        pad = np.zeros((arr.shape[0], channels - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    if arr.shape[1] > channels:
        arr = arr[:, :channels]
    if arr.shape[0] == expected_len:
        return arr.astype(np.float32)
    if arr.shape[0] > expected_len:
        return arr[:expected_len].astype(np.float32)
    if arr.shape[0] == 0:
        return np.zeros((expected_len, channels), dtype=np.float32)
    pad = np.repeat(arr[-1:], expected_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)


def _quat_xyzw_to_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] < 4:
        return np.zeros((q.shape[0] if q.ndim == 2 else 0,), dtype=np.float32)
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    w = q[:, 3]
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return yaw.astype(np.float32)


def _pose_to_xyh(positions: np.ndarray, quaternions: np.ndarray, expected_len: int) -> np.ndarray:
    pos = _fit_len_2d(positions, expected_len=expected_len, channels=3)
    quat = _fit_len_2d(quaternions, expected_len=expected_len, channels=4)
    yaw = _quat_xyzw_to_yaw(quat)
    return np.stack([pos[:, 0], pos[:, 1], yaw], axis=1).astype(np.float32)


def _quat_xyzw_to_rotmat_single(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    if q.size < 4:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = [float(v) for v in q[:4]]
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
        ],
        dtype=np.float32,
    )


def _scale_intrinsics(
    intrinsic: np.ndarray,
    *,
    src_h: int,
    src_w: int,
    dst_h: int,
    dst_w: int,
) -> np.ndarray:
    kk = np.asarray(intrinsic, dtype=np.float32).copy()
    if kk.ndim != 2 or kk.shape[0] < 3 or kk.shape[1] < 3:
        return kk
    src_h = max(1, int(src_h))
    src_w = max(1, int(src_w))
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    kk[0, 0] *= sx
    kk[0, 2] *= sx
    kk[1, 1] *= sy
    kk[1, 2] *= sy
    return kk


def _project_local_xy_valid_mask(
    local_xy: np.ndarray,
    camera_for_projection: Dict[str, Any],
    *,
    image_h: int,
    image_w: int,
    eps: float = 1e-3,
) -> np.ndarray:
    xy = np.asarray(local_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] <= 0 or xy.shape[1] < 2:
        return np.zeros((0,), dtype=np.bool_)
    rr = np.asarray(camera_for_projection.get("sensor2lidar_rotation"), dtype=np.float32)
    tt = np.asarray(camera_for_projection.get("sensor2lidar_translation"), dtype=np.float32).reshape(-1)
    kk = np.asarray(camera_for_projection.get("intrinsics"), dtype=np.float32)
    if rr.shape != (3, 3) or tt.size < 3 or kk.ndim != 2 or kk.shape[0] < 3 or kk.shape[1] < 3:
        return np.zeros((xy.shape[0],), dtype=np.bool_)

    plane_z = float(camera_for_projection.get("project_plane_z_m", 0.0))
    local_to_lidar = np.asarray(
        camera_for_projection.get("local_to_lidar_rotation", np.eye(3, dtype=np.float32)),
        dtype=np.float32,
    )
    points_local = np.concatenate(
        [xy[:, :2].astype(np.float32, copy=False), np.zeros((xy.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    if local_to_lidar.shape == (3, 3):
        # Row-vector conversion: p_lidar = p_local @ R_local_to_lidar^T
        points_lidar = points_local @ local_to_lidar.T
    else:
        points_lidar = points_local
    points_lidar[:, 2] = float(plane_z)
    points_cam = (points_lidar - tt[None, :3]) @ rr
    points_img_h = points_cam @ kk.T
    depth = points_img_h[:, 2]
    safe_depth = np.maximum(depth, float(eps))
    points_img = points_img_h[:, :2] / safe_depth[:, None]
    in_image = (
        (points_img[:, 0] >= 0.0)
        & (points_img[:, 0] <= max(0, int(image_w) - 1))
        & (points_img[:, 1] >= 0.0)
        & (points_img[:, 1] <= max(0, int(image_h) - 1))
    )
    finite = np.isfinite(points_img).all(axis=1)
    return (depth > float(eps)) & in_image & finite


def _project_local_xy_points_and_valid(
    local_xy: np.ndarray,
    camera_for_projection: Dict[str, Any],
    *,
    image_h: int,
    image_w: int,
    eps: float = 1e-3,
    allow_behind_camera: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(local_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] <= 0 or xy.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.bool_)
    rr = np.asarray(camera_for_projection.get("sensor2lidar_rotation"), dtype=np.float32)
    tt = np.asarray(camera_for_projection.get("sensor2lidar_translation"), dtype=np.float32).reshape(-1)
    kk = np.asarray(camera_for_projection.get("intrinsics"), dtype=np.float32)
    if rr.shape != (3, 3) or tt.size < 3 or kk.ndim != 2 or kk.shape[0] < 3 or kk.shape[1] < 3:
        return np.zeros((xy.shape[0], 2), dtype=np.float32), np.zeros((xy.shape[0],), dtype=np.bool_)
    plane_z = float(camera_for_projection.get("project_plane_z_m", 0.0))
    local_to_lidar = np.asarray(
        camera_for_projection.get("local_to_lidar_rotation", np.eye(3, dtype=np.float32)),
        dtype=np.float32,
    )
    points_local = np.concatenate(
        [xy[:, :2].astype(np.float32, copy=False), np.zeros((xy.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    if local_to_lidar.shape == (3, 3):
        # Row-vector conversion: p_lidar = p_local @ R_local_to_lidar^T
        points_lidar = points_local @ local_to_lidar.T
    else:
        points_lidar = points_local
    points_lidar[:, 2] = float(plane_z)
    points_cam = (points_lidar - tt[None, :3]) @ rr
    points_img_h = points_cam @ kk.T
    depth = points_img_h[:, 2]
    if allow_behind_camera:
        abs_depth = np.abs(depth)
        safe_depth = np.where(
            abs_depth > float(eps),
            depth,
            np.where(depth >= 0.0, float(eps), -float(eps)),
        )
        valid_depth = abs_depth > float(eps)
    else:
        safe_depth = np.maximum(depth, float(eps))
        valid_depth = depth > float(eps)
    points_img = points_img_h[:, :2] / safe_depth[:, None]
    in_image = (
        (points_img[:, 0] >= 0.0)
        & (points_img[:, 0] <= max(0, int(image_w) - 1))
        & (points_img[:, 1] >= 0.0)
        & (points_img[:, 1] <= max(0, int(image_h) - 1))
    )
    finite = np.isfinite(points_img).all(axis=1)
    valid = valid_depth & in_image & finite
    return points_img.astype(np.float32, copy=False), valid.astype(np.bool_, copy=False)


def _camera_candidate_score(
    camera_for_projection: Dict[str, Any],
    *,
    image_h: int,
    image_w: int,
    history_traj: np.ndarray,
    gt_future_traj: np.ndarray,
) -> int:
    anchor = np.zeros((1, 2), dtype=np.float32)
    gt_xy = np.asarray(gt_future_traj, dtype=np.float32)
    hist_xy = np.asarray(history_traj, dtype=np.float32)
    gt_line = anchor
    hist_line = anchor
    if gt_xy.ndim == 2 and gt_xy.shape[0] > 0 and gt_xy.shape[1] >= 2:
        gt_line = np.concatenate([anchor, gt_xy[:, :2]], axis=0)
    if hist_xy.ndim == 2 and hist_xy.shape[0] > 0 and hist_xy.shape[1] >= 2:
        hist_line = np.concatenate([hist_xy[:, :2], anchor], axis=0)
    gt_valid = _project_local_xy_valid_mask(
        gt_line,
        camera_for_projection,
        image_h=image_h,
        image_w=image_w,
    )
    hist_valid = _project_local_xy_valid_mask(
        hist_line,
        camera_for_projection,
        image_h=image_h,
        image_w=image_w,
    )
    gt_score = int(np.count_nonzero(gt_valid))
    hist_score = int(np.count_nonzero(hist_valid))
    anchor_points, anchor_valid = _project_local_xy_points_and_valid(
        anchor,
        camera_for_projection,
        image_h=image_h,
        image_w=image_w,
        allow_behind_camera=True,
    )
    anchor_bonus = 0
    anchor_x = float("nan")
    anchor_y = float("nan")
    if anchor_points.shape[0] > 0 and np.isfinite(anchor_points[0]).all():
        x = float(anchor_points[0, 0])
        y = float(anchor_points[0, 1])
        x = float(np.clip(x, 0.0, max(0, int(image_w) - 1)))
        y = float(np.clip(y, 0.0, max(0, int(image_h) - 1)))
        anchor_x = x
        anchor_y = y
        target_x = 0.5 * float(image_w)
        target_y = 0.82 * float(image_h)
        dx = abs(x - target_x) / max(1.0, float(image_w))
        dy = abs(y - target_y) / max(1.0, float(image_h))
        anchor_bonus = int(round(120.0 * max(0.0, 1.0 - dx - dy)))
        if y < 0.55 * float(image_h):
            anchor_bonus -= 180
        if not bool(anchor_valid[0]):
            anchor_bonus -= 20
    else:
        anchor_bonus -= 40

    # Forward-direction sanity in front camera:
    # points ahead should generally project above (smaller v) the ego anchor.
    forward_bonus = 0
    if np.isfinite(anchor_x) and np.isfinite(anchor_y):
        probe = np.asarray([[2.0, 0.0], [5.0, 0.0], [8.0, 0.0]], dtype=np.float32)
        probe_pts, probe_valid = _project_local_xy_points_and_valid(
            probe,
            camera_for_projection,
            image_h=image_h,
            image_w=image_w,
            allow_behind_camera=False,
        )
        if probe_pts.shape[0] >= 2 and probe_valid.shape[0] >= 2:
            if bool(probe_valid[0]) and np.isfinite(probe_pts[0]).all():
                if float(probe_pts[0, 1]) < (anchor_y - 3.0):
                    forward_bonus += 70
                else:
                    forward_bonus -= 90
            if bool(probe_valid[1]) and np.isfinite(probe_pts[1]).all():
                if float(probe_pts[1, 1]) < (anchor_y - 8.0):
                    forward_bonus += 90
                else:
                    forward_bonus -= 120
            if (
                bool(probe_valid[0])
                and bool(probe_valid[1])
                and np.isfinite(probe_pts[0]).all()
                and np.isfinite(probe_pts[1]).all()
            ):
                if float(probe_pts[1, 1]) < float(probe_pts[0, 1]) - 2.0:
                    forward_bonus += 60
                else:
                    forward_bonus -= 80
        else:
            forward_bonus -= 40

    return gt_score * 12 + hist_score * 2 + anchor_bonus + forward_bonus


def _build_ego2world_from_sample(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    hist_pos = np.asarray(sample.get("gt_history_positions_global"), dtype=np.float32)
    hist_quat = np.asarray(sample.get("gt_history_quaternions_global"), dtype=np.float32)
    if hist_pos.ndim != 2 or hist_quat.ndim != 2:
        return None
    n = min(int(hist_pos.shape[0]), int(hist_quat.shape[0]))
    if n <= 0:
        return None
    pos = hist_pos[n - 1]
    quat = hist_quat[n - 1]
    if pos.shape[0] < 3 or quat.shape[0] < 4:
        return None
    rot = _quat_xyzw_to_rotmat_single(quat[:4])
    ego2world = np.eye(4, dtype=np.float32)
    ego2world[:3, :3] = rot
    ego2world[:3, 3] = pos[:3].astype(np.float32)
    return ego2world


def _build_sensor2local_from_world2cam(
    *,
    world2cam: np.ndarray,
    ego2world: np.ndarray,
    local_to_cur_rot: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    try:
        w2c = np.asarray(world2cam, dtype=np.float64)
        e2w = np.asarray(ego2world, dtype=np.float64)
        l2c = np.asarray(local_to_cur_rot, dtype=np.float64)
    except Exception:
        return None
    if w2c.shape != (4, 4) or e2w.shape != (4, 4) or l2c.shape != (3, 3):
        return None

    r_wc = w2c[:3, :3]
    t_wc = w2c[:3, 3]
    r_w_cur = e2w[:3, :3]
    t_w_cur = e2w[:3, 3]

    # local -> world : R_wl = R_w_cur * R_cur_local
    r_w_local = r_w_cur @ l2c
    a_col = r_wc @ r_w_local
    if not np.isfinite(a_col).all():
        return None

    # Re-orthonormalize for numerical stability.
    try:
        u, _, vh = np.linalg.svd(a_col)
        a_col = u @ vh
        if np.linalg.det(a_col) < 0.0:
            u[:, -1] *= -1.0
            a_col = u @ vh
    except Exception:
        return None

    b_col = r_wc @ t_w_cur + t_wc
    rr = a_col.T.astype(np.float32)
    tt = (-rr.astype(np.float64) @ b_col).astype(np.float32)
    if not np.isfinite(rr).all() or not np.isfinite(tt).all():
        return None
    return rr, tt


def _build_b2d_camera_for_projection(
    sample: Dict[str, Any],
    *,
    image_h: int,
    image_w: int,
    history_traj: np.ndarray,
    gt_future_traj: np.ndarray,
) -> Optional[Dict[str, Any]]:
    intrinsic_raw = sample.get("camera_intrinsic")
    camera_extrinsic_raw = sample.get("camera_extrinsic")
    if intrinsic_raw is None:
        return None

    cam2ego_raw = None
    world2cam_raw = None
    if isinstance(camera_extrinsic_raw, (list, tuple)):
        if len(camera_extrinsic_raw) >= 1:
            cam2ego_raw = camera_extrinsic_raw[0]
        if len(camera_extrinsic_raw) >= 2:
            world2cam_raw = camera_extrinsic_raw[1]
    elif isinstance(camera_extrinsic_raw, dict):
        cam2ego_raw = camera_extrinsic_raw.get("cam2ego")
        world2cam_raw = camera_extrinsic_raw.get("world2cam")
    elif camera_extrinsic_raw is not None:
        cam2ego_raw = camera_extrinsic_raw

    try:
        intrinsic = np.asarray(intrinsic_raw, dtype=np.float32)
    except Exception:
        return None
    if intrinsic.ndim != 2 or intrinsic.shape[0] < 3 or intrinsic.shape[1] < 3:
        return None

    src_w = int(image_w)
    src_h = int(image_h)
    original_size = sample.get("original_image_size")
    if isinstance(original_size, (list, tuple)) and len(original_size) >= 2:
        try:
            src_w = max(1, int(original_size[0]))
            src_h = max(1, int(original_size[1]))
        except Exception:
            src_w = int(image_w)
            src_h = int(image_h)
    intrinsic_scaled = _scale_intrinsics(
        intrinsic,
        src_h=src_h,
        src_w=src_w,
        dst_h=int(image_h),
        dst_w=int(image_w),
    )

    ego2world = _build_ego2world_from_sample(sample)

    cam2ego = None
    if cam2ego_raw is not None:
        try:
            cam2ego_arr = np.asarray(cam2ego_raw, dtype=np.float32)
            if cam2ego_arr.shape == (4, 4):
                cam2ego = cam2ego_arr
        except Exception:
            cam2ego = None

    cam2lidar = None
    source = "none"
    if world2cam_raw is not None and ego2world is not None:
        try:
            world2cam = np.asarray(world2cam_raw, dtype=np.float32)
            if world2cam.shape == (4, 4):
                # Annotation chain: lidar2cam = world2cam @ lidar2world
                lidar2cam = world2cam @ ego2world
                cam2lidar = np.linalg.inv(lidar2cam).astype(np.float32)
                source = "world2cam_lidar2world"
        except Exception:
            cam2lidar = None

    if cam2lidar is None and cam2ego is not None:
        # Dataloader fallback when lidar2ego is identity-like.
        cam2lidar = cam2ego.astype(np.float32)
        source = "cam2ego_direct"

    if cam2lidar is None:
        return None

    rr = np.asarray(cam2lidar[:3, :3], dtype=np.float32)
    tt = np.asarray(cam2lidar[:3, 3], dtype=np.float32)
    if rr.shape != (3, 3) or tt.shape[0] < 3:
        return None

    # Image projection uses waypoints in lidar frame with z=-lidar2ego[2,3].
    plane_z_m = 0.0
    lidar2ego_est = None
    if cam2ego is not None:
        try:
            lidar2ego_est = cam2ego @ np.linalg.inv(cam2lidar)
            if lidar2ego_est.shape == (4, 4):
                plane_z_m = float(-lidar2ego_est[2, 3])
        except Exception:
            lidar2ego_est = None
            plane_z_m = 0.0
    if not np.isfinite(plane_z_m):
        plane_z_m = 0.0
    plane_z_m = float(np.clip(plane_z_m, -4.0, 1.0))

    # Fold local-frame rotation + z-plane into effective camera extrinsic for
    # navsim_eval_viz's generic projection path (which assumes local==lidar, z=0).
    local_to_lidar = _R_CUR_TO_VEHICLE.T.astype(np.float32)
    z_offset = np.asarray([0.0, 0.0, float(plane_z_m)], dtype=np.float32)
    rr_eff = (local_to_lidar.T @ rr).astype(np.float32)
    tt_eff = ((tt[:3] - z_offset) @ local_to_lidar).astype(np.float32)

    out: Dict[str, Any] = {
        "intrinsics": intrinsic_scaled.astype(np.float32),
        "sensor2lidar_rotation": rr_eff,
        "sensor2lidar_translation": tt_eff,
    }
    if source == "world2cam_lidar2world":
        out["cam_frame"] = "raw_annotation"

    debug_best: Dict[str, Any] = {
        "source": source,
        "plane_z_m": float(plane_z_m),
        "cam2lidar_t_raw": [float(tt[0]), float(tt[1]), float(tt[2])],
        "cam2lidar_t_eff": [float(tt_eff[0]), float(tt_eff[1]), float(tt_eff[2])],
        "anchor_frame": "vehicle_xy_to_lidar_xy",
    }
    if lidar2ego_est is not None and isinstance(lidar2ego_est, np.ndarray) and lidar2ego_est.shape == (4, 4):
        debug_best["lidar2ego_est_t"] = [
            float(lidar2ego_est[0, 3]),
            float(lidar2ego_est[1, 3]),
            float(lidar2ego_est[2, 3]),
        ]
    out["_debug_best"] = debug_best
    out["_debug_projection"] = {
        "best_score": 0,
        "best": debug_best,
        "top": [debug_best],
    }
    out["_projection_chain_params"] = {
        "intrinsics": intrinsic_scaled.astype(np.float32),
        "cam2lidar_rotation_raw": rr.astype(np.float32),
        "cam2lidar_translation_raw": tt.astype(np.float32),
        "local_to_lidar_rotation": local_to_lidar.astype(np.float32),
        "plane_z_m": float(plane_z_m),
        "image_h": int(image_h),
        "image_w": int(image_w),
    }
    return out


def _projection_chain_for_polyline(
    local_xy: np.ndarray,
    *,
    chain_params: Dict[str, Any],
    image_h: int,
    image_w: int,
    max_points: int,
    eps: float = 1e-3,
) -> Dict[str, Any]:
    arr = np.asarray(local_xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        arr = np.zeros((0, 2), dtype=np.float32)
    if arr.shape[0] > max(0, int(max_points)):
        arr = arr[: int(max_points), :2]

    rr = np.asarray(chain_params.get("cam2lidar_rotation_raw"), dtype=np.float32)
    tt = np.asarray(chain_params.get("cam2lidar_translation_raw"), dtype=np.float32).reshape(-1)
    kk = np.asarray(chain_params.get("intrinsics"), dtype=np.float32)
    local_to_lidar = np.asarray(chain_params.get("local_to_lidar_rotation"), dtype=np.float32)
    plane_z_m = float(chain_params.get("plane_z_m", 0.0))

    if (
        rr.shape != (3, 3)
        or tt.shape[0] < 3
        or kk.ndim != 2
        or kk.shape[0] < 3
        or kk.shape[1] < 3
        or local_to_lidar.shape != (3, 3)
    ):
        return {"num_points": int(arr.shape[0]), "error": "invalid chain params"}

    points_local = np.concatenate([arr[:, :2], np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
    points_lidar = points_local @ local_to_lidar.T
    points_lidar[:, 2] = float(plane_z_m)
    points_cam = (points_lidar - tt[None, :3]) @ rr
    points_img_h = points_cam @ kk.T
    depth = points_img_h[:, 2]
    safe_depth = np.maximum(depth, float(eps))
    points_img = points_img_h[:, :2] / safe_depth[:, None]
    in_image = (
        (points_img[:, 0] >= 0.0)
        & (points_img[:, 0] <= max(0, int(image_w) - 1))
        & (points_img[:, 1] >= 0.0)
        & (points_img[:, 1] <= max(0, int(image_h) - 1))
    )
    finite = np.isfinite(points_img).all(axis=1)
    valid = (depth > float(eps)) & in_image & finite

    return {
        "num_points": int(arr.shape[0]),
        "local_xy": arr[:, :2].astype(np.float32).tolist(),
        "lidar_xyz": points_lidar.astype(np.float32).tolist(),
        "cam_xyz": points_cam.astype(np.float32).tolist(),
        "img_uv": points_img.astype(np.float32).tolist(),
        "depth": depth.astype(np.float32).tolist(),
        "valid": valid.astype(np.bool_).tolist(),
    }


def _dump_projection_chain_debug(
    *,
    out_dir: Path,
    token: str,
    rank: int,
    image_h: int,
    image_w: int,
    camera_for_projection: Dict[str, Any],
    history_traj: np.ndarray,
    gt_future_traj: np.ndarray,
    pred_future_traj: np.ndarray,
    max_points: int,
) -> Optional[Path]:
    chain_params = camera_for_projection.get("_projection_chain_params")
    if not isinstance(chain_params, dict):
        return None

    hist_xy = np.asarray(history_traj, dtype=np.float32)
    gt_xy = np.asarray(gt_future_traj, dtype=np.float32)
    pred_xy = np.asarray(pred_future_traj, dtype=np.float32)
    anchor = np.array([[0.0, 0.0]], dtype=np.float32)

    hist_line = np.concatenate([hist_xy[:, :2], anchor], axis=0) if hist_xy.ndim == 2 and hist_xy.shape[0] > 0 else anchor
    gt_line = np.concatenate([anchor, gt_xy[:, :2]], axis=0) if gt_xy.ndim == 2 and gt_xy.shape[0] > 0 else anchor
    pred_line = np.concatenate([anchor, pred_xy[:, :2]], axis=0) if pred_xy.ndim == 2 and pred_xy.shape[0] > 0 else anchor

    payload = {
        "token": token,
        "rank": int(rank),
        "image_h": int(image_h),
        "image_w": int(image_w),
        "params": {
            "plane_z_m": float(chain_params.get("plane_z_m", 0.0)),
            "cam2lidar_translation_raw": np.asarray(
                chain_params.get("cam2lidar_translation_raw", np.zeros((3,), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)[:3].tolist(),
            "local_to_lidar_rotation": np.asarray(
                chain_params.get("local_to_lidar_rotation", np.eye(3, dtype=np.float32)),
                dtype=np.float32,
            ).tolist(),
        },
        "history": _projection_chain_for_polyline(
            hist_line,
            chain_params=chain_params,
            image_h=int(image_h),
            image_w=int(image_w),
            max_points=int(max_points),
        ),
        "gt": _projection_chain_for_polyline(
            gt_line,
            chain_params=chain_params,
            image_h=int(image_h),
            image_w=int(image_w),
            max_points=int(max_points),
        ),
        "pred": _projection_chain_for_polyline(
            pred_line,
            chain_params=chain_params,
            image_h=int(image_h),
            image_w=int(image_w),
            max_points=int(max_points),
        ),
    }

    debug_dir = out_dir / "projection_chain_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{token}_rank{rank}.json"
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        "[infer_b2d][proj_chain]",
        f"token={token}",
        f"plane_z_m={payload['params']['plane_z_m']:.4f}",
        f"json={debug_path}",
    )
    print(
        "[infer_b2d][proj_chain][counts]",
        f"hist={payload['history'].get('num_points', 0)}",
        f"gt={payload['gt'].get('num_points', 0)}",
        f"pred={payload['pred'].get('num_points', 0)}",
    )
    print(f"[infer_b2d][proj_chain][full] {json.dumps(payload, ensure_ascii=False)}")
    return debug_path


def _safe_traj_np(traj: np.ndarray, expected_len: int) -> np.ndarray:
    return _fit_len_2d(traj, expected_len=expected_len, channels=3)


def _cmd_from_future_yaw(future_xyh: np.ndarray, yaw_thresh_deg: float = 8.0) -> np.ndarray:
    if future_xyh.ndim != 2 or future_xyh.shape[0] == 0:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    yaw_deg = float(np.degrees(float(future_xyh[-1, 2])))
    if yaw_deg > yaw_thresh_deg:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if yaw_deg < -yaw_thresh_deg:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _warn_planning_label_once(key: str, msg: str) -> None:
    if key in _WARNED_PLANNING_LABEL_ISSUES:
        return
    _WARNED_PLANNING_LABEL_ISSUES.add(key)
    print(f"[infer_b2d][planning][warn] {msg}")


def _normalize_b2d_actor_name(raw_name: Any) -> str:
    token = str(raw_name or "").strip().lower()
    if token in _B2D_CLASS_TO_INDEX:
        return token
    if "pedestrian" in token or token.startswith("walker."):
        return "pedestrian"
    if "bicycle" in token or "crossbike" in token or "omafiets" in token:
        return "bicycle"
    if "traffic_light" in token or "traffic.light" in token:
        return "traffic_light"
    if "traffic_cone" in token or "traffic.cone" in token:
        return "traffic_cone"
    if (
        "traffic_sign" in token
        or "speed_limit" in token
        or token == "traffic.stop"
        or token == "traffic.yield"
    ):
        return "traffic_sign"
    if "truck" in token or "firetruck" in token:
        return "truck"
    if "van" in token or "ambulance" in token:
        return "van"
    if "vehicle." in token or "car" in token:
        return "car"
    return "others"


def _to_matrix4x4(value: Any) -> Optional[np.ndarray]:
    try:
        mat = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if mat.shape != (4, 4):
        return None
    return mat


def _dataset_global_index(dataset: Any, dataset_idx: int) -> int:
    valid_indices = getattr(dataset, "valid_indices", None)
    if isinstance(valid_indices, (list, tuple, np.ndarray)) and len(valid_indices) > 0:
        try:
            return int(valid_indices[int(dataset_idx) % len(valid_indices)])
        except Exception:
            return int(dataset_idx)
    return int(dataset_idx)


def _extract_b2d_planning_labels(
    dataset: Any,
    *,
    dataset_idx: int,
    future_frames: int,
    sample_interval: int,
) -> Optional[Dict[str, np.ndarray]]:
    infos = getattr(dataset, "data_infos", None)
    if not isinstance(infos, list) or len(infos) <= 0:
        _warn_planning_label_once("missing_data_infos", "dataset has no data_infos; skip planning metrics.")
        return None

    global_idx = _dataset_global_index(dataset, int(dataset_idx))
    if global_idx < 0 or global_idx >= len(infos):
        _warn_planning_label_once("bad_global_idx", f"invalid global index {global_idx}; skip planning metrics.")
        return None

    cur_frame = infos[global_idx]
    if not isinstance(cur_frame, dict):
        _warn_planning_label_once("bad_cur_frame", "current frame entry is not dict; skip planning metrics.")
        return None

    required_keys = ("gt_boxes", "gt_ids", "gt_names", "npc2world", "sensors")
    missing = [k for k in required_keys if k not in cur_frame]
    if missing:
        _warn_planning_label_once(
            "missing_planning_fields",
            "current ann entry missing planning fields "
            f"{missing}; planning metrics unavailable for this ann format.",
        )
        return None

    sensors = cur_frame.get("sensors", {})
    lidar_info = sensors.get("LIDAR_TOP", {}) if isinstance(sensors, dict) else {}
    world2lidar_cur = _to_matrix4x4(lidar_info.get("world2lidar"))
    if world2lidar_cur is None:
        _warn_planning_label_once(
            "missing_world2lidar",
            "LIDAR_TOP.world2lidar missing/invalid; skip planning metrics.",
        )
        return None

    gt_boxes = np.asarray(cur_frame.get("gt_boxes"), dtype=np.float32)
    if gt_boxes.ndim != 2:
        _warn_planning_label_once("bad_gt_boxes", "gt_boxes has invalid shape; skip planning metrics.")
        return None
    if gt_boxes.shape[1] < 9:
        pad = np.zeros((gt_boxes.shape[0], 9 - gt_boxes.shape[1]), dtype=np.float32)
        gt_boxes = np.concatenate([gt_boxes, pad], axis=1)
    gt_boxes = gt_boxes[:, :9].astype(np.float32, copy=False)

    gt_ids = np.asarray(cur_frame.get("gt_ids"))
    if gt_ids.ndim != 1 or gt_ids.shape[0] != gt_boxes.shape[0]:
        _warn_planning_label_once(
            "bad_gt_ids",
            "gt_ids shape mismatch with gt_boxes; skip planning metrics.",
        )
        return None
    n_agents = int(gt_boxes.shape[0])
    if n_agents <= 0:
        frames = max(0, int(future_frames))
        return {
            "gt_boxes": np.zeros((0, 9), dtype=np.float32),
            "gt_attr": np.zeros((0, 4 * frames + 10), dtype=np.float32),
        }

    gt_names_raw = cur_frame.get("gt_names", [])
    if not isinstance(gt_names_raw, (list, tuple, np.ndarray)):
        gt_names_raw = ["others"] * n_agents
    if len(gt_names_raw) < n_agents:
        gt_names_raw = list(gt_names_raw) + ["others"] * (n_agents - len(gt_names_raw))
    gt_names = [str(x) for x in gt_names_raw[:n_agents]]

    frames = max(0, int(future_frames))
    step = max(1, int(sample_interval))
    future_track = np.zeros((n_agents, frames + 1, 2), dtype=np.float32)
    future_mask = np.zeros((n_agents, frames + 1), dtype=np.float32)
    future_yaw = np.zeros((n_agents, frames + 1), dtype=np.float32)
    gt_fut_goal = np.zeros((n_agents, 1), dtype=np.float32)
    agent_lcf_feat = np.zeros((n_agents, 9), dtype=np.float32)

    for agent_idx in range(n_agents):
        cls_name = _normalize_b2d_actor_name(gt_names[agent_idx])
        cls_idx = int(_B2D_CLASS_TO_INDEX.get(cls_name, _B2D_CLASS_TO_INDEX["others"]))
        agent_lcf_feat[agent_idx, 0:2] = gt_boxes[agent_idx, 0:2]
        agent_lcf_feat[agent_idx, 2] = gt_boxes[agent_idx, 6]
        agent_lcf_feat[agent_idx, 3:5] = gt_boxes[agent_idx, 7:9]
        agent_lcf_feat[agent_idx, 5:8] = gt_boxes[agent_idx, 3:6]
        agent_lcf_feat[agent_idx, 8] = float(cls_idx)

    scene_name = str(cur_frame.get("folder", ""))
    for step_idx in range(frames + 1):
        adj_global_idx = global_idx + step_idx * step
        if adj_global_idx < 0 or adj_global_idx >= len(infos):
            break
        adj_frame = infos[adj_global_idx]
        if not isinstance(adj_frame, dict):
            continue
        if str(adj_frame.get("folder", "")) != scene_name:
            break

        adj_ids = np.asarray(adj_frame.get("gt_ids"))
        adj_npc2world = np.asarray(adj_frame.get("npc2world"), dtype=np.float32)
        if adj_ids.ndim != 1 or adj_npc2world.ndim != 3 or adj_npc2world.shape[1:] != (4, 4):
            continue

        for agent_idx in range(n_agents):
            box_id = gt_ids[agent_idx]
            matches = np.where(adj_ids == box_id)[0]
            if matches.size != 1:
                continue
            matched_idx = int(matches[0])
            if matched_idx < 0 or matched_idx >= adj_npc2world.shape[0]:
                continue
            adj_box2lidar = world2lidar_cur @ adj_npc2world[matched_idx]
            future_track[agent_idx, step_idx, :] = adj_box2lidar[0:2, 3]
            future_mask[agent_idx, step_idx] = 1.0
            future_yaw[agent_idx, step_idx] = float(
                np.arctan2(adj_box2lidar[1, 0], adj_box2lidar[0, 0])
            )

    for agent_idx in range(n_agents):
        coord_diff = future_track[agent_idx, -1] - future_track[agent_idx, 0]
        if float(np.max(coord_diff)) < 1.0:
            gt_fut_goal[agent_idx, 0] = 9.0
        else:
            box_mot_yaw = float(np.arctan2(coord_diff[1], coord_diff[0]) + np.pi)
            gt_fut_goal[agent_idx, 0] = float(np.floor(box_mot_yaw / (np.pi / 4.0)))

    future_track_offset = future_track[:, 1:, :] - future_track[:, :-1, :]
    future_mask_offset = future_mask[:, 1:]
    future_track_offset[future_mask_offset == 0] = 0.0
    future_yaw_offset = future_yaw[:, 1:] - future_yaw[:, :-1]
    future_yaw_offset[future_yaw_offset > np.pi] -= np.pi * 2.0
    future_yaw_offset[future_yaw_offset < -np.pi] += np.pi * 2.0

    gt_attr = np.concatenate(
        [
            future_track_offset.reshape(n_agents, -1),
            future_mask_offset,
            gt_fut_goal,
            agent_lcf_feat,
            future_yaw_offset,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    return {"gt_boxes": gt_boxes, "gt_attr": gt_attr}


def _fill_poly_mask(mask: np.ndarray, poly: np.ndarray, value: int = 1) -> None:
    if mask.ndim != 2:
        return
    poly = np.asarray(poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        return
    pil_img = Image.fromarray(mask, mode="L")
    draw = ImageDraw.Draw(pil_img)
    draw.polygon([(float(x), float(y)) for x, y in poly[:, :2]], fill=int(value))
    mask[...] = np.asarray(pil_img, dtype=np.uint8)


class _B2DPlanningMetricLite:
    def __init__(self) -> None:
        self.X_BOUND = np.array([-50.0, 50.0, 0.5], dtype=np.float32)
        self.Y_BOUND = np.array([-50.0, 50.0, 0.5], dtype=np.float32)
        self.bev_resolution = np.array([self.X_BOUND[2], self.Y_BOUND[2]], dtype=np.float32)
        self.bev_start_position = np.array(
            [self.X_BOUND[0] + self.X_BOUND[2] / 2.0, self.Y_BOUND[0] + self.Y_BOUND[2] / 2.0],
            dtype=np.float32,
        )
        self.bev_dimension = np.array(
            [
                int((self.X_BOUND[1] - self.X_BOUND[0]) / self.X_BOUND[2]),
                int((self.Y_BOUND[1] - self.Y_BOUND[0]) / self.Y_BOUND[2]),
            ],
            dtype=np.int32,
        )
        self.dx = np.array([self.X_BOUND[2], self.Y_BOUND[2]], dtype=np.float32)
        self.bx = np.array(
            [self.X_BOUND[0] + self.X_BOUND[2] / 2.0, self.Y_BOUND[0] + self.Y_BOUND[2] / 2.0],
            dtype=np.float32,
        )
        self.ego_width = 1.85
        self.ego_length = 4.084
        self.vehicle_indices = {0, 1, 2}
        self.human_indices = {3, 7}
        self._ego_rc = self._build_ego_footprint_rc()

    @property
    def bev_h(self) -> int:
        return int(self.bev_dimension[0])

    @property
    def bev_w(self) -> int:
        return int(self.bev_dimension[1])

    def _build_ego_footprint_rc(self) -> np.ndarray:
        pts = np.array(
            [
                [-self.ego_length / 2.0 + 0.5, self.ego_width / 2.0],
                [self.ego_length / 2.0 + 0.5, self.ego_width / 2.0],
                [self.ego_length / 2.0 + 0.5, -self.ego_width / 2.0],
                [-self.ego_length / 2.0 + 0.5, -self.ego_width / 2.0],
            ],
            dtype=np.float32,
        )
        pts = (pts - self.bx[None, :]) / self.dx[None, :]
        pts[:, [0, 1]] = pts[:, [1, 0]]
        poly = np.round(np.stack([pts[:, 0], pts[:, 1]], axis=-1)).astype(np.int32)
        mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        _fill_poly_mask(mask, poly, value=1)
        rr, cc = np.where(mask > 0)
        if rr.size == 0:
            rr = np.array([0], dtype=np.int32)
            cc = np.array([0], dtype=np.int32)
        return np.stack([rr, cc], axis=-1).astype(np.float32)

    def _agent_poly_region(self, x_a: float, y_a: float, yaw_a: float, length: float, width: float) -> np.ndarray:
        lidar2cv_rot = np.array([[1, 0], [0, -1]], dtype=np.float32)
        trans_a = np.array([[x_a], [y_a]], dtype=np.float32)
        rot_mat_a = np.array(
            [[np.cos(yaw_a), -np.sin(yaw_a)], [np.sin(yaw_a), np.cos(yaw_a)]],
            dtype=np.float32,
        )
        agent_corner = np.array(
            [
                [length / 2.0, -length / 2.0, -length / 2.0, length / 2.0],
                [width / 2.0, width / 2.0, -width / 2.0, -width / 2.0],
            ],
            dtype=np.float32,
        )
        agent_corner_lidar = rot_mat_a @ agent_corner + trans_a
        agent_corner_cv = (
            (lidar2cv_rot @ agent_corner_lidar)
            - self.bev_start_position[:2, None]
            + self.bev_resolution[:2, None] / 2.0
        ).T / self.bev_resolution[:2]
        return np.round(agent_corner_cv).astype(np.int32)

    def build_occupancy(self, gt_boxes: np.ndarray, gt_attr: np.ndarray, n_future: int) -> np.ndarray:
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        gt_attr = np.asarray(gt_attr, dtype=np.float32)
        n_future = max(0, int(n_future))
        if n_future <= 0:
            return np.zeros((0, self.bev_h, self.bev_w), dtype=np.uint8)
        if gt_boxes.ndim != 2 or gt_attr.ndim != 2 or gt_boxes.shape[0] <= 0 or gt_attr.shape[0] <= 0:
            return np.zeros((n_future, self.bev_h, self.bev_w), dtype=np.uint8)

        n_agents = min(int(gt_boxes.shape[0]), int(gt_attr.shape[0]))
        gt_boxes = gt_boxes[:n_agents]
        gt_attr = gt_attr[:n_agents]

        attr_dim = int(gt_attr.shape[1])
        t_attr = (attr_dim - 10) // 4
        if t_attr <= 0:
            return np.zeros((n_future, self.bev_h, self.bev_w), dtype=np.uint8)
        t_use = min(int(n_future), int(t_attr))
        if t_use <= 0:
            return np.zeros((n_future, self.bev_h, self.bev_w), dtype=np.uint8)

        segmentation = np.zeros((t_use, self.bev_h, self.bev_w), dtype=np.uint8)
        pedestrian = np.zeros((t_use, self.bev_h, self.bev_w), dtype=np.uint8)

        gt_agent_fut_trajs = gt_attr[:, : t_use * 2].reshape(n_agents, t_use, 2)
        gt_agent_fut_mask = gt_attr[:, t_attr * 2 : t_attr * 2 + t_use].reshape(n_agents, t_use)
        yaw_start = t_attr * 3 + 10
        gt_agent_fut_yaw = gt_attr[:, yaw_start : yaw_start + t_use].reshape(n_agents, t_use, 1)

        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        boxes = gt_boxes.copy()
        boxes[:, 6:7] = -1.0 * (boxes[:, 6:7] + np.pi / 2.0)
        gt_agent_fut_trajs = gt_agent_fut_trajs + boxes[:, None, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + boxes[:, None, 6:7]

        cls_col = t_attr * 3 + 9
        for t in range(t_use):
            for i in range(n_agents):
                if float(gt_agent_fut_mask[i, t]) < 0.5:
                    continue
                cls_idx = int(round(float(gt_attr[i, cls_col]))) if cls_col < gt_attr.shape[1] else -1
                agent_length = float(boxes[i, 4])
                agent_width = float(boxes[i, 3])
                x_a = float(gt_agent_fut_trajs[i, t, 0])
                y_a = float(gt_agent_fut_trajs[i, t, 1])
                yaw_a = float(gt_agent_fut_yaw[i, t, 0])
                poly = self._agent_poly_region(x_a, y_a, yaw_a, agent_length, agent_width)
                if cls_idx in self.vehicle_indices:
                    _fill_poly_mask(segmentation[t], poly, value=1)
                if cls_idx in self.human_indices:
                    _fill_poly_mask(pedestrian[t], poly, value=1)

        occupancy = np.logical_or(segmentation > 0, pedestrian > 0).astype(np.uint8)
        if occupancy.shape[0] < n_future:
            pad = np.zeros((n_future - occupancy.shape[0], self.bev_h, self.bev_w), dtype=np.uint8)
            occupancy = np.concatenate([occupancy, pad], axis=0)
        return occupancy[:n_future]

    def evaluate_single_coll(self, traj_xy: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        occupancy = np.asarray(occupancy, dtype=np.uint8)
        n_future = min(int(traj_xy.shape[0]), int(occupancy.shape[0]))
        if n_future <= 0:
            return np.zeros((0,), dtype=np.bool_)
        traj_xy = traj_xy[:n_future]
        occupancy = occupancy[:n_future]

        trajs = traj_xy.reshape(n_future, 1, 2).copy()
        trajs[:, :, [0, 1]] = trajs[:, :, [1, 0]]
        trajs = trajs / self.dx[None, None, :]
        trajs = trajs + self._ego_rc[None, :, :]

        r = (self.bev_h - trajs[:, :, 0]).astype(np.int32)
        c = trajs[:, :, 1].astype(np.int32)
        r = np.clip(r, 0, self.bev_h - 1)
        c = np.clip(c, 0, self.bev_w - 1)

        collision = np.zeros((n_future,), dtype=np.bool_)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            valid = (rr >= 0) & (rr < self.bev_h) & (cc >= 0) & (cc < self.bev_w)
            if np.any(valid):
                collision[t] = bool(np.any(occupancy[t, rr[valid], cc[valid]] > 0))
        return collision

    def evaluate_coll(
        self,
        trajs: np.ndarray,
        gt_trajs: np.ndarray,
        occupancy: np.ndarray,
    ) -> np.ndarray:
        trajs = np.asarray(trajs, dtype=np.float32)
        gt_trajs = np.asarray(gt_trajs, dtype=np.float32)
        occupancy = np.asarray(occupancy, dtype=np.uint8)
        if trajs.ndim == 2:
            trajs = trajs[None, ...]
        if gt_trajs.ndim == 2:
            gt_trajs = gt_trajs[None, ...]
        if occupancy.ndim == 3:
            occupancy = occupancy[None, ...]

        bsz = min(int(trajs.shape[0]), int(gt_trajs.shape[0]), int(occupancy.shape[0]))
        n_future = min(int(trajs.shape[1]), int(gt_trajs.shape[1]), int(occupancy.shape[1]))
        if bsz <= 0 or n_future <= 0:
            return np.zeros((0,), dtype=np.float32)

        trajs = trajs[:bsz, :n_future]
        gt_trajs = gt_trajs[:bsz, :n_future]
        occupancy = occupancy[:bsz, :n_future]

        obj_box_coll_sum = np.zeros((n_future,), dtype=np.float32)
        ti = np.arange(n_future)

        for i in range(bsz):
            gt_box_coll = self.evaluate_single_coll(gt_trajs[i], occupancy[i])
            m2 = ~gt_box_coll
            box_coll = self.evaluate_single_coll(trajs[i], occupancy[i])
            if np.any(m2):
                obj_box_coll_sum[ti[m2]] += box_coll[ti[m2]].astype(np.float32)

        return obj_box_coll_sum

    def compute_l2(self, traj_xy: np.ndarray, gt_xy: np.ndarray) -> float:
        traj_xy = np.asarray(traj_xy, dtype=np.float32)
        gt_xy = np.asarray(gt_xy, dtype=np.float32)
        n = min(int(traj_xy.shape[0]), int(gt_xy.shape[0]))
        if n <= 0:
            return float("nan")
        diff = traj_xy[:n] - gt_xy[:n]
        return float(np.linalg.norm(diff, axis=-1).mean())


def _empty_planning_metric_dict() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "planning_metrics_available": False,
        "plan_L2_1s": float("nan"),
        "plan_L2_2s": float("nan"),
        "plan_L2_3s": float("nan"),
        "plan_L2_avg": float("nan"),
        "plan_obj_box_col_1s": float("nan"),
        "plan_obj_box_col_2s": float("nan"),
        "plan_obj_box_col_3s": float("nan"),
        "plan_obj_box_col_avg": float("nan"),
    }
    return out


def _compute_planning_metrics_for_scene(
    pred_xyh: np.ndarray,
    gt_xyh: np.ndarray,
    planning_labels: Optional[Dict[str, np.ndarray]],
    planning_metric: _B2DPlanningMetricLite,
    target_fps: int,
) -> Dict[str, Any]:
    out = _empty_planning_metric_dict()
    if planning_labels is None:
        return out

    pred_xyh = np.asarray(pred_xyh, dtype=np.float32)
    gt_xyh = np.asarray(gt_xyh, dtype=np.float32)
    n = min(int(pred_xyh.shape[0]), int(gt_xyh.shape[0]))
    if n <= 0:
        return out

    gt_boxes = np.asarray(planning_labels.get("gt_boxes"), dtype=np.float32)
    gt_attr = np.asarray(planning_labels.get("gt_attr"), dtype=np.float32)
    occupancy = planning_metric.build_occupancy(gt_boxes, gt_attr, n_future=n)
    if occupancy.shape[0] <= 0:
        return out

    n = min(n, int(occupancy.shape[0]))
    # Convert model/training vehicle frame (x forward, y left) back to
    # Bench2Drive lidar frame (x right, y forward) for collision rasterization.
    pred_xy_vehicle = pred_xyh[:n, :2]
    gt_xy_vehicle = gt_xyh[:n, :2]
    pred_xy = np.stack([-pred_xy_vehicle[:, 1], pred_xy_vehicle[:, 0]], axis=1).astype(np.float32)
    gt_xy = np.stack([-gt_xy_vehicle[:, 1], gt_xy_vehicle[:, 0]], axis=1).astype(np.float32)
    occupancy = occupancy[:n]
    out["planning_metrics_available"] = True

    l2_values = []
    obj_box_col_values = []
    for sec in (1, 2, 3):
        horizon = min(n, max(1, int(round(float(sec) * float(target_fps)))))
        pred_h = pred_xy[:horizon]
        gt_h = gt_xy[:horizon]
        occ_h = occupancy[:horizon]
        l2_val = planning_metric.compute_l2(pred_h, gt_h)
        obj_box_col = planning_metric.evaluate_coll(
            pred_h[None, ...],
            gt_h[None, ...],
            occ_h[None, ...],
        )
        out[f"plan_L2_{sec}s"] = float(np.nan_to_num(l2_val))
        out[f"plan_obj_box_col_{sec}s"] = float(np.nan_to_num(np.mean(obj_box_col)))
        l2_values.append(out[f"plan_L2_{sec}s"])
        obj_box_col_values.append(out[f"plan_obj_box_col_{sec}s"])

    out["plan_L2_avg"] = float(np.nan_to_num(np.mean(l2_values)))
    out["plan_obj_box_col_avg"] = float(np.nan_to_num(np.mean(obj_box_col_values)))
    return out


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


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_eval(args: argparse.Namespace, external_pipe: Optional[WanVideoPipeline] = None) -> None:
    dist_info = _init_distributed(bool(args.distributed))
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model_future_frames = int(args.model_future_frames) if args.model_future_frames is not None else int(args.num_future_frames)

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
            print("[infer_b2d] using external pipe from training process")

    _set_vae_progress(bool(args.show_vae_progress))
    pipe.target_fps = int(args.target_fps)
    # Bench2Drive launch defaults to velocity conditioning; the future trajectory
    # itself is still decoded in the UNIVERSE local x/y/heading space.
    pipe.trajectory_norm_mode = "UNIVERSE_odo"
    pipe.trajectory_use_relative = False
    pipe.trajectory_condition_mode = "velocity"
    infer_output_mode = getattr(args, "infer_output_mode", None)
    infer_trajectory_only = bool(getattr(args, "infer_trajectory_only", False))
    if infer_output_mode == "trajectory":
        infer_trajectory_only = True
    elif infer_output_mode in {"video", "both"}:
        infer_trajectory_only = False
    pipe.use_mixed_latent_attention_mask = bool(args.use_mixed_latent_attention_mask)
    pipe.infer_trajectory_only = infer_trajectory_only

    if external_pipe is None:
        if args.full_ckpt:
            if os.path.exists(args.full_ckpt):
                if rank == 0:
                    print(f"[infer_b2d] loading full checkpoint: {args.full_ckpt}")
                state_dict_raw = load_state_dict(args.full_ckpt)
                state_dict = _normalize_train_ckpt_keys(state_dict_raw)
                missing, unexpected = pipe.load_state_dict(state_dict, strict=False)
                if rank == 0:
                    print(
                        "[infer_b2d] full_ckpt loaded:",
                        f"keys={len(state_dict)}",
                        f"missing={len(missing)}",
                        f"unexpected={len(unexpected)}",
                    )
            elif rank == 0:
                print(f"[infer_b2d][warn] full_ckpt not found: {args.full_ckpt}; fallback to base model")
        elif args.lora_checkpoint:
            if rank == 0:
                print(f"[infer_b2d] loading LoRA checkpoint: {args.lora_checkpoint}")
            pipe.load_lora(pipe.dit, args.lora_checkpoint, alpha=args.lora_alpha)

    dataset_stride = int(args.frame_interval) if args.frame_interval is not None else int(args.dataset_stride)
    dataset_kwargs = {
        "data_root": args.data_root,
        "ann_file": args.ann_file,
        "num_history_frames": args.num_history_frames,
        "num_future_frames": args.num_future_frames,
        "dataset_stride": dataset_stride,
        "target_fps": args.target_fps,
        "original_fps": args.original_fps,
        "height": args.height,
        "width": args.width,
        "use_focal_unification": bool(args.use_focal_unification),
        "focal_target_px": args.focal_target_px,
        "use_structured_prompt": bool(args.use_structured_prompt),
    }
    init_sig = inspect.signature(Bench2DriveEvalDataset.__init__)
    if "load_multi_views" in init_sig.parameters:
        dataset_kwargs["load_multi_views"] = False
    dataset = Bench2DriveEvalDataset(**dataset_kwargs)
    if hasattr(dataset, "load_multi_views"):
        dataset.load_multi_views = False
    if rank == 0:
        focal_config = getattr(dataset, "focal_config", None)
        print(
            "[infer_b2d][focal]",
            f"enabled={bool(getattr(focal_config, 'enabled', False))}",
            f"target_focal_px={getattr(focal_config, 'target_focal_px', None)}",
            f"output={args.width}x{args.height}",
        )

    projection_debug_token = str(getattr(args, "projection_debug_token", "") or "").strip()
    dataset_size = int(len(dataset))
    max_scenes = dataset_size if args.max_scenes is None else min(int(args.max_scenes), dataset_size)
    scene_seed = int(args.seed) & 0xFFFFFFFF
    scene_rng = np.random.default_rng(scene_seed)
    if max_scenes < dataset_size:
        # Random subset for eval/viz to avoid always selecting the first contiguous scenes.
        eval_indices = scene_rng.choice(dataset_size, size=max_scenes, replace=False).tolist()
    else:
        eval_indices = list(range(dataset_size))
        scene_rng.shuffle(eval_indices)

    if projection_debug_token:
        forced_idx: Optional[int] = None
        infos = getattr(dataset, "data_infos", None)
        if isinstance(infos, list) and len(infos) > 0:
            for dataset_idx in range(dataset_size):
                global_idx = _dataset_global_index(dataset, int(dataset_idx))
                if global_idx < 0 or global_idx >= len(infos):
                    continue
                info = infos[global_idx]
                if not isinstance(info, dict):
                    continue
                scene_name = str(info.get("folder", info.get("scene_name", info.get("scene", ""))))
                frame_idx = int(info.get("frame_idx", -1))
                token = f"{scene_name.replace('/', '_')}_{frame_idx:05d}"
                if token == projection_debug_token:
                    forced_idx = int(dataset_idx)
                    break
        if forced_idx is not None:
            eval_indices = [int(forced_idx)] + [int(x) for x in eval_indices if int(x) != int(forced_idx)]
            if rank == 0:
                print(
                    "[infer_b2d] projection token forced into eval set:",
                    f"token={projection_debug_token}",
                    f"dataset_idx={forced_idx}",
                    "rank0_pinned=True",
                )
        elif rank == 0:
            print(
                "[infer_b2d][proj_chain][warn]",
                f"projection_debug_token not found: {projection_debug_token}",
            )

    local_indices = eval_indices[rank::world_size]
    global_viz_indices: set[int] = set()
    global_viz_order: List[int] = []
    viz_index_by_eval_idx: Dict[int, int] = {}
    if bool(args.save_viz):
        viz_cap = max(0, int(args.viz_max_scenes))
        if viz_cap > 0 and len(eval_indices) > 0:
            if len(eval_indices) <= viz_cap:
                global_viz_order = [int(x) for x in eval_indices]
            else:
                selected = scene_rng.choice(
                    np.asarray(eval_indices, dtype=np.int64),
                    size=viz_cap,
                    replace=False,
                )
                global_viz_indices = {int(x) for x in selected.tolist()}
                global_viz_order = [int(x) for x in eval_indices if int(x) in global_viz_indices]
            if len(global_viz_order) <= 0:
                global_viz_order = [int(x) for x in eval_indices if int(x) in global_viz_indices]
            global_viz_indices = set(global_viz_order)
            for i, eval_idx in enumerate(global_viz_order, start=1):
                viz_index_by_eval_idx[int(eval_idx)] = i

    out_dir = Path(args.output_dir)
    viz_dir = out_dir / "viz"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_viz:
            viz_dir.mkdir(parents=True, exist_ok=True)

    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0:
        print(
            f"[infer_b2d] dataset={len(dataset)} eval={len(eval_indices)} local={len(local_indices)} "
            f"history={args.num_history_frames} future={args.num_future_frames} model_future={model_future_frames} "
            f"scene_seed={scene_seed}"
        )
        if bool(args.save_viz):
            print(
                "[infer_b2d] viz selection:",
                f"global_cap={int(args.viz_max_scenes)}",
                f"global_selected={len(global_viz_indices)}",
            )
            print(
                "[infer_b2d] projection debug:",
                f"steps={int(args.projection_debug_steps)}",
                "(set --projection_debug_steps > 0 to print per-sample candidate details)",
            )
            if projection_debug_token:
                print(
                    "[infer_b2d] projection chain token debug:",
                    f"token={projection_debug_token}",
                    f"max_points={int(args.projection_debug_max_points)}",
                )
        if bool(args.compute_planning_metrics):
            print("[infer_b2d] planning metrics: enabled (Bench2Drive L2/collision)")
        else:
            print("[infer_b2d] planning metrics: disabled")

    results: List[Dict[str, Any]] = []
    viz_saved = 0
    planning_metric = _MigratedB2DPlanningMetricLite() if bool(args.compute_planning_metrics) else None
    planning_sample_interval = max(1, int(getattr(dataset, "sample_interval", dataset_stride)))
    printed_prompt_steps = 0
    printed_projection_debug = 0

    iterator = (
        tqdm(local_indices, desc="Infer-B2D", total=len(local_indices), dynamic_ncols=True)
        if (rank == 0 and args.show_eval_progress)
        else local_indices
    )
    for idx in iterator:
        row: Dict[str, Any] = {"dataset_idx": int(idx), "valid": True, "rank": rank}
        try:
            sample = dataset[int(idx)]
            scene = str(sample.get("scene_name", sample.get("scene", "")))
            frame_idx = int(sample.get("frame_idx", -1))
            token = f"{scene.replace('/', '_')}_{frame_idx:05d}"
            row["token"] = token
            row["scene"] = scene
            row["frame_idx"] = frame_idx

            history_video = sample.get("video_hist")
            if history_video is None:
                history_video = sample.get("longcat_video")
            if history_video is None:
                history_video = sample.get("video", [])[: int(args.num_history_frames)]
            gt_video = sample["video"]
            gt_future_video = sample.get("video_fut", gt_video[args.num_history_frames:])

            gt_future_pos = np.asarray(
                sample.get("gt_future_positions", np.zeros((args.num_future_frames, 3), dtype=np.float32)),
                dtype=np.float32,
            )
            gt_future_quat = np.asarray(
                sample.get("gt_future_quaternions", np.zeros((args.num_future_frames, 4), dtype=np.float32)),
                dtype=np.float32,
            )
            gt_history_pos = np.asarray(
                sample.get("gt_history_positions", np.zeros((args.num_history_frames, 3), dtype=np.float32)),
                dtype=np.float32,
            )
            gt_history_quat = np.asarray(
                sample.get("gt_history_quaternions", np.zeros((args.num_history_frames, 4), dtype=np.float32)),
                dtype=np.float32,
            )
            gt_future_traj = _pose_to_xyh(gt_future_pos, gt_future_quat, expected_len=int(args.num_future_frames))
            history_positions_full = _pose_to_xyh(gt_history_pos, gt_history_quat, expected_len=int(args.num_history_frames))
            # Drop the current anchor from history_positions: the pipeline
            # already treats the current frame as the origin of future motion.
            history_positions = history_positions_full[:-1] if history_positions_full.shape[0] > 1 else history_positions_full

            ego_vel_raw = np.asarray(sample.get("ego_vel", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if history_positions_full.shape[0] >= 2:
                # Prefer pose-derived local velocity so the prompt speed and
                # trajectory prefix use the same sampled frame interval.
                delta = history_positions_full[-1] - history_positions_full[-2]
                ego_vel = np.array(
                    [delta[0] * float(args.target_fps), delta[1] * float(args.target_fps), 0.0],
                    dtype=np.float32,
                )
            else:
                ego_vel = np.zeros((3,), dtype=np.float32)
                if ego_vel_raw.size >= 1:
                    ego_vel[0] = float(ego_vel_raw[0])
                if ego_vel_raw.size >= 2:
                    ego_vel[1] = float(ego_vel_raw[1])

            gt_cmd_onehot = _cmd_from_future_yaw(gt_future_traj, yaw_thresh_deg=float(args.command_yaw_threshold_deg))
            gt_cmd_text = one_hot_to_cmd(gt_cmd_onehot)
            speed_mps = float(np.linalg.norm(ego_vel[:2]))
            prompt = _build_prompt_fixed(torch.from_numpy(gt_cmd_onehot), speed_mps)
            if rank == 0 and printed_prompt_steps < max(0, int(args.debug_prompt_steps)):
                n = printed_prompt_steps + 1
                print(f"[infer_b2d][prompt][{n}/{args.debug_prompt_steps}] positive: {prompt}")
                print(f"[infer_b2d][prompt][{n}/{args.debug_prompt_steps}] negative: {args.negative_prompt}")
                printed_prompt_steps += 1

            total_frames = int(args.num_history_frames + model_future_frames)
            # Keep this call shape aligned with NavSIM/nuScenes so checkpoint
            # behavior is comparable across datasets.
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
                    trajectory_len=model_future_frames,
                    ego_vel=ego_vel,
                    history_positions=history_positions,
                    seed=args.seed + int(idx),
                    rand_device=str(device),
                    tiled=True,
                    progress_bar_cmd=tqdm if args.show_denoise_progress else _no_progress_bar,
                    output_mode=infer_output_mode,
                )

            pred_video = out
            traj_pred = None
            if infer_output_mode == "trajectory":
                pred_video = []
                traj_pred = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
            elif infer_output_mode == "video":
                pred_video = out
                traj_pred = None
            elif isinstance(out, (list, tuple)) and len(out) >= 2:
                pred_video = out[0]
                traj_pred = out[1]

            pred_traj = None
            if traj_pred is not None:
                if torch.is_tensor(traj_pred):
                    pred_traj = traj_pred.detach().float().cpu().numpy()
                elif isinstance(traj_pred, (list, tuple)) and len(traj_pred) > 0 and torch.is_tensor(traj_pred[0]):
                    pred_traj = traj_pred[0].detach().float().cpu().numpy()
                else:
                    pred_traj = np.asarray(traj_pred, dtype=np.float32)
                if pred_traj.ndim == 3:
                    pred_traj = pred_traj[0]
            if pred_traj is None:
                pred_traj = np.zeros((model_future_frames, 3), dtype=np.float32)

            pred_traj = _safe_traj_np(pred_traj, model_future_frames)
            gt_traj_for_metric = _safe_traj_np(gt_future_traj, model_future_frames)
            if planning_metric is not None:
                planning_labels = _extract_b2d_planning_labels(
                    dataset,
                    dataset_idx=int(idx),
                    future_frames=int(model_future_frames),
                    sample_interval=planning_sample_interval,
                )
                planning_metric_dict = _migrated_compute_planning_metrics_for_scene(
                    pred_xyh=pred_traj,
                    gt_xyh=gt_traj_for_metric,
                    planning_labels=planning_labels,
                    planning_metric=planning_metric,
                    target_fps=int(args.target_fps),
                )
                row.update(planning_metric_dict)
            else:
                row.update(_empty_planning_metric_dict())


            row["cmd_gt"] = gt_cmd_text

            row["hist_frames_gt"] = int(len(history_video))
            row["future_frames_gt"] = int(len(gt_future_video))
            row["pred_frames"] = int(len(pred_video) if isinstance(pred_video, list) else 0)
            row["hist_frames_ok"] = bool(len(history_video) == args.num_history_frames)
            row["future_frames_ok"] = bool(len(gt_future_video) == args.num_future_frames)
            row["pred_frames_ok"] = bool(row["pred_frames"] == total_frames)
            row["speed_mps"] = float(np.linalg.norm(ego_vel[:2]))

            viz_index = viz_index_by_eval_idx.get(int(idx))
            viz_index_prefix = format_viz_index_prefix(viz_index)

            debug_camera_for_projection: Optional[Dict[str, Any]] = None
            if rank == 0 and projection_debug_token and token == projection_debug_token:
                debug_camera_for_projection = _build_b2d_camera_for_projection(
                    sample,
                    image_h=int(args.height),
                    image_w=int(args.width),
                    history_traj=history_positions,
                    gt_future_traj=gt_traj_for_metric,
                )
                if debug_camera_for_projection is not None:
                    chain_debug_path = _dump_projection_chain_debug(
                        out_dir=out_dir,
                        token=(f"{viz_index_prefix}{token}" if (args.save_viz and int(idx) in global_viz_indices) else token),
                        rank=int(rank),
                        image_h=int(args.height),
                        image_w=int(args.width),
                        camera_for_projection=debug_camera_for_projection,
                        history_traj=history_positions,
                        gt_future_traj=gt_traj_for_metric,
                        pred_future_traj=pred_traj,
                        max_points=max(1, int(args.projection_debug_max_points)),
                    )
                    if chain_debug_path is not None:
                        row["viz_projection_chain_debug_path"] = str(chain_debug_path)
                else:
                    print(
                        "[infer_b2d][proj_chain][warn]",
                        f"token={token}",
                        "failed to build camera_for_projection",
                    )

            if args.save_viz and int(idx) in global_viz_indices:
                viz_path = viz_dir / f"{viz_index_prefix}{token}_rank{rank}.mp4"
                if viz_index is not None:
                    row["viz_index"] = int(viz_index)
                projection_image_path = None
                camera_for_projection = None
                if bool(args.save_projected_traj_image):
                    camera_for_projection = debug_camera_for_projection
                    if camera_for_projection is None:
                        camera_for_projection = _build_b2d_camera_for_projection(
                            sample,
                            image_h=int(args.height),
                            image_w=int(args.width),
                            history_traj=history_positions,
                            gt_future_traj=gt_traj_for_metric,
                        )
                    if camera_for_projection is not None:
                        projection_dir = viz_dir / "projected_traj"
                        projection_dir.mkdir(parents=True, exist_ok=True)
                        projection_image_path = projection_dir / f"{viz_index_prefix}{token}_rank{rank}.png"
                        dbg_full = camera_for_projection.get("_debug_projection")
                        dbg_best = camera_for_projection.get("_debug_best")
                        if dbg_best is not None:
                            row["viz_projection_debug"] = json.dumps(dbg_best, ensure_ascii=False)
                        if (
                            rank == 0
                            and int(args.projection_debug_steps) > 0
                            and printed_projection_debug < int(args.projection_debug_steps)
                            and isinstance(dbg_full, dict)
                        ):
                            printed_projection_debug += 1
                            print(
                                f"[infer_b2d][proj_debug][{printed_projection_debug}/{int(args.projection_debug_steps)}] "
                                f"token={token} best={json.dumps(dbg_full.get('best', {}), ensure_ascii=False)} "
                                f"top={json.dumps(dbg_full.get('top', []), ensure_ascii=False)}"
                            )
                score_dict = {
                    "plan_obj_box_col_1s": row.get("plan_obj_box_col_1s", np.nan),
                    "plan_obj_box_col_2s": row.get("plan_obj_box_col_2s", np.nan),
                    "plan_obj_box_col_3s": row.get("plan_obj_box_col_3s", np.nan),
                    "plan_obj_box_col_avg": row.get("plan_obj_box_col_avg", np.nan),
                }
                extra_info = {
                    "nav_cmd": f"gt={gt_cmd_text}",
                    "speed_mps": row["speed_mps"],
                }
                save_viz_video(
                    out_path=viz_path,
                    gt_video=gt_video,
                    pred_video=pred_video,
                    history_traj=history_positions,
                    gt_future_traj=gt_traj_for_metric,
                    pred_future_traj=pred_traj,
                    width=args.width,
                    height=args.height,
                    plot_height=args.viz_plot_height,
                    fps=max(1, int(args.target_fps)),
                    score_dict=score_dict,
                    extra_info=extra_info,
                    camera_for_projection=camera_for_projection,
                    num_history_frames=int(args.num_history_frames),
                    projection_image_path=projection_image_path,
                    projection_overlay_on_video=False,
                )
                row["viz_path"] = str(viz_path)
                if projection_image_path is not None:
                    row["viz_projection_path"] = str(projection_image_path)
                viz_saved += 1
        except Exception as exc:
            row["valid"] = False
            row["error"] = str(exc)
            if args.print_errors:
                print(f"[infer_b2d][error] idx={idx} err={exc}", flush=True)
        results.append(row)

    if rank == 0 and hasattr(iterator, "close"):
        iterator.close()

    final_results = _gather_results(results, device=device, rank=rank, world_size=world_size)
    if rank == 0 and final_results is not None:
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        csv_path = out_dir / f"bench2drive_infer_{timestamp}.csv"
        _write_csv(csv_path, final_results)

        valid_rows = [r for r in final_results if bool(r.get("valid", False))]
        summary = {
            "num_total": int(len(final_results)),
            "num_valid": int(len(valid_rows)),
            "num_failed": int(len(final_results) - len(valid_rows)),
        }
        planning_rows = [r for r in valid_rows if bool(r.get("planning_metrics_available", False))]
        summary["planning_num_valid"] = int(len(planning_rows))
        for key in (
            "plan_L2_1s",
            "plan_L2_2s",
            "plan_L2_3s",
            "plan_L2_avg",
            "plan_obj_box_col_1s",
            "plan_obj_box_col_2s",
            "plan_obj_box_col_3s",
            "plan_obj_box_col_avg",
        ):
            summary[key] = (
                float(np.nanmean([r.get(key, np.nan) for r in planning_rows]))
                if planning_rows
                else float("nan")
            )
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        l2_parts: List[str] = []
        for sec in (1, 2, 3):
            key = f"plan_L2_{sec}s"
            val = summary.get(key, float("nan"))
            if not np.isnan(val):
                l2_parts.append(f"{sec}s={float(val):.4f}")
        l2_avg = summary.get("plan_L2_avg", float("nan"))
        if not np.isnan(l2_avg):
            l2_parts.append(f"avg={float(l2_avg):.4f}")
        if l2_parts:
            print("[infer_b2d][avg] Planning L2 (m): " + ", ".join(l2_parts))
        box_col_parts: List[str] = []
        for sec in (1, 2, 3):
            key = f"plan_obj_box_col_{sec}s"
            val = summary.get(key, float("nan"))
            if not np.isnan(val):
                box_col_parts.append(f"{sec}s={float(val):.4f}")
        box_col_avg = summary.get("plan_obj_box_col_avg", float("nan"))
        if not np.isnan(box_col_avg):
            box_col_parts.append(f"avg={float(box_col_avg):.4f}")
        if box_col_parts:
            print("[infer_b2d][avg] Planning box collision: " + ", ".join(box_col_parts))

        print(
            "[infer_b2d] done:",
            f"total={summary['num_total']}",
            f"valid={summary['num_valid']}",
            f"failed={summary['num_failed']}",
            f"planning_valid={summary['planning_num_valid']}",
            f"csv={csv_path}",
            f"viz_dir={viz_dir if args.save_viz else 'disabled'}",
        )


def main() -> None:
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
