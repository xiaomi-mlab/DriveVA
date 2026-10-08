"""
NAVSIM v1 dataset adapter for UNIVERSE training.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

try:
    from examples.wanvideo.UNIVERSE_infer.navsim_dataset import (
        DEFAULT_NEGATIVE_PROMPT,
        FocalUnificationConfig,
        _build_prompt_fixed,
        _ensure_navsim_importable,
        _intrinsics_at,
        _resolve_video_pil,
        focal_stats_record,
        summarize_focal_stats,
    )
except ImportError:
    from navsim_dataset import (  # type: ignore
        DEFAULT_NEGATIVE_PROMPT,
        FocalUnificationConfig,
        _build_prompt_fixed,
        _ensure_navsim_importable,
        _intrinsics_at,
        _resolve_video_pil,
        focal_stats_record,
        summarize_focal_stats,
    )


@dataclass
class NavsimUNIVERSEConfig:
    repo_root: str
    navsim_log_path: str
    sensor_blobs_path: str
    cache_path: Optional[str] = None
    use_cache_only: bool = False
    force_cache_computation: bool = False
    num_history_frames: int = 5
    num_future_frames: int = 8
    frame_interval: Optional[int] = None
    has_route: bool = True
    max_scenes: Optional[int] = None
    train_log_names: Optional[List[str]] = None
    val_log_names: Optional[List[str]] = None
    image_height: int = 480
    image_width: int = 832
    image_normalize: str = "[-1,1]"
    surround_view: bool = False
    use_focal_unification: bool = False
    focal_target_px: Optional[Any] = None
    focal_stats_samples: int = 0
    skip_missing_files: bool = False
    quiet_scene_loader: bool = True


def _fit_len_2d(arr: Any, expected_len: int, channels: int) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.ndim != 2:
        arr_np = np.zeros((expected_len, channels), dtype=np.float32)
    if arr_np.shape[1] < channels:
        pad = np.zeros((arr_np.shape[0], channels - arr_np.shape[1]), dtype=np.float32)
        arr_np = np.concatenate([arr_np, pad], axis=1)
    if arr_np.shape[1] > channels:
        arr_np = arr_np[:, :channels]
    if arr_np.shape[0] == expected_len:
        return arr_np.astype(np.float32)
    if arr_np.shape[0] > expected_len:
        return arr_np[:expected_len].astype(np.float32)
    if arr_np.shape[0] == 0:
        return np.zeros((expected_len, channels), dtype=np.float32)
    pad = np.repeat(arr_np[-1:], expected_len - arr_np.shape[0], axis=0)
    return np.concatenate([arr_np, pad], axis=0).astype(np.float32)


def _normalize_traj_xyh(traj: Any, expected_len: int) -> np.ndarray:
    traj_np = np.asarray(traj, dtype=np.float32)
    if traj_np.ndim != 2:
        return np.zeros((expected_len, 3), dtype=np.float32)
    if (
        traj_np.shape[0] == expected_len + 1
        and traj_np.shape[1] >= 2
        and float(np.linalg.norm(traj_np[0, :2])) < 1e-4
    ):
        traj_np = traj_np[1:]
    return _fit_len_2d(traj_np, expected_len=expected_len, channels=3)


def _normalize_history_prefix(history: Any, num_history_frames: int) -> np.ndarray:
    full_len = max(1, int(num_history_frames))
    prefix_len = max(1, full_len - 1)
    hist_np = np.asarray(history, dtype=np.float32)
    if hist_np.ndim != 2:
        return np.zeros((prefix_len, 3), dtype=np.float32)
    hist_np = _fit_len_2d(hist_np, expected_len=hist_np.shape[0], channels=3)
    if hist_np.shape[0] == full_len and full_len > 1:
        hist_np = hist_np[:-1]
    return _fit_len_2d(hist_np, expected_len=prefix_len, channels=3)


def _normalize_ego_vel_planar(vel_raw: Any) -> np.ndarray:
    vel = np.zeros((3,), dtype=np.float32)
    if vel_raw is None:
        return vel
    vel_np = np.asarray(vel_raw, dtype=np.float32).reshape(-1)
    if vel_np.size >= 1:
        vel[0] = float(vel_np[0])
    if vel_np.size >= 2:
        vel[1] = float(vel_np[1])
    return vel


def _first_present(mapping: Dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _quiet_stdout(enabled: bool):
    if enabled:
        return contextlib.redirect_stdout(io.StringIO())
    return contextlib.nullcontext()


class NavsimUNIVERSEDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: NavsimUNIVERSEConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.load_from_cache = False

        repo_root = Path(cfg.repo_root).resolve()
        _ensure_navsim_importable(repo_root)

        from navsim.common.dataclasses import SceneFilter, SensorConfig
        from navsim.common.dataloader import SceneLoader
        from navsim.planning.training.dataset import CacheOnlyDataset, Dataset as NavDataset
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

        try:
            from navsim.agents.videodrive.videodrive_features import (
                TrajectoryTargetBuilder,
                VideoDriveFeatureBuilder,
            )
        except Exception:
            from examples.wanvideo.UNIVERSE_infer.eval_navsim_pdm import (
                _CompatUNIVERSEFeatureBuilder as VideoDriveFeatureBuilder,
                _CompatTrajectoryTargetBuilder as TrajectoryTargetBuilder,
            )

        log_names = cfg.train_log_names if split == "train" else cfg.val_log_names
        scene_filter = SceneFilter(
            num_history_frames=int(cfg.num_history_frames),
            num_future_frames=int(cfg.num_future_frames),
            frame_interval=cfg.frame_interval,
            has_route=bool(cfg.has_route),
            max_scenes=cfg.max_scenes,
            log_names=log_names,
        )
        self._scene_loader = SceneLoader(
            data_path=Path(cfg.navsim_log_path),
            sensor_blobs_path=Path(cfg.sensor_blobs_path),
            scene_filter=scene_filter,
            sensor_config=SensorConfig.build_all_sensors(include=True),
            load_image_path=True,
        )
        view_mode = "surround6" if cfg.surround_view else "front"
        feature_builders = [
            VideoDriveFeatureBuilder(
                prompt_frames=int(cfg.num_history_frames),
                normalize=cfg.image_normalize,
                view_mode=view_mode,
            )
        ]
        target_builders = [
            TrajectoryTargetBuilder(
                trajectory_sampling=TrajectorySampling(num_poses=int(cfg.num_future_frames), interval_length=0.5),
                normalize=cfg.image_normalize,
                view_mode=view_mode,
            )
        ]

        if cfg.use_cache_only:
            if not cfg.cache_path:
                raise ValueError("cache_path must be set when use_cache_only=True")
            self._dataset = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=feature_builders,
                target_builders=target_builders,
                log_names=log_names,
            )
            self._cache_only = True
        else:
            self._dataset = NavDataset(
                scene_loader=self._scene_loader,
                feature_builders=feature_builders,
                target_builders=target_builders,
                cache_path=cfg.cache_path,
                force_cache_computation=cfg.force_cache_computation,
                is_decoder=False,
            )
            self._cache_only = False

    def __len__(self) -> int:
        return len(self._dataset)

    def _sample_token(self, idx: int, item: Any) -> str:
        if self._cache_only and isinstance(item, tuple) and len(item) >= 3:
            return str(item[2])
        try:
            return str(self._scene_loader.tokens[idx])
        except Exception:
            return str(idx)

    def set_focal_target_px(self, target_focal_px: Any) -> None:
        self.cfg.focal_target_px = target_focal_px

    def _focal_config(self) -> FocalUnificationConfig:
        return FocalUnificationConfig.build(
            enabled=bool(self.cfg.use_focal_unification) and not bool(self.cfg.surround_view),
            target_focal_px=self.cfg.focal_target_px,
            output_width=int(self.cfg.image_width),
            output_height=int(self.cfg.image_height),
        )

    def _frame_image_size_wh(self, frame: Any) -> Optional[tuple[int, int]]:
        if isinstance(frame, (list, tuple)):
            if len(frame) == 0:
                return None
            frame = frame[0]
        if isinstance(frame, (str, Path)):
            with Image.open(frame) as img:
                return (int(img.size[0]), int(img.size[1]))
        if isinstance(frame, Image.Image):
            return (int(frame.size[0]), int(frame.size[1]))
        if torch.is_tensor(frame):
            shape = tuple(int(v) for v in frame.shape)
            if len(shape) == 3:
                if shape[0] in (1, 3):
                    return (shape[2], shape[1])
                return (shape[1], shape[0])
        if isinstance(frame, np.ndarray):
            if frame.ndim >= 2:
                return (int(frame.shape[1]), int(frame.shape[0]))
        return None

    def _collect_focal_records_from_frames(self, frames: Any, camera_intrinsics: Any) -> List[Dict[str, float]]:
        records: List[Dict[str, float]] = []
        if frames is None:
            return records
        if torch.is_tensor(frames):
            frame_iter = [frames[i] for i in range(int(frames.shape[0]))] if frames.ndim >= 4 else [frames]
        else:
            frame_iter = list(frames)
        for frame_idx, frame in enumerate(frame_iter):
            image_size = self._frame_image_size_wh(frame)
            if image_size is None:
                continue
            record = focal_stats_record(
                image_size,
                _intrinsics_at(camera_intrinsics, frame_idx),
                output_width=int(self.cfg.image_width),
                output_height=int(self.cfg.image_height),
            )
            if record is not None:
                records.append(record)
        return records

    def collect_focal_stats(self, max_samples: int) -> Dict[str, float]:
        if bool(self.cfg.surround_view):
            return {"count": 0.0}
        records: List[Dict[str, float]] = []
        limit = min(max(0, int(max_samples)), len(self))
        for idx in range(limit):
            try:
                with _quiet_stdout(bool(self.cfg.quiet_scene_loader)):
                    item = self._dataset[idx]
                if self._cache_only:
                    features, targets, _ = item
                else:
                    features, targets = item
                history_frames = features.get("image_paths") or features.get("images")
                future_frames = targets.get("future_image_paths") or targets.get("future_frames")
                records.extend(
                    self._collect_focal_records_from_frames(history_frames, features.get("camera_intrinsics"))
                )
                records.extend(
                    self._collect_focal_records_from_frames(future_frames, targets.get("future_camera_intrinsics"))
                )
            except Exception as exc:
                if not self.cfg.skip_missing_files:
                    raise
                print(f"[train][focal][warn] skip stats idx={idx}: {exc}")
        return summarize_focal_stats(records)

    def _build_sample(self, idx: int) -> Dict[str, Any]:
        with _quiet_stdout(bool(self.cfg.quiet_scene_loader)):
            item = self._dataset[idx]
        if self._cache_only:
            features, targets, _ = item
        else:
            features, targets = item
        sample_token = self._sample_token(idx, item)

        history_frames = features.get("image_paths") or features.get("images")
        future_frames = targets.get("future_image_paths") or targets.get("future_frames")
        if history_frames is None or future_frames is None:
            raise ValueError("Missing history or future frames in NavSIM features/targets.")

        video_hist = _resolve_video_pil(
            history_frames,
            height=int(self.cfg.image_height),
            width=int(self.cfg.image_width),
            surround_view=bool(self.cfg.surround_view),
            normalize_mode=self.cfg.image_normalize,
            camera_intrinsics=features.get("camera_intrinsics"),
            focal_config=self._focal_config(),
        )
        video_fut = _resolve_video_pil(
            future_frames,
            height=int(self.cfg.image_height),
            width=int(self.cfg.image_width),
            surround_view=bool(self.cfg.surround_view),
            normalize_mode=self.cfg.image_normalize,
            camera_intrinsics=targets.get("future_camera_intrinsics"),
            focal_config=self._focal_config(),
        )
        video = video_hist + video_fut

        trajectory = _normalize_traj_xyh(targets.get("trajectory"), expected_len=int(self.cfg.num_future_frames))
        history_positions = _normalize_history_prefix(
            features.get("history_trajectory"),
            num_history_frames=int(self.cfg.num_history_frames),
        )
        ego_vel = _normalize_ego_vel_planar(features.get("vel"))
        speed = float(np.linalg.norm(ego_vel[:2]))
        driving_command = _first_present(features, ("driving_command", "command"))
        prompt = _build_prompt_fixed(history_positions, driving_command, speed, None)

        return {
            "source": "navsim",
            "token": sample_token,
            "prompt": prompt,
            "video": video,
            "longcat_video": video_hist,
            "num_frames": len(video),
            "trajectory": torch.as_tensor(trajectory, dtype=torch.float32),
            "history_positions": torch.as_tensor(history_positions, dtype=torch.float32),
            "ego_vel": torch.as_tensor(ego_vel, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self.cfg.skip_missing_files:
            return self._build_sample(idx)

        last_error: Optional[Exception] = None
        for offset in range(len(self)):
            sample_idx = (idx + offset) % len(self)
            try:
                return self._build_sample(sample_idx)
            except (FileNotFoundError, ValueError, OSError) as exc:
                last_error = exc
                print(f"[train][dataset][warn] skip idx={sample_idx}: {exc}")
        raise RuntimeError("No valid NavSIM sample found after skip_missing_files scan.") from last_error


NavsimDriveLawConfig = NavsimUNIVERSEConfig
NavsimDriveLawDataset = NavsimUNIVERSEDataset
