"""
Runtime helpers shared by UNIVERSE NavSIM and nuScenes inference scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


NAV_CMDS = ["turn left", "go straight", "turn right"]
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, "
    "flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, "
    "temporal inconsistency, jerky motion, choppy framerate"
)


@dataclass
class FocalUnificationConfig:
    enabled: bool = False
    target_focal_px: Optional[float] = None
    output_width: int = 832
    output_height: int = 480
    pad_color: Tuple[int, int, int] = (0, 0, 0)

    @classmethod
    def build(
        cls,
        *,
        enabled: bool,
        target_focal_px: Any,
        output_width: int,
        output_height: int,
    ) -> "FocalUnificationConfig":
        target = _parse_target_focal_px(target_focal_px)
        return cls(
            enabled=bool(enabled) and target is not None and float(target) > 0.0,
            target_focal_px=target,
            output_width=int(output_width),
            output_height=int(output_height),
        )


def _purge_imported_navsim_modules() -> None:
    to_delete = [
        module_name
        for module_name in list(sys.modules.keys())
        if module_name == "navsim" or module_name.startswith("navsim.")
    ]
    for module_name in to_delete:
        del sys.modules[module_name]


def _ensure_navsim_importable(repo_root: Path) -> None:
    candidates = [
        repo_root / "third_party",
        Path(__file__).resolve().parents[3] / "third_party",
    ]

    candidates = list(dict.fromkeys(candidates))

    required_file = Path("navsim") / "common" / "dataclasses.py"

    for navsim_root in candidates:
        if navsim_root.exists() and (navsim_root / required_file).exists():
            selected_root = navsim_root.resolve()
            selected_root_str = str(selected_root)
            resolved_candidates = [p.resolve() for p in candidates if p.exists()]

            for candidate_root in resolved_candidates:
                candidate_root_str = str(candidate_root)
                while candidate_root_str in sys.path:
                    sys.path.remove(candidate_root_str)
            sys.path.insert(0, selected_root_str)

            loaded_navsim = sys.modules.get("navsim")
            loaded_root: Optional[Path] = None
            if loaded_navsim is not None:
                loaded_file = getattr(loaded_navsim, "__file__", None)
                if loaded_file:
                    try:
                        loaded_root = Path(str(loaded_file)).resolve().parents[1]
                    except Exception:
                        loaded_root = None
            if loaded_navsim is not None and loaded_root != selected_root:
                _purge_imported_navsim_modules()
            return

    raise FileNotFoundError(
        "Cannot locate a NavSIM package "
        "(expected navsim/common/dataclasses.py). Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def _read_image_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _parse_target_focal_px(value: Any) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "" or raw.lower() in {"none", "null", "auto"}:
        return None
    target = float(raw)
    if target <= 0.0:
        return None
    return target


def _coerce_intrinsics(intrinsics: Any) -> Optional[np.ndarray]:
    if intrinsics is None:
        return None
    try:
        arr = np.asarray(intrinsics, dtype=np.float32)
    except Exception:
        return None
    if arr.shape == (3, 3):
        k = arr.copy()
    elif arr.ndim == 1 and arr.size >= 4:
        fx, fy, cx, cy = [float(v) for v in arr[:4]]
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    else:
        return None
    if not np.isfinite(k).all() or float(k[0, 0]) <= 0.0 or float(k[1, 1]) <= 0.0:
        return None
    return k


def _intrinsics_at(camera_intrinsics: Any, index: int) -> Any:
    if camera_intrinsics is None:
        return None
    if torch.is_tensor(camera_intrinsics):
        if camera_intrinsics.ndim >= 3 and index < int(camera_intrinsics.shape[0]):
            return camera_intrinsics[index].detach().cpu().numpy()
        return camera_intrinsics.detach().cpu().numpy()
    if isinstance(camera_intrinsics, np.ndarray):
        if camera_intrinsics.ndim >= 3 and index < int(camera_intrinsics.shape[0]):
            return camera_intrinsics[index]
        return camera_intrinsics
    if isinstance(camera_intrinsics, Sequence) and not isinstance(camera_intrinsics, (str, bytes)):
        if len(camera_intrinsics) == 0:
            return None
        if index < len(camera_intrinsics):
            return camera_intrinsics[index]
        return camera_intrinsics[-1]
    return camera_intrinsics


def focal_stats_record(
    image_size_wh: Tuple[int, int],
    intrinsics: Any,
    *,
    output_width: int,
    output_height: int,
) -> Optional[Dict[str, float]]:
    k = _coerce_intrinsics(intrinsics)
    if k is None:
        return None
    src_w = max(1, int(image_size_wh[0]))
    src_h = max(1, int(image_size_wh[1]))
    out_w = max(1, int(output_width))
    out_h = max(1, int(output_height))
    raw_fx = float(k[0, 0])
    raw_fy = float(k[1, 1])
    eff_fx = raw_fx * float(out_w) / float(src_w)
    eff_fy = raw_fy * float(out_h) / float(src_h)
    return {
        "raw_fx": raw_fx,
        "raw_fy": raw_fy,
        "raw_focal_mean": 0.5 * (raw_fx + raw_fy),
        "effective_fx": eff_fx,
        "effective_fy": eff_fy,
        "effective_focal_mean": 0.5 * (eff_fx + eff_fy),
        "src_width": float(src_w),
        "src_height": float(src_h),
    }


def summarize_focal_stats(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if len(records) == 0:
        return {"count": 0.0}
    out: Dict[str, float] = {"count": float(len(records))}
    for key in ("raw_fx", "raw_fy", "raw_focal_mean", "effective_fx", "effective_fy", "effective_focal_mean"):
        values = np.asarray([float(r[key]) for r in records if key in r and np.isfinite(float(r[key]))], dtype=np.float64)
        if values.size == 0:
            continue
        out[f"{key}_min"] = float(np.min(values))
        out[f"{key}_p25"] = float(np.percentile(values, 25))
        out[f"{key}_median"] = float(np.median(values))
        out[f"{key}_p75"] = float(np.percentile(values, 75))
        out[f"{key}_max"] = float(np.max(values))
    return out


def _resize_to_output_canvas(
    image: Image.Image,
    *,
    width: int,
    height: int,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if image.size == (int(width), int(height)):
        return image.convert("RGB")
    return image.convert("RGB").resize((int(width), int(height)), Image.BILINEAR)


def focal_unify_image(
    image: Image.Image,
    intrinsics: Any,
    *,
    target_focal_px: float,
    output_width: int,
    output_height: int,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Optional[np.ndarray], Dict[str, Any]]:
    k = _coerce_intrinsics(intrinsics)
    if k is None:
        resized = _resize_to_output_canvas(image, width=output_width, height=output_height, pad_color=pad_color)
        return resized, None, {"applied": False, "reason": "missing_intrinsics"}

    src_w, src_h = image.size
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    target = float(target_focal_px)
    if target <= 0.0 or src_w <= 0 or src_h <= 0:
        resized = _resize_to_output_canvas(image, width=output_width, height=output_height, pad_color=pad_color)
        return resized, None, {"applied": False, "reason": "invalid_target_or_size"}

    scale_x = target / fx
    scale_y = target / fy
    scaled_w = max(1, int(round(float(src_w) * scale_x)))
    scaled_h = max(1, int(round(float(src_h) * scale_y)))
    resized = image.convert("RGB").resize((scaled_w, scaled_h), Image.BILINEAR)

    k_out = k.astype(np.float32, copy=True)
    k_out[0, 0] *= float(scaled_w) / float(src_w)
    k_out[1, 1] *= float(scaled_h) / float(src_h)
    k_out[0, 2] *= float(scaled_w) / float(src_w)
    k_out[1, 2] *= float(scaled_h) / float(src_h)

    crop_left = max(0, (scaled_w - int(output_width)) // 2)
    crop_top = max(0, (scaled_h - int(output_height)) // 2)
    crop_right = crop_left + min(int(output_width), scaled_w)
    crop_bottom = crop_top + min(int(output_height), scaled_h)
    cropped = resized.crop((crop_left, crop_top, crop_right, crop_bottom))

    pad_left = max(0, (int(output_width) - cropped.size[0]) // 2)
    pad_top = max(0, (int(output_height) - cropped.size[1]) // 2)
    canvas = Image.new("RGB", (int(output_width), int(output_height)), pad_color)
    canvas.paste(cropped, (pad_left, pad_top))

    k_out[0, 2] = k_out[0, 2] - float(crop_left) + float(pad_left)
    k_out[1, 2] = k_out[1, 2] - float(crop_top) + float(pad_top)

    meta = {
        "applied": True,
        "target_focal_px": target,
        "src_size": (int(src_w), int(src_h)),
        "scaled_size": (int(scaled_w), int(scaled_h)),
        "output_size": (int(output_width), int(output_height)),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "fx_out": float(k_out[0, 0]),
        "fy_out": float(k_out[1, 1]),
    }
    return canvas, k_out, meta


def preprocess_camera_image(
    image: Image.Image,
    *,
    width: int,
    height: int,
    intrinsics: Any = None,
    focal_config: Optional[FocalUnificationConfig] = None,
) -> Tuple[Image.Image, Optional[np.ndarray], Dict[str, Any]]:
    if focal_config is not None and focal_config.enabled and focal_config.target_focal_px is not None:
        return focal_unify_image(
            image,
            intrinsics,
            target_focal_px=float(focal_config.target_focal_px),
            output_width=int(focal_config.output_width),
            output_height=int(focal_config.output_height),
            pad_color=focal_config.pad_color,
        )
    resized = _resize_to_output_canvas(image, width=width, height=height)
    return resized, None, {"applied": False, "reason": "disabled"}


def _mosaic_surround(paths: List[str | Path]) -> Image.Image:
    imgs = [_read_image_rgb(p) for p in paths]
    if len(imgs) == 0:
        raise ValueError("surround-view frame list is empty")
    base_w, base_h = imgs[0].size
    imgs = [img.resize((base_w, base_h), Image.BILINEAR) for img in imgs]
    while len(imgs) < 6:
        imgs.append(Image.new("RGB", (base_w, base_h), (0, 0, 0)))
    row1 = Image.new("RGB", (base_w * 3, base_h), (0, 0, 0))
    row2 = Image.new("RGB", (base_w * 3, base_h), (0, 0, 0))
    for idx, img in enumerate(imgs[:3]):
        row1.paste(img, (idx * base_w, 0))
    for idx, img in enumerate(imgs[3:6]):
        row2.paste(img, (idx * base_w, 0))
    mosaic = Image.new("RGB", (base_w * 3, base_h * 2), (0, 0, 0))
    mosaic.paste(row1, (0, 0))
    mosaic.paste(row2, (0, base_h))
    return mosaic


def _tensor_to_pil(frame: torch.Tensor, normalize_mode: Optional[str]) -> Image.Image:
    arr = frame.detach().cpu().float()
    if arr.ndim == 3:
        arr = arr.permute(1, 2, 0)
    if normalize_mode == "[0,1]":
        arr = arr * 255.0
    elif normalize_mode == "[-1,1]":
        arr = (arr + 1.0) * 127.5
    else:
        vmin = float(arr.min().item())
        vmax = float(arr.max().item())
        arr = arr * 255.0 if vmin >= -0.1 and vmax <= 1.1 else (arr + 1.0) * 127.5
    arr = arr.clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(arr).convert("RGB")


def _resolve_video_pil(
    frames: List[Any],
    height: int,
    width: int,
    surround_view: bool,
    normalize_mode: Optional[str] = None,
    camera_intrinsics: Any = None,
    focal_config: Optional[FocalUnificationConfig] = None,
) -> List[Image.Image]:
    out: List[Image.Image] = []
    for frame_idx, frame in enumerate(frames):
        if isinstance(frame, (list, tuple)):
            img = _mosaic_surround(list(frame)) if surround_view else _read_image_rgb(frame[0])
        elif isinstance(frame, (str, Path)):
            img = _read_image_rgb(frame)
        elif torch.is_tensor(frame):
            img = _tensor_to_pil(frame, normalize_mode)
        elif isinstance(frame, np.ndarray):
            img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
        elif isinstance(frame, Image.Image):
            img = frame.convert("RGB")
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
        frame_intrinsics = None if surround_view else _intrinsics_at(camera_intrinsics, frame_idx)
        img, _, _ = preprocess_camera_image(
            img,
            width=int(width),
            height=int(height),
            intrinsics=frame_intrinsics,
            focal_config=focal_config,
        )
        out.append(img)
    return out


def one_hot_to_cmd(one_hot: Any) -> str:
    if torch.is_tensor(one_hot):
        values = one_hot.detach().cpu().flatten().tolist()
    elif isinstance(one_hot, np.ndarray):
        values = one_hot.flatten().tolist()
    else:
        values = list(one_hot)
    for idx, value in enumerate(values[: len(NAV_CMDS)]):
        if int(value) == 1:
            return NAV_CMDS[idx]
    return "unknown"


def _build_prompt_fixed(*args: Any) -> str:
    """
    Build the inference prompt from driving command and ego dynamics.

    Supported signatures:
    - _build_prompt_fixed(cmd_onehot, speed_mps)
    - _build_prompt_fixed(history_xyh, cmd_onehot, speed_mps, accel_mps2)
    """
    accel_mps2: Optional[float] = None
    if len(args) == 2:
        cmd_onehot, speed_mps = args
    elif len(args) == 4:
        _, cmd_onehot, speed_mps, accel_mps2 = args
    else:
        raise TypeError(
            "_build_prompt_fixed expects (cmd_onehot, speed_mps) or "
            "(history_xyh, cmd_onehot, speed_mps, accel_mps2)"
        )

    cmd = one_hot_to_cmd(cmd_onehot).lower()
    speed_mps = float(speed_mps)
    accel_mps2 = None if accel_mps2 is None else float(accel_mps2)

    if speed_mps < 5.0:
        speed_desc = "at low speed"
    elif speed_mps < 15.0:
        speed_desc = "at moderate speed"
    else:
        speed_desc = "at highway speed"

    if "left" in cmd:
        motion_trend, turning_desc = "turning left", "with controlled steering"
    elif "right" in cmd:
        motion_trend, turning_desc = "turning right", "with controlled steering"
    elif "straight" in cmd:
        motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"
    else:
        motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"

    technical = f"[Technical: speed {speed_mps:.2f}m/s"
    if accel_mps2 is not None:
        technical += f", accel {accel_mps2:.2f}m/s^2"
    technical += "]"

    return (
        "A high-quality, photorealistic dashboard camera view of autonomous driving. "
        f"Based on the past 2 seconds video showing {motion_trend} {turning_desc}, "
        "predict and generate the next 4 seconds of realistic driving continuation, "
        f"following command: {cmd}. "
        f"Keep temporal consistency, realistic physics, and smooth motion, moving {speed_desc}. "
        f"{technical}"
    )
