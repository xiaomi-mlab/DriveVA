"""
Bench2Drive dataset adapter for DiffSynth-Studio.
Loads Bench2Drive pkl annotations for Wan video generation and planning evaluation.
"""

import torch
import numpy as np
import pickle
import os
from PIL import Image
from typing import Dict, List, Optional, Tuple
# from scipy.spatial.transform import Rotation as R # 移除 Scipy 依赖

import cv2
from concurrent.futures import ThreadPoolExecutor

try:
    from .navsim_dataset import FocalUnificationConfig, preprocess_camera_image
except ImportError:
    from navsim_dataset import FocalUnificationConfig, preprocess_camera_image


class ImageCropAndResize:
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize((int(self.width), int(self.height)), Image.BILINEAR)

# === Numpy 几何运算工具函数 (替代 Scipy) ===
def _mat2quat(m):
    """
    Numpy implementation of rotation matrix to quaternion [x, y, z, w]
    """
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] + m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    return np.array([x, y, z, w], dtype=np.float32)

def _quat2mat(q):
    """
    Numpy implementation of quaternion [x, y, z, w] to rotation matrix
    """
    x, y, z, w = q
    x2 = x * x; y2 = y * y; z2 = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z

    return np.array([
        [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)]
    ], dtype=np.float32)

def _convert_relative_to_vehicle_frame(
    relative_positions: np.ndarray,
    relative_quaternions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert from current frame (X right, Y forward, Z up) to vehicle frame
    (X forward, Y left, Z up).
    """
    r_cur_to_vehicle = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    pos_vehicle = (r_cur_to_vehicle @ relative_positions.T).T

    quats_vehicle = []
    for q in relative_quaternions:
        rot_cur = _quat2mat(q)
        rot_vehicle = r_cur_to_vehicle @ rot_cur @ r_cur_to_vehicle.T
        quats_vehicle.append(_mat2quat(rot_vehicle))

    return pos_vehicle.astype(np.float32), np.array(quats_vehicle, dtype=np.float32)
# ==========================================


def generate_navigation_prompt(info, ego_vel=None, history_positions=None, fps=10):
    """
    根据导航指令和速度生成驾驶场景描述
    导航动作放在最前面，后面补充场景信息

    Args:
        info: 数据帧信息
        ego_vel: 自车速度向量 [vx, vy, vz]
        history_positions: 历史轨迹位置
        fps: 帧率

    Returns:
        str: 场景描述prompt
    """
    # 导航指令 - 直接作为开头
    COMMAND_PREFIX = {
        0: "Driving forward",
        1: "Turning left",
        2: "Turning right",
        3: "Going straight",
        4: "Following the lane",
        5: "Changing to the left lane",
        6: "Changing to the right lane",
    }

    COMMAND_NEXT = {
        1: "then turning left",
        2: "then turning right",
        3: "then going straight",
        5: "then changing left",
        6: "then changing right",
    }

    command_near = info.get('command_near', 4)
    command_far = info.get('command_far', 4)

    # 导航动作开头
    parts = [COMMAND_PREFIX.get(command_near, "Driving forward")]

    # 后续动作
    if command_near != command_far and command_far in COMMAND_NEXT:
        parts[0] += f", {COMMAND_NEXT[command_far]}"

    # 速度信息
    speed_kmh = compute_speed(ego_vel, history_positions, fps)
    parts.append(f"at {speed_kmh:.0f} km/h.")

    # 场景描述
    scene_desc = extract_scene_description(info.get('folder', ''))
    if scene_desc:
        parts.append(scene_desc)

    # 补充描述
    parts.append("Smooth motion, temporally consistent.")

    return " ".join(parts)


def extract_scene_description(folder):
    """从文件夹名提取场景描述"""
    folder_lower = folder.lower()

    descriptions = []

    # 场景类型
    scene_map = {
        'accident': "Accident scene ahead.",
        'parking': "Parking area.",
        'pedestrian': "Pedestrians present.",
        'crossing': "Pedestrian crossing.",
        'junction': "At junction.",
        'signalized': "Traffic light ahead.",
        'highway': "Highway driving.",
        'merge': "Merge zone.",
        'cutin': "Vehicle cutting in.",
        'cutout': "Vehicle cutting out.",
        'slowtraffic': "Slow traffic.",
        'obstacle': "Obstacle ahead.",
        'construction': "Construction zone.",
        'static': "Static obstacles present.",
        'dynamic': "Dynamic objects moving.",
    }

    for keyword, desc in scene_map.items():
        if keyword in folder_lower:
            descriptions.append(desc)
            break

    return " ".join(descriptions)


def compute_speed(ego_vel, history_positions, fps):
    """计算速度（km/h）"""
    if ego_vel is not None:
        speed_mps = np.linalg.norm(np.array(ego_vel)[:2])
    elif history_positions is not None and len(history_positions) >= 2:
        displacements = np.diff(np.array(history_positions)[-3:], axis=0)
        speed_mps = np.mean(np.linalg.norm(displacements[:, :2], axis=1)) * fps
    else:
        speed_mps = 0.0
    return speed_mps * 3.6


def generate_structured_prompt(info):
    """
    生成结构化的Intent Token Prompt
    根据近处(command_near)和远处(command_far)导航点ID及坐标生成多阶段指令

    格式:
    <DRIVE_CMD>
    <INTENT=...>
    <DIRECTION=...>
    <WAYPOINT=x,y>
    <TIME_BIN=T1>
    </DRIVE_CMD>
    """
    # 导航指令映射表
    CMD_MAPPING = {
        0: ("KEEP", "FORWARD"),
        1: ("TURN", "LEFT"),
        2: ("TURN", "RIGHT"),
        3: ("KEEP", "STRAIGHT"),
        4: ("KEEP", "FORWARD"),
        5: ("LANE_CHANGE", "LEFT"),
        6: ("LANE_CHANGE", "RIGHT"),
    }

    # 获取指令ID
    command_near = info.get('command_near', 4)
    command_far = info.get('command_far', 4)

    # 获取导航点坐标 (通常是世界坐标或相对于地图原点的坐标)
    # 格式化为保留2位小数的字符串
    near_xy = info.get('command_near_xy', [0.0, 0.0])
    far_xy = info.get('command_far_xy', [0.0, 0.0])

    near_xy_str = f"{near_xy[0]:.2f},{near_xy[1]:.2f}"
    far_xy_str = f"{far_xy[0]:.2f},{far_xy[1]:.2f}"

    # --- 生成 T1 阶段指令 (基于近处导航点) ---
    intent_1, direction_1 = CMD_MAPPING.get(command_near, ("KEEP", "FORWARD"))

    prompt = (
        f"<DRIVE_CMD>\n"
        f"<INTENT={intent_1}>\n"
        f"<DIRECTION={direction_1}>\n"
        f"<TIME_BIN=T1>\n"
        f"</DRIVE_CMD>"
    )

    # --- 生成 T2 阶段指令 (基于远处导航点) ---
    # 如果远处指令与近处不同，或者坐标距离较远（这里简化为只要存在就输出，或者逻辑上区分T2）
    # 通常 T2 代表更长期的目标
    if command_far != command_near:
        intent_2, direction_2 = CMD_MAPPING.get(command_far, ("KEEP", "FORWARD"))
        prompt += (
            f"\n<DRIVE_CMD>\n"
            f"<INTENT={intent_2}>\n"
            f"<DIRECTION={direction_2}>\n"
            f"<TIME_BIN=T2>\n"
            f"</DRIVE_CMD>"
        )
    else:
        # 即使指令相同，如果坐标不同，也可以作为 T2 阶段的延续目标
        # 检查坐标是否显著不同 (简单的欧氏距离判断，或者直接输出)
        # 这里直接输出作为远期目标
        prompt += (
            f"\n<DRIVE_CMD>\n"
            f"<INTENT={intent_1}>\n" # 延续 T1 的意图
            f"<DIRECTION={direction_1}>\n"
            f"<TIME_BIN=T2>\n"
            f"</DRIVE_CMD>"
        )

    return prompt
class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

class Bench2DriveDataset(torch.utils.data.Dataset):
    """
    Bench2Drive数据集适配器

    输入格式 (来自 Bench2Drive pkl):
    - 历史10帧前视图像作为longcat_video条件输入
    - 相机内参/外参、自车轨迹等可选

    输出格式 (用于Wan训练):
    - video: 历史10帧 + 未来40帧 = 50帧总视频序列 (训练目标)
    - longcat_video: 历史10帧图像 (条件输入)
    - prompt: 场景描述文本

    轨迹信息 (用于规划评估):
    - gt_positions: 全局位置
    - gt_quaternions: 全局四元数
    - gt_history_positions/quaternions: 历史轨迹
    - gt_future_positions/quaternions: 未来轨迹 (相对于当前帧)
    """

    CAMERA_NAMES = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        num_history_frames: int = 2,      # 历史帧数 (longcat_video输入)
        num_future_frames: int = 8,       # 未来预测帧数
        dataset_stride: int = 2,           # 数据集采样的滑动窗口步长 (例如每隔10帧取一个样本)
        target_fps: int = 2,               # 目标帧率 (用于降采样)
        original_fps: int = 10,            # 原始数据帧率
        height: int = 480,
        width: int = 832,
        max_pixels: int = 1280 * 720,
        height_division_factor: int = 16,
        width_division_factor: int = 16,
        time_division_factor: int = 4,
        time_division_remainder: int = 1,
        stitch_history_views: bool = False, # 是否拼接历史多视角图像
        repeat: int = 1,
        prompt_template: str = "Driving scene video from front camera view.",
        use_focal_unification: bool = False,
        focal_target_px: Optional[str] = None,
        return_trajectory: bool = False,    # 是否返回轨迹信息
        use_structured_prompt: bool = True # 新增参数：是否使用结构化Prompt
    ):
        self.data_root = data_root
        self.num_history_frames = num_history_frames
        self.num_future_frames = num_future_frames
        self.num_total_frames = num_history_frames + num_future_frames

        # 计算采样间隔 (降采样)
        self.original_fps = original_fps
        self.target_fps = target_fps
        self.sample_interval = max(1, int(original_fps / target_fps))

        self.dataset_stride = dataset_stride
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.stitch_history_views = stitch_history_views
        self.repeat = repeat
        self.prompt_template = prompt_template
        self.return_trajectory = return_trajectory
        self.fps = target_fps # 使用目标帧率作为prompt生成的参考
        self.use_structured_prompt = use_structured_prompt
        self.load_from_cache = ann_file is None
        self.load_multi_views = True
        self.focal_config = FocalUnificationConfig.build(
            enabled=bool(use_focal_unification),
            target_focal_px=focal_target_px,
            output_width=int(width),
            output_height=int(height),
        )

        # 图像处理器
        self.image_processor = ImageCropAndResize(
            height, width, max_pixels,
            height_division_factor, width_division_factor
        )

        # 加载标注文件
        print(f"Loading annotations: {ann_file}")
        with open(ann_file, "rb") as f:
            pkl_data = _NumpyCompatUnpickler(f).load()
            if isinstance(pkl_data, dict) and 'infos' in pkl_data:
                self.data_infos = pkl_data['infos']
            elif isinstance(pkl_data, list):
                self.data_infos = pkl_data
            else:
                self.data_infos = pkl_data
        print(f"Loaded {len(self.data_infos)} frames")
        print(f"Config: history={num_history_frames}, future={num_future_frames}, total={self.num_total_frames}")
        print(f"Sampling: Original {original_fps}Hz -> Target {target_fps}Hz (Interval={self.sample_interval})")
        print(f"Dataset Stride: {dataset_stride}")

        # 构建时序索引
        self._build_temporal_index()

    def _build_temporal_index(self):
        """构建时序索引，确保采样的帧序列来自同一场景"""
        print("Building temporal index...")
        self.scene_indices = {}
        self.valid_indices = []

        for idx, info in enumerate(self.data_infos):
            scene_name = info['folder']
            if scene_name not in self.scene_indices:
                self.scene_indices[scene_name] = []
            self.scene_indices[scene_name].append(idx)

        # 计算每个场景中可用的采样起点
        # 需要的总帧数对应的原始帧跨度 = (总帧数 - 1) * 采样间隔 + 1
        total_span_needed = (self.num_total_frames - 1) * self.sample_interval + 1

        for scene_name, indices in self.scene_indices.items():
            indices = sorted(indices)
            # 使用 dataset_stride 进行跳跃采样
            for i in range(0, len(indices)):
                if i % self.dataset_stride != 0:
                    continue
                if i + total_span_needed <= len(indices):
                    # 验证帧索引连续性 (检查首尾帧号差值是否符合预期)
                    # 我们需要确保中间没有丢帧（虽然indices是连续的，但frame_idx必须也是连续的）
                    start_idx_in_list = i
                    end_idx_in_list = i + total_span_needed - 1

                    frame_idx_start = self.data_infos[indices[start_idx_in_list]]['frame_idx']
                    frame_idx_end = self.data_infos[indices[end_idx_in_list]]['frame_idx']

                    expected_diff = total_span_needed - 1

                    if (frame_idx_end - frame_idx_start) == expected_diff:
                        self.valid_indices.append(indices[i])

        print(f"Found {len(self.scene_indices)} scenes")
        print(f"Dataset initialized: {len(self.valid_indices)} valid sequences")

    def _load_image(
        self,
        image_path: str,
        intrinsic: Optional[np.ndarray] = None,
    ) -> Tuple[Image.Image, Tuple[int, int], Optional[np.ndarray]]:
        """
        加载并resize图像

        Args:
            image_path: 图像路径

        Returns:
            image: resize后的PIL.Image
            original_size: (width, height) 原始图像尺寸
        """
        full_path = os.path.join(self.data_root, image_path)

        # 优化：使用 OpenCV 加载并 resize，速度比 PIL 快
        image_cv = cv2.imread(full_path)
        if image_cv is None:
            # Fallback 或报错
            raise ValueError(f"Failed to load image: {full_path}")

        h, w, _ = image_cv.shape
        original_size = (w, h)

        # resize到目标尺寸 (cv2.resize 接受 (width, height))
        # BGR -> RGB
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image_cv)
        image, processed_intrinsic, _ = preprocess_camera_image(
            image,
            width=int(self.width),
            height=int(self.height),
            intrinsics=intrinsic,
            focal_config=self.focal_config,
        )

        return image, original_size, processed_intrinsic

    def _get_frame_info(self, global_idx: int) -> dict:
        """获取指定全局索引的帧信息"""
        return self.data_infos[global_idx]

    def _get_camera_path(self, info: dict, camera_name: str = 'CAM_FRONT') -> str:
        """获取相机图像路径"""
        return info['sensors'][camera_name]['data_path']

    def _get_camera_intrinsic(self, info: dict, camera_name: str = 'CAM_FRONT') -> np.ndarray:
        """获取相机内参 (3x3)"""
        return np.array(info['sensors'][camera_name]['intrinsic'])

    def _get_camera_intrinsic_vec(self, info: dict, camera_name: str = 'CAM_FRONT') -> np.ndarray:
        """获取相机内参向量 [fx, fy, cx, cy]。"""
        intrinsic = self._get_camera_intrinsic(info, camera_name)
        return self._intrinsic_vec_from_matrix(intrinsic)

    def _intrinsic_vec_from_matrix(self, intrinsic: np.ndarray) -> np.ndarray:
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        return np.array([fx, fy, cx, cy], dtype=np.float32)

    def _get_camera_extrinsic(self, info: dict, camera_name: str = 'CAM_FRONT') -> Tuple[np.ndarray, np.ndarray]:
        """获取相机外参"""
        sensor_info = info['sensors'][camera_name]
        cam2ego = np.array(sensor_info['cam2ego'])
        world2cam = np.array(sensor_info['world2cam'])
        return cam2ego, world2cam

    def _get_ego_pose(self, info: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取自车位姿 (ego-to-world / camera-to-world)

        Returns:
            position: (3,) 全局位置
            quaternion: (4,) 四元数 [x, y, z, w]
        """
        # 从world2ego获取ego2world
        world2ego = np.array(info['sensors']['LIDAR_TOP']['world2lidar'])
        ego2world = np.linalg.inv(world2ego)

        position = ego2world[:3, 3]
        rotation_matrix = ego2world[:3, :3]
        quaternion = _mat2quat(rotation_matrix)  # [x, y, z, w]

        return position.astype(np.float32), quaternion.astype(np.float32)

    def _compute_trajectory(self, frame_infos: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算完整轨迹 (全局坐标系)

        Returns:
            positions: (N, 3) 所有帧的全局位置
            quaternions: (N, 4) 所有帧的全局四元数 [x, y, z, w]
        """
        positions = []
        quaternions = []

        for info in frame_infos:
            pos, quat = self._get_ego_pose(info)
            positions.append(pos)
            quaternions.append(quat)

        return np.array(positions, dtype=np.float32), np.array(quaternions, dtype=np.float32)

    def _compute_relative_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        reference_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算相对于参考帧的轨迹 (用于评估未来轨迹预测)

        Args:
            positions: (N, 3) 全局位置
            quaternions: (N, 4) 全局四元数 [x, y, z, w]
            reference_idx: 参考帧索引 (通常是当前帧，即历史最后一帧)

        Returns:
            relative_positions: (N, 3) 相对位置
            relative_quaternions: (N, 4) 相对四元数
        """
        ref_pos = positions[reference_idx]
        ref_rot = _quat2mat(quaternions[reference_idx])

        relative_positions = []
        relative_quaternions = []

        for i in range(len(positions)):
            # 相对位置: 转换到参考帧坐标系
            rel_pos = ref_rot.T @ (positions[i] - ref_pos)
            relative_positions.append(rel_pos)

            # 相对旋转
            rot = _quat2mat(quaternions[i])
            rel_rot = ref_rot.T @ rot
            rel_quat = _mat2quat(rel_rot)
            relative_quaternions.append(rel_quat)

        return np.array(relative_positions, dtype=np.float32), np.array(relative_quaternions, dtype=np.float32)






    def _load_raw_4_views(self, global_idx: int) -> List[np.ndarray]:
        """读取单帧4视角原图 (BGR): FL, FR, F, B"""
        info = self._get_frame_info(global_idx)
        # 顺序: 左前, 右前, 前, 后
        layout_order = [
            'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_FRONT', 'CAM_BACK'
        ]
        paths = [os.path.join(self.data_root, self._get_camera_path(info, name)) for name in layout_order]

        def load_cv2(p):
            img = cv2.imread(p)
            if img is None: raise ValueError(f"Error loading {p}")
            return img

        with ThreadPoolExecutor(max_workers=4) as executor:
            images = list(executor.map(load_cv2, paths))
        return images

    def _stitch_2x2_and_resize(self, images_cv: List[np.ndarray]) -> Tuple[Image.Image, Tuple[int, int], dict]:
        """
        将4张图拼成自定义布局并resize
        布局:
        [FL, FR, B] (上排3张)
        [   F     ] (下排1张，拉伸至与上排同宽)

        Returns:
            image: 拼接后resize的图像
            original_size: 拼接后未resize前的尺寸
            front_camera_bbox: 前视图在最终图像中的位置 {'x_min', 'y_min', 'x_max', 'y_max', 'w', 'h'}
        """
        # images_cv 原始顺序: [FL, FR, F, B]
        img_fl = images_cv[0]
        img_fr = images_cv[1]
        img_f  = images_cv[2]
        img_b  = images_cv[3]


        row1 = np.hstack([img_fl, img_fr, img_b])
        row1_h, row1_w, _ = row1.shape

        f_h, f_w, _ = img_f.shape
        new_f_h = int(f_h * (row1_w / f_w))
        img_f_resized = cv2.resize(img_f, (row1_w, new_f_h), interpolation=cv2.INTER_LINEAR)

        # 3. 整体纵向拼接
        canvas_cv = np.vstack([row1, img_f_resized])

        canvas_h, canvas_w, _ = canvas_cv.shape

        # 记录前视图在拼接图中的位置 (在resize前)
        front_camera_bbox_before_resize = {
            'x_min': 0,
            'y_min': row1_h,
            'x_max': row1_w,
            'y_max': row1_h + new_f_h,
            'w': row1_w,
            'h': new_f_h
        }

        # 使用 OpenCV Resize 到最终目标尺寸
        resized_cv = cv2.resize(canvas_cv, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        resized_cv = cv2.cvtColor(resized_cv, cv2.COLOR_BGR2RGB)

        # 计算resize后的前视图位置
        scale_x = self.width / canvas_w
        scale_y = self.height / canvas_h

        front_camera_bbox = {
            'x_min': int(front_camera_bbox_before_resize['x_min'] * scale_x),
            'y_min': int(front_camera_bbox_before_resize['y_min'] * scale_y),
            'x_max': int(front_camera_bbox_before_resize['x_max'] * scale_x),
            'y_max': int(front_camera_bbox_before_resize['y_max'] * scale_y),
            'w': int(front_camera_bbox_before_resize['w'] * scale_x),
            'h': int(front_camera_bbox_before_resize['h'] * scale_y)
        }

        return Image.fromarray(resized_cv), (canvas_w, canvas_h), front_camera_bbox

    def _load_stitched_frame(self, global_idx: int) -> Tuple[Image.Image, Tuple[int, int], dict]:
        """
        [优化版] 加载单帧的4个视角并拼接

        Returns:
            image: 拼接后的图像
            original_size: 拼接后未resize前的尺寸
            front_camera_bbox: 前视图在最终图像中的位置信息
        """
        images_cv = self._load_raw_4_views(global_idx)
        return self._stitch_2x2_and_resize(images_cv)


    def _create_stitched_history_from_cache(self, history_raw_images: List[List[np.ndarray]]) -> Image.Image:
        """
        从缓存的原始图像拼接历史帧 (3行4列)
        """
        if not history_raw_images or not history_raw_images[0]:
             return Image.new('RGB', (self.width, self.height))

        # 假设单张图尺寸 (从第一张图获取)
        h, w, _ = history_raw_images[0][0].shape

        # Map layout_order to CAMERA_NAMES
        # layout: 0:FL, 1:F, 2:FR, 3:BL, 4:B, 5:BR
        # target: F(1), FL(0), FR(2), B(4), BL(3), BR(5)
        reorder_idx = [1, 0, 2, 4, 3, 5]

        all_images_ordered = []
        for frame_imgs in history_raw_images:
            # frame_imgs 是按 layout_order 排列的
            for idx in reorder_idx:
                all_images_ordered.append(frame_imgs[idx])

        # 使用 Numpy 拼接
        cols = 4
        rows = 3

        # 创建黑色画布 (避免图片数量不足导致 hstack/vstack 失败)
        canvas_h = rows * h
        canvas_w = cols * w
        canvas_cv = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for idx, img in enumerate(all_images_ordered):
            if idx >= rows * cols: break # 防止溢出

            r = idx // cols
            c = idx % cols

            # 计算位置
            y_start = r * h
            y_end = y_start + h
            x_start = c * w
            x_end = x_start + w

            # 填充
            canvas_cv[y_start:y_end, x_start:x_end] = img

        final_cv = cv2.resize(canvas_cv, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        final_cv = cv2.cvtColor(final_cv, cv2.COLOR_BGR2RGB)

        return Image.fromarray(final_cv)

    def _create_stitched_history(self, history_indices: List[int]) -> Image.Image:
        """保留旧接口以防万一，但该功能在当前版本未实现"""
        # 注意：此方法依赖 _load_raw_6_views，但未在本版本中实现
        # 如果需要使用，请实现 _load_raw_6_views 和 _stitch_2x3_and_resize 方法
        return Image.new('RGB', (self.width, self.height))

    def __getitem__(self, idx: int) -> dict:
        """
        获取训练样本

        Returns:
            dict: {
                'video': List[PIL.Image],       # 历史10帧 + 未来40帧 = 50帧 (训练目标)
                'longcat_video': List[PIL.Image], # 历史10帧 (条件输入)
                'prompt': str,
                'num_frames': int,              # 总帧数 (50)
                'ego_trajectory': np.ndarray,   # 可选: 相对轨迹
                'camera_intrinsic': np.ndarray, # 相机内参矩阵 (3x3)
                'camera_intrinsic_vec': np.ndarray, # 相机内参向量 [fx, fy, cx, cy]
                'camera_extrinsic': Tuple,      # 相机外参
                'stitched_history': PIL.Image,  # (可选) 拼接的历史多视角图像

                # 轨迹信息 (用于规划评估, 当return_trajectory=True时)
                'gt_positions': np.ndarray,         # (N, 3) 全局位置
                'gt_quaternions': np.ndarray,       # (N, 4) 全局四元数
                'gt_history_positions': np.ndarray, # (num_history, 3) 历史位置
                'gt_history_quaternions': np.ndarray, # (num_history, 4) 历史四元数
                'gt_future_positions': np.ndarray,  # (num_future, 3) 未来位置 (相对当前帧)
                'gt_future_quaternions': np.ndarray, # (num_future, 4) 未来四元数 (相对当前帧)
            }
        """
        start_global_idx = self.valid_indices[idx % len(self.valid_indices)]

        scene_name = self.data_infos[start_global_idx]['folder']
        scene_frame_indices = self.scene_indices[scene_name]
        local_start = scene_frame_indices.index(start_global_idx)

        # 采样所有帧索引 (历史 + 未来)
        all_frame_indices = []
        for i in range(self.num_total_frames):
            # 使用 sample_interval 进行降采样
            frame_offset = i * self.sample_interval
            all_frame_indices.append(scene_frame_indices[local_start + frame_offset])

        # 历史帧索引 (前10帧)
        history_indices = all_frame_indices[:self.num_history_frames]

        # 缓存历史帧的原始图像数据 (List[List[numpy array]])
        # 只有当需要 stitch_history_views 时才缓存
        history_raw_images_cache = []

        # 加载所有帧 (50帧完整视频序列)
        all_frames = []
        original_size = None
        processed_front_intrinsics = []
        front_camera_bboxes = []  # 存储每帧的前视图位置信息

        for i, global_idx in enumerate(all_frame_indices):
            is_history = i < self.num_history_frames

            if self.load_multi_views:
                # 拼接模式：加载视图拼接图 (历史和未来都拼接)
                result = self._load_stitched_frame(global_idx)
                if len(result) == 3:
                    image, orig_size, bbox = result
                    front_camera_bboxes.append(bbox)
                else:
                    image, orig_size = result
                    front_camera_bboxes.append(None)
                processed_front_intrinsics.append(None)
            else:
                # 原始模式：只加载前视
                info = self._get_frame_info(global_idx)
                image_path = self._get_camera_path(info, 'CAM_FRONT')
                intrinsic_raw = self._get_camera_intrinsic(info, 'CAM_FRONT')
                image, orig_size, processed_intrinsic = self._load_image(image_path, intrinsic=intrinsic_raw)
                processed_front_intrinsics.append(processed_intrinsic)
                # 单视图模式，前视图占据整个图像
                front_camera_bboxes.append({
                    'x_min': 0,
                    'y_min': 0,
                    'x_max': self.width,
                    'y_max': self.height,
                    'w': self.width,
                    'h': self.height
                })

            all_frames.append(image)
            if original_size is None:
                original_size = orig_size

        # 历史帧 (longcat_video条件输入)
        longcat_video = all_frames[:self.num_history_frames]

        # 获取帧信息
        all_infos = [self._get_frame_info(idx) for idx in all_frame_indices]
        current_info = all_infos[self.num_history_frames - 1]
        current_intrinsic_raw = self._get_camera_intrinsic(current_info, 'CAM_FRONT')
        current_processed_intrinsic = None
        current_intrinsic_idx = max(0, min(len(processed_front_intrinsics) - 1, self.num_history_frames - 1))
        if len(processed_front_intrinsics) > 0:
            current_processed_intrinsic = processed_front_intrinsics[current_intrinsic_idx]
        current_intrinsic = (
            np.asarray(current_processed_intrinsic, dtype=np.float32)
            if current_processed_intrinsic is not None
            else np.asarray(current_intrinsic_raw, dtype=np.float32)
        )
        current_info = all_infos[self.num_history_frames - 1]  # 当前帧 = 历史最后一帧

        data = {
            'video': all_frames,                    # 50帧完整序列 (训练目标)
            'longcat_video': longcat_video,         # 历史10帧 (条件输入)
            'original_image_size': original_size,  #原始图像尺寸 (width, height)
            'target_image_size': (self.width, self.height),  # 目标图像尺寸
            'num_frames': self.num_total_frames,    # 50
            'front_camera_bboxes': front_camera_bboxes,  # 每帧前视图的位置信息
        }






        # 相机参数
        data['raw_original_image_size'] = original_size
        if current_processed_intrinsic is not None:
            data['original_image_size'] = (self.width, self.height)
        data['camera_intrinsic'] = current_intrinsic
        data['camera_intrinsic_vec'] = self._intrinsic_vec_from_matrix(current_intrinsic)
        data['camera_extrinsic'] = self._get_camera_extrinsic(current_info, 'CAM_FRONT')

        # 轨迹信息 (用于规划评估)
        if self.return_trajectory:
            # 计算全局轨迹
            gt_positions, gt_quaternions = self._compute_trajectory(all_infos)
            data['gt_positions'] = gt_positions
            data['gt_quaternions'] = gt_quaternions

            # 历史轨迹 (全局坐标)
            data['gt_history_positions_global'] = gt_positions[:self.num_history_frames]
            data['gt_history_quaternions_global'] = gt_quaternions[:self.num_history_frames]

            # 未来轨迹 (相对于当前帧)
            current_idx = self.num_history_frames - 1
            relative_positions, relative_quaternions = self._compute_relative_trajectory(
                gt_positions, gt_quaternions, current_idx
            )
            # Align ego-frame conventions with the rest of the training codebase:
            # Bench2Drive raw relative poses are in (X right, Y forward, Z up),
            # while our model expects (X forward, Y left, Z up).
            relative_positions, relative_quaternions = _convert_relative_to_vehicle_frame(
                relative_positions, relative_quaternions
            )
            data['gt_future_positions'] = relative_positions[self.num_history_frames:]
            data['gt_future_quaternions'] = relative_quaternions[self.num_history_frames:]

            # 历史轨迹也转换为相对坐标（用于对齐）
            data['gt_history_positions'] = relative_positions[:self.num_history_frames]
            data['gt_history_quaternions'] = relative_quaternions[:self.num_history_frames]

        ego_vel = None
        if ego_vel is None and self.return_trajectory and "gt_history_positions" in data:
            if len(data["gt_history_positions"]) >= 2:
                delta = data["gt_history_positions"][-1] - data["gt_history_positions"][-2]
                ego_vel = delta * self.fps
        # print("ego_vel",ego_vel,"current_info.get('ego_vel')",current_info.get('ego_vel'))
        if ego_vel is not None:
            data['ego_vel'] = np.array(ego_vel, dtype=np.float32)

        # 根据配置选择 Prompt 生成方式
        if self.use_structured_prompt:
            prompt = generate_structured_prompt(current_info)
        else:
            prompt = generate_navigation_prompt(
                    current_info,
                    ego_vel=ego_vel,
                    history_positions=data.get('gt_history_positions'), # 注意：如果return_trajectory=False，这里可能为None
                    fps=self.fps
                )
        data['prompt'] = prompt

        return data

    def __len__(self) -> int:
        return len(self.valid_indices) * self.repeat


class Bench2DriveEvalDataset(Bench2DriveDataset):
    """Bench2Drive评测数据集"""

    def __init__(self, *args, **kwargs):
        kwargs['repeat'] = 1
        kwargs['return_trajectory'] = True  # 评测时总是返回轨迹信息
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        data = super().__getitem__(idx)

        start_global_idx = self.valid_indices[idx % len(self.valid_indices)]
        info = self._get_frame_info(start_global_idx)

        data['scene_name'] = info['folder']
        data['frame_idx'] = info['frame_idx']
        data['sample_idx'] = idx

        return data
