from __future__ import annotations

import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d, precompute_freqs_cis
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.longcat_video_dit import LongCatVideoTransformer3DModel
try:
    from ..models.trajectory_modules import TrajectoryEncoder, TrajectoryHead
except ImportError:
    from examples.wanvideo.driveva_infer.trajectory_modules import TrajectoryEncoder, TrajectoryHead
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.trajectory_encoder: Optional[TrajectoryEncoder] = None
        self.trajectory_head: Optional[TrajectoryHead] = None
        self.target_fps: Optional[float] = None
        self.num_history_frames: int = 0
        # Relative trajectory normalization (zero-centered). Scales can be tuned per dataset.
        # Trajectory normalization offsets/scales
        self.noise_x_offset = 2
        self.noise_x_scale = 15
        self.noise_y_offset = 5
        self.noise_y_scale = 12
        self.vel_norm_max = 20.0
        # Trajectory normalization/encoding behavior. Dataset entry points set
        # the checkpoint-specific mode explicitly.
        # - trajectory_use_relative: whether to convert to relative deltas before normalization.
        self.trajectory_norm_mode = "driveva_odo"
        self.trajectory_use_relative = True
        # Training/inference behavior switches.
        # - train_future_video_noise_only: exclude history-condition latents from video loss.
        # - infer_replace_history_latents_before_decode: force decoded history frames to come
        #   from clean history latents.
        self.train_future_video_noise_only = False
        self.infer_replace_history_latents_before_decode = False
        self.trajectory_condition_mode = "auto"
        # Mixed video/trajectory token visibility:
        # - history video tokens and trajectory prefix tokens can attend to each other.
        # - future video tokens cannot attend to future trajectory tokens.
        # - future trajectory tokens cannot attend to future video tokens.
        self.use_mixed_latent_attention_mask = False
        # During trajectory-only inference, skip future video latent denoising and only
        # iteratively infer future trajectory latents.
        self.infer_trajectory_only = False
        self.pad_trajectory_only_video = True
        self.infer_output_mode = "both"
        self.in_iteration_models = ("dit", "trajectory_encoder", "trajectory_head")
        self.in_iteration_models_2 = ("dit2", "trajectory_encoder", "trajectory_head")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_Trajectory(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
            WanVideoUnit_LongCatVideo(),
        ]
        self.post_units = []
        self.model_fn = model_fn_wan_video
    
    def _ensure_traj_dim(self, traj: torch.Tensor, dim: int = 3) -> torch.Tensor:
        """Ensure trajectory has the requested last-dimension size by slicing or padding with zeros."""
        if traj.shape[-1] == dim:
            return traj
        if traj.shape[-1] > dim:
            return traj[..., :dim]
        pad = traj.new_zeros(*traj.shape[:-1], dim - traj.shape[-1])
        return torch.cat([traj, pad], dim=-1)

    def _trajectory_prefix_usage(self, has_history: bool, has_velocity: bool) -> tuple[bool, bool]:
        mode = str(getattr(self, "trajectory_condition_mode", "auto")).strip().lower()
        if mode == "velocity":
            return False, bool(has_velocity)
        if mode == "history":
            return bool(has_history), False
        if has_history:
            return True, False
        return False, bool(has_velocity)

    def _build_mixed_sequence_partition(
        self,
        *,
        num_video_tokens: int,
        num_cond_tokens: int,
        traj_len: int,
        traj_prefix_len: int,
    ) -> Optional[Dict[str, int]]:
        if not bool(getattr(self, "use_mixed_latent_attention_mask", False)):
            return None

        num_video_tokens = max(0, int(num_video_tokens))
        num_cond_tokens = max(0, min(int(num_cond_tokens), num_video_tokens))
        traj_len = max(0, int(traj_len))
        traj_prefix_len = max(0, min(int(traj_prefix_len), traj_len))
        future_video_tokens = max(0, num_video_tokens - num_cond_tokens)
        future_traj_tokens = max(0, traj_len - traj_prefix_len)

        if num_cond_tokens <= 0 and traj_prefix_len <= 0 and future_traj_tokens <= 0:
            return None

        return {
            "past_video_tokens": num_cond_tokens,
            "future_video_tokens": future_video_tokens,
            "traj_prefix_tokens": traj_prefix_len,
            "future_traj_tokens": future_traj_tokens,
        }

    @staticmethod
    def _pad_video_to_length(video: Any, target_length: int) -> Any:
        if not isinstance(video, list):
            return video
        target_length = max(0, int(target_length))
        if len(video) >= target_length or target_length <= 0:
            return video
        if len(video) == 0:
            return video
        last_frame = video[-1]
        while len(video) < target_length:
            video.append(last_frame.copy() if hasattr(last_frame, "copy") else last_frame)
        return video

    @staticmethod
    def _normalize_infer_output_mode(output_mode: Optional[str], infer_trajectory_only: bool) -> str:
        if output_mode is None:
            return "both"
        mode = str(output_mode).strip().lower().replace("-", "_")
        aliases = {
            "traj": "trajectory",
            "trajectory_only": "trajectory",
            "video_only": "video",
            "video_and_trajectory": "both",
            "all": "both",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"trajectory", "video", "both"}:
            raise ValueError(f"infer output mode must be trajectory, video, or both; got {output_mode!r}")
        return mode

    def _to_relative_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Convert absolute positions to relative deltas, keep the first point absolute."""
        if traj.ndim == 2:
            traj = traj.unsqueeze(0)
        traj_xy = traj[..., :2]
        delta_xy = traj_xy[:, 1:] - traj_xy[:, :-1]
        rel_xy = torch.cat([traj_xy[:, 0:1], delta_xy], dim=1)
        if traj.shape[-1] > 2:
            heading = traj[..., 2:3]
            delta_h = heading[:, 1:] - heading[:, :-1]
            rel_h = torch.cat([heading[:, 0:1], delta_h], dim=1)
            return torch.cat([rel_xy, rel_h], dim=-1)
        return rel_xy

    def _from_relative_trajectory(self, traj_rel: torch.Tensor) -> torch.Tensor:
        """Inverse of _to_relative_trajectory: accumulate deltas back to absolute positions."""
        if traj_rel.ndim == 2:
            traj_rel = traj_rel.unsqueeze(0)
        return torch.cumsum(traj_rel, dim=1)

    def norm_trajectory(self, traj: torch.Tensor, is_relative: bool = False, target_fps: Optional[float] = None) -> torch.Tensor:
        """Normalize x/y/heading to the range used by the released checkpoints."""
        traj_xy = traj[..., :2]
        x = traj_xy[..., 0:1]
        y = traj_xy[..., 1:2]
        # Match the trajectory normalization used by the released checkpoints.
        x = 2 * (x + 1.57) / 66.74 - 1
        y = 2 * (y + 19.68) / 42.0 - 1
        if traj.shape[-1] > 2:
            heading = traj[..., 2:3]
            heading = 2 * (heading + 1.67) / 3.53 - 1
            return torch.cat([x, y, heading], dim=-1)
        return torch.cat([x, y], dim=-1)


    def denorm_trajectory(self, traj: torch.Tensor, is_relative: bool = False, target_fps: Optional[float] = None) -> torch.Tensor:
        """Inverse of the released-checkpoint trajectory normalization."""
        traj_xy = traj[..., :2]
        x = traj_xy[..., 0:1]
        y = traj_xy[..., 1:2]
        x = (x + 1) / 2 * 66.74 - 1.57
        y = (y + 1) / 2 * 42.0 - 19.68
        if traj.shape[-1] > 2:
            heading = traj[..., 2:3]
            heading = (heading + 1) / 2 * 3.53 - 1.67
            return torch.cat([x, y, heading], dim=-1)
        return torch.cat([x, y], dim=-1)


    def norm_velocity(self, vel: torch.Tensor, target_fps: Optional[float] = None) -> torch.Tensor:
        # Velocity is encoded directly by a dedicated MLP (TrajectoryEncoder.vel_proj).
        # Keep this helper as identity for backward-compatible call sites.
        return vel[..., :2]


    def denorm_velocity(self, vel: torch.Tensor, target_fps: Optional[float] = None) -> torch.Tensor:
        # Inverse of identity norm_velocity.
        return vel[..., :2]



    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            from ..models.longcat_video_dit import LayerNorm_FP32, RMSNorm_FP32
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                    LayerNorm_FP32: AutoWrappedModule,
                    RMSNorm_FP32: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: List[ModelConfig] = [],
        tokenizer_config: Optional[ModelConfig] = None,
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = False,
        use_usp=False,
        use_trajectory: bool = True,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()

        if tokenizer_config is None:
            local_model_path = None
            model_id = "Wan-AI/Wan2.2-TI2V-5B"
            for model_config in model_configs:
                if getattr(model_config, "local_model_path", None):
                    local_model_path = model_config.local_model_path
                    break
                if getattr(model_config, "model_id", None):
                    model_id = model_config.model_id
            tokenizer_config = ModelConfig(
                model_id=model_id,
                origin_file_pattern="google/*",
                local_model_path=local_model_path,
                skip_download=True,
            )
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        if use_trajectory and pipe.dit is not None:
            # DriveVA and UNIVERSE add lightweight trajectory tokens beside Wan's video
            # patch tokens; the encoder/head are loaded from the full checkpoint.
            pipe.trajectory_encoder = TrajectoryEncoder(point_dim=3, output_dim=pipe.dit.dim).to(
                device=device, dtype=torch_dtype
            )
            pipe.trajectory_head = TrajectoryHead(pipe.dit.dim, out_dim=3).to(
                device=device, dtype=torch_dtype
            )

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    def _ensure_scheduler_training_state(self) -> None:
        timesteps = getattr(self.scheduler, "timesteps", None)
        need_reset = (
            not bool(getattr(self.scheduler, "training", False))
            or timesteps is None
            or len(timesteps) != int(self.scheduler.num_train_timesteps)
            or not hasattr(self.scheduler, "linear_timesteps_weights")
        )
        if need_reset:
            self.scheduler.set_timesteps(int(self.scheduler.num_train_timesteps), training=True)


    def training_loss(self, **inputs):
        self._ensure_scheduler_training_state()

        return_loss_breakdown = bool(inputs.pop("return_loss_breakdown", False))
        num_train_steps = int(self.scheduler.num_train_timesteps)
        max_timestep_boundary = int(float(inputs.get("max_timestep_boundary", 1.0)) * num_train_steps)
        min_timestep_boundary = int(float(inputs.get("min_timestep_boundary", 0.0)) * num_train_steps)
        max_timestep_boundary = max(1, min(max_timestep_boundary, num_train_steps))
        min_timestep_boundary = max(0, min(min_timestep_boundary, max_timestep_boundary - 1))
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        traj_target = None
        has_vel_token = False
        if "traj_tokens" in inputs and inputs["traj_tokens"] is not None:
            traj = inputs.get("trajectory")
            target_fps = inputs.get("target_fps", getattr(self, "target_fps", None))
            history_positions = inputs.get("history_positions", inputs.get("history_trajectory"))
            vel = inputs.get("ego_vel")
            use_history_prefix, use_velocity_prefix = self._trajectory_prefix_usage(
                has_history=history_positions is not None,
                has_velocity=vel is not None,
            )

            hist_norm = None
            hist_len = 0
            if use_history_prefix and history_positions is not None:
                if not torch.is_tensor(history_positions):
                    history_positions = torch.from_numpy(np.asarray(history_positions))
                if history_positions.ndim == 2:
                    history_positions = history_positions.unsqueeze(0)
                history_positions = history_positions.to(device=self.device, dtype=self.torch_dtype)
                history_positions = self._ensure_traj_dim(history_positions, dim=3)
                if self.trajectory_use_relative:
                    history_positions = self._to_relative_trajectory(history_positions)
                hist_norm = self.norm_trajectory(
                    history_positions,
                    is_relative=self.trajectory_use_relative,
                    target_fps=target_fps,
                )
                hist_len = hist_norm.shape[1]

            if not torch.is_tensor(traj):
                traj = torch.from_numpy(np.asarray(traj))
            if traj.ndim == 2:
                traj = traj.unsqueeze(0)
            traj = traj.to(device=self.device, dtype=self.torch_dtype)
            traj = self._ensure_traj_dim(traj, dim=3)
            if self.trajectory_use_relative:
                traj = self._to_relative_trajectory(traj)
            traj_norm = self.norm_trajectory(
                traj,
                is_relative=self.trajectory_use_relative,
                target_fps=target_fps,
            )

            if use_velocity_prefix and vel is not None:
                if not torch.is_tensor(vel):
                    vel = torch.from_numpy(np.asarray(vel))
                if vel.ndim == 1:
                    vel = vel.unsqueeze(0)
                vel = vel.to(device=self.device, dtype=self.torch_dtype)[..., :2]
                vel_norm = self.norm_velocity(vel, target_fps=target_fps)
            else:
                vel_norm = None

            traj_noise = torch.randn_like(traj_norm)
            traj_noisy = self.scheduler.add_noise(traj_norm, traj_noise, timestep)
            traj_noisy = torch.clamp(traj_noisy, min=-1, max=1)

            vel_cond = None
            if vel_norm is not None:
                vel_cond = vel_norm
                has_vel_token = True
            prefix_len = hist_len + (1 if vel_cond is not None else 0)

            traj_proj = self.trajectory_encoder.traj_proj
            if hasattr(traj_proj, "weight"):
                enc_dtype = traj_proj.weight.dtype
            else:
                enc_dtype = next(traj_proj.parameters()).dtype
            if traj_noisy.dtype != enc_dtype:
                traj_noisy = traj_noisy.to(enc_dtype)
            if hist_norm is not None and hist_norm.dtype != enc_dtype:
                hist_norm = hist_norm.to(enc_dtype)
            if vel_cond is not None and vel_cond.dtype != enc_dtype:
                vel_cond = vel_cond.to(enc_dtype)

            inputs["traj_tokens"] = self.trajectory_encoder(
                traj_noisy,
                history_positions=hist_norm,
                velocity=vel_cond,
            )
            inputs["traj_has_vel"] = has_vel_token
            inputs["traj_prefix_len"] = prefix_len
            if hist_norm is not None:
                inputs["traj_prefix_mode"] = "history"
            elif has_vel_token:
                inputs["traj_prefix_mode"] = "velocity"

            traj_target = self.scheduler.training_target(traj_norm, traj_noise, timestep)
            inputs["return_traj_pred"] = True

        inputs.setdefault("traj_postprocess", False)
        inputs.setdefault("pipe", self)
        noise_pred = self.model_fn(**inputs, timestep=timestep)

        if isinstance(noise_pred, dict):
            video_pred = noise_pred.get("video")
            traj_pred = noise_pred.get("traj")
        else:
            video_pred = noise_pred
            traj_pred = None

        video_pred_for_loss = video_pred
        training_target_for_loss = training_target
        if bool(getattr(self, "train_future_video_noise_only", False)):
            longcat_latents = inputs.get("longcat_latents")
            if torch.is_tensor(longcat_latents) and longcat_latents.ndim >= 3:
                cond_t = int(longcat_latents.shape[2])
                if 0 < cond_t < video_pred.shape[2]:
                    video_pred_for_loss = video_pred[:, :, cond_t:]
                    training_target_for_loss = training_target[:, :, cond_t:]

        video_loss = torch.nn.functional.mse_loss(
            video_pred_for_loss.float(),
            training_target_for_loss.float(),
        )
        traj_loss_weight = 0.0
        if traj_pred is not None and traj_target is not None:
            traj_pred_points = traj_pred
            prefix_len = int(inputs.get("traj_prefix_len", 1 if has_vel_token else 0))
            if prefix_len > 0 and traj_pred.shape[1] > prefix_len:
                traj_pred_points = traj_pred[:, prefix_len:]
            traj_loss = torch.nn.functional.mse_loss(traj_pred_points.float(), traj_target.float())
            traj_loss_weight = 1.0
        else:
            traj_loss = torch.zeros_like(video_loss)

        video_loss_scale = max(0.0, float(inputs.get("video_loss_scale", 1.0)))
        trajectory_loss_scale = max(0.0, float(inputs.get("trajectory_loss_scale", 1.0)))
        loss_unweighted = video_loss_scale * video_loss + trajectory_loss_scale * traj_loss_weight * traj_loss
        loss_weight = self.scheduler.training_weight(timestep).to(device=loss_unweighted.device)
        loss = loss_unweighted * loss_weight

        if return_loss_breakdown:
            return {
                "loss": loss,
                "loss_unweighted": loss_unweighted.detach(),
                "video_loss": (video_loss * video_loss_scale * loss_weight).detach(),
                "trajectory_loss": (
                    traj_loss * traj_loss_weight * trajectory_loss_scale * loss_weight
                ).detach(),
                "lr_weight": loss_weight.detach(),
                "trajectory_loss_weight": float(traj_loss_weight),
                "video_loss_scale": float(video_loss_scale),
                "trajectory_loss_scale": float(trajectory_loss_scale),
            }
        return loss


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[List[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # VACE
        vace_video: Optional[List[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[List[Image.Image]] = None,
        animate_face_video: Optional[List[Image.Image]] = None,
        animate_inpaint_video: Optional[List[Image.Image]] = None,
        animate_mask_video: Optional[List[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[List[Image.Image]] = None,
        # Trajectory conditioning
        ego_vel: Optional[torch.Tensor] = None,
        history_positions: Optional[torch.Tensor] = None,
        trajectory_len: Optional[int] = None,
        output_mode: Optional[str] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[Tuple[int, int]] = (30, 52),
        tile_stride: Optional[Tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        # return intermediates
        return_first_step_latents: Optional[bool] = False,
        return_inputs_shared: Optional[bool] = False,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "ego_vel": ego_vel,
            "history_positions": history_positions,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        first_step_latents = None
        traj_noisy = None
        vel_noisy = None
        hist_noisy = None
        legacy_trajectory_only = output_mode is None and bool(getattr(self, "infer_trajectory_only", False))
        infer_mode = self._normalize_infer_output_mode(
            output_mode,
            bool(getattr(self, "infer_trajectory_only", False)),
        )
        trajectory_only_infer = False
        requested_num_frames = int(num_frames)
        if (
            (infer_mode == "trajectory" or legacy_trajectory_only)
            and trajectory_len is not None
        ):
            latents_value = inputs_shared.get("latents")
            longcat_latents_value = inputs_shared.get("longcat_latents")
            if (
                torch.is_tensor(latents_value)
                and latents_value.ndim == 5
                and torch.is_tensor(longcat_latents_value)
                and longcat_latents_value.ndim == 5
            ):
                cond_t = min(int(latents_value.shape[2]), int(longcat_latents_value.shape[2]))
                if cond_t > 0:
                    trajectory_only_infer = True
                    inputs_shared["traj_rope_total_latent_frames"] = int(latents_value.shape[2])
                    inputs_shared["latents"] = latents_value[:, :, :cond_t].clone()
                    if torch.is_tensor(inputs_shared.get("latents_for_resad")):
                        inputs_shared["latents_for_resad"] = inputs_shared["latents"].clone()
                    if torch.is_tensor(inputs_shared.get("noise")):
                        inputs_shared["noise"] = inputs_shared["noise"][:, :, :cond_t].clone()
                    if torch.is_tensor(inputs_shared.get("input_latents")):
                        inputs_shared["input_latents"] = inputs_shared["input_latents"][:, :, :cond_t].clone()
                    inputs_shared["trajectory_only_infer"] = True
        if self.trajectory_encoder is not None and infer_mode != "video":
            # At inference we start from noisy future trajectory points and run
            # the same scheduler updates used for video latents.
            traj_len = None
            if trajectory_len is not None:
                traj_len = int(trajectory_len)
                batch_size = inputs_shared["latents"].shape[0]
                traj_noise = torch.randn((batch_size, traj_len, 3), device=self.device, dtype=self.torch_dtype)
                traj_noisy = torch.clamp(traj_noise, min=-1, max=1)

            if traj_len is not None:
                target_fps = inputs_shared.get("target_fps", getattr(self, "target_fps", None))
                hist = inputs_shared.get("history_positions", inputs_shared.get("history_trajectory"))
                use_history_prefix, use_velocity_prefix = self._trajectory_prefix_usage(
                    has_history=(hist is not None),
                    has_velocity=(ego_vel is not None),
                )
                hist_len = 0
                if use_history_prefix and hist is not None:
                    if not torch.is_tensor(hist):
                        hist = torch.from_numpy(np.asarray(hist))
                    if hist.ndim == 2:
                        hist = hist.unsqueeze(0)
                    hist = hist.to(device=self.device, dtype=self.torch_dtype)
                    hist = self._ensure_traj_dim(hist, dim=3)
                    if self.trajectory_use_relative:
                        hist = self._to_relative_trajectory(hist)
                    hist_noisy = self.norm_trajectory(hist, is_relative=self.trajectory_use_relative, target_fps=target_fps)
                    hist_len = hist_noisy.shape[1]

                if use_velocity_prefix and ego_vel is not None:
                    vel_tensor = ego_vel
                    if not torch.is_tensor(vel_tensor):
                        vel_tensor = torch.from_numpy(np.asarray(vel_tensor))
                    if vel_tensor.ndim == 1:
                        vel_tensor = vel_tensor.unsqueeze(0)
                    vel_tensor = vel_tensor.to(device=self.device, dtype=self.torch_dtype)[..., :2]
                    vel_norm = self.norm_velocity(vel_tensor, target_fps=target_fps)
                    vel_noisy = vel_norm

                traj_proj = self.trajectory_encoder.traj_proj
                if hasattr(traj_proj, "weight"):
                    enc_dtype = traj_proj.weight.dtype
                else:
                    enc_dtype = next(traj_proj.parameters()).dtype
                if hist_noisy is not None and hist_noisy.dtype != enc_dtype:
                    hist_noisy = hist_noisy.to(enc_dtype)
                if vel_noisy is not None and vel_noisy.dtype != enc_dtype:
                    vel_noisy = vel_noisy.to(enc_dtype)

                inputs_shared["return_traj_pred"] = True
                inputs_shared["traj_has_vel"] = vel_noisy is not None
                inputs_shared["traj_prefix_len"] = hist_len + (1 if vel_noisy is not None else 0)
                if hist_noisy is not None:
                    inputs_shared["traj_prefix_mode"] = "history"
                elif vel_noisy is not None:
                    inputs_shared["traj_prefix_mode"] = "velocity"

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            if traj_noisy is not None:
                # Rebuild trajectory tokens from the latest noisy points at each
                # denoise step so DiT can jointly refine video and planning.
                traj_proj = self.trajectory_encoder.traj_proj
                if hasattr(traj_proj, "weight"):
                    enc_dtype = traj_proj.weight.dtype
                else:
                    enc_dtype = next(traj_proj.parameters()).dtype
                traj_cond = traj_noisy if traj_noisy.dtype == enc_dtype else traj_noisy.to(enc_dtype)
                vel_cond = vel_noisy
                if vel_cond is not None and vel_cond.dtype != enc_dtype:
                    vel_cond = vel_cond.to(enc_dtype)
                hist_cond = hist_noisy
                if hist_cond is not None and hist_cond.dtype != enc_dtype:
                    hist_cond = hist_cond.to(enc_dtype)
                inputs_shared["traj_tokens"] = self.trajectory_encoder(
                    traj_cond, history_positions=hist_cond, velocity=vel_cond
                )
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            traj_pred_posi = None
            if isinstance(noise_pred_posi, dict):
                traj_pred_posi = noise_pred_posi.get("traj")
                noise_pred_posi = noise_pred_posi.get("video")
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                    if isinstance(noise_pred_nega, dict):
                        noise_pred_nega = noise_pred_nega.get("video")
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            if trajectory_only_infer:
                if first_step_latents is None and return_first_step_latents:
                    first_step_latents = inputs_shared["latents"].clone()
                inputs_shared["latents_for_resad"] = inputs_shared["latents"].clone()
            else:
                step_latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
                if first_step_latents is None and return_first_step_latents:
                    first_step_latents = step_latents.clone()
                inputs_shared["latents"] = step_latents
                inputs_shared["latents_for_resad"] = step_latents.clone()
                if "first_frame_latents" in inputs_shared:
                    inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
            if traj_noisy is not None and traj_pred_posi is not None:
                traj_pred_points = traj_pred_posi
                prefix_len = int(inputs_shared.get("traj_prefix_len", 1 if vel_noisy is not None else 0))
                if prefix_len > 0 and traj_pred_posi.shape[1] > prefix_len:
                    # Remove history/velocity prefix tokens: only future points are denoised.
                    traj_pred_points = traj_pred_posi[:, prefix_len:]
                if traj_pred_points.shape[1] != traj_noisy.shape[1]:
                    # Fallback for unexpected shapes: keep the latest future-length slice.
                    traj_pred_points = traj_pred_points[:, -traj_noisy.shape[1]:]
                traj_noisy = self.scheduler.step(traj_pred_points, self.scheduler.timesteps[progress_id], traj_noisy)

        # VACE (TODO: remove it)
        if vace_reference_image is not None or (animate_pose_video is not None and animate_face_video is not None):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                f = len(vace_reference_image)
            else:
                f = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        if bool(getattr(self, "infer_replace_history_latents_before_decode", False)):
            longcat_latents = inputs_shared.get("longcat_latents")
            latents = inputs_shared.get("latents")
            if torch.is_tensor(longcat_latents) and torch.is_tensor(latents):
                cond_t = min(int(longcat_latents.shape[2]), int(latents.shape[2]))
                if cond_t > 0:
                    clean_hist = longcat_latents.to(device=latents.device, dtype=latents.dtype)
                    latents[:, :, :cond_t] = clean_hist[:, :, :cond_t]
        # Decode
        video = None
        if infer_mode != "trajectory":
            self.load_models_to_device(['vae'])
            video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            video = self.vae_output_to_video(video)
            if trajectory_only_infer and bool(getattr(self, "pad_trajectory_only_video", False)):
                video = self._pad_video_to_length(video, requested_num_frames)
        self.load_models_to_device([])
        if traj_noisy is not None:
            target_fps = inputs_shared.get("target_fps", getattr(self, "target_fps", None))
            traj_abs = self.denorm_trajectory(traj_noisy, is_relative=self.trajectory_use_relative, target_fps=target_fps)
            if self.trajectory_use_relative:
                traj_abs = self._from_relative_trajectory(traj_abs)
            inputs_shared["traj_denoised"] = traj_abs
            if vel_noisy is not None:
                inputs_shared["vel_denoised"] = self.denorm_velocity(vel_noisy, target_fps=target_fps)
        traj_denoised = inputs_shared.get("traj_denoised")
        vel_denoised = inputs_shared.get("vel_denoised")

        if return_first_step_latents or return_inputs_shared:
            inputs_shared["infer_output_mode"] = infer_mode
            if return_inputs_shared and "context" in inputs_posi:
                inputs_shared["context"] = inputs_posi["context"]
            if return_first_step_latents and return_inputs_shared:
                return video, first_step_latents, inputs_shared
            if return_first_step_latents:
                return video, first_step_latents
            return video, inputs_shared
        if infer_mode == "trajectory":
            if traj_denoised is None and vel_denoised is None:
                return None
            if vel_denoised is not None:
                return traj_denoised, vel_denoised
            return traj_denoised
        if infer_mode == "video":
            return video
        if traj_denoised is not None or vel_denoised is not None:
            return video, traj_denoised, vel_denoised
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_Trajectory(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("trajectory", "ego_vel", "history_positions", "history_trajectory"),
            onload_model_names=("trajectory_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, trajectory, ego_vel, history_positions, history_trajectory):
        if trajectory is None or pipe.trajectory_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        # Explicit trajectory conditioning path for callers that already have a
        # target trajectory tensor; infer_all normally uses trajectory_len instead.
        traj = trajectory
        if not isinstance(traj, torch.Tensor):
            traj = torch.from_numpy(np.asarray(traj))
        if traj.ndim == 2:
            traj = traj.unsqueeze(0)
        traj = traj.to(device=pipe.device, dtype=pipe.torch_dtype)
        traj = pipe._ensure_traj_dim(traj, dim=3)
        target_fps = pipe.target_fps
        if pipe.trajectory_use_relative:
            traj = pipe._to_relative_trajectory(traj)
        traj = pipe.norm_trajectory(traj, is_relative=pipe.trajectory_use_relative, target_fps=target_fps)
        traj_proj = pipe.trajectory_encoder.traj_proj
        if hasattr(traj_proj, "weight"):
            enc_dtype = traj_proj.weight.dtype
        else:
            enc_dtype = next(traj_proj.parameters()).dtype
        if traj.dtype != enc_dtype:
            traj = traj.to(enc_dtype)

        hist = history_positions if history_positions is not None else history_trajectory
        use_history_prefix, use_velocity_prefix = pipe._trajectory_prefix_usage(
            has_history=(hist is not None),
            has_velocity=(ego_vel is not None),
        )
        hist_norm = None
        hist_len = 0
        if use_history_prefix and hist is not None:
            if not isinstance(hist, torch.Tensor):
                hist = torch.from_numpy(np.asarray(hist))
            if hist.ndim == 2:
                hist = hist.unsqueeze(0)
            hist = hist.to(device=pipe.device, dtype=pipe.torch_dtype)
            hist = pipe._ensure_traj_dim(hist, dim=3)
            if pipe.trajectory_use_relative:
                hist = pipe._to_relative_trajectory(hist)
            hist_norm = pipe.norm_trajectory(hist, is_relative=pipe.trajectory_use_relative, target_fps=target_fps)
            if hist_norm.dtype != enc_dtype:
                hist_norm = hist_norm.to(enc_dtype)
            hist_len = hist_norm.shape[1]

        vel = None
        if use_velocity_prefix and ego_vel is not None:
            vel = ego_vel
            if not isinstance(vel, torch.Tensor):
                vel = torch.from_numpy(np.asarray(vel))
            if vel.ndim == 1:
                vel = vel.unsqueeze(0)
            vel = vel.to(device=pipe.device, dtype=pipe.torch_dtype)[..., :2]
            vel = pipe.norm_velocity(vel, target_fps=target_fps)
            if vel is not None and vel.dtype != enc_dtype:
                vel = vel.to(enc_dtype)

        prefix_len = hist_len + (1 if vel is not None else 0)
        tokens = pipe.trajectory_encoder(traj, history_positions=hist_norm, velocity=vel)
        out = {"traj_tokens": tokens, "traj_prefix_len": prefix_len, "traj_has_vel": vel is not None}
        if hist_norm is not None:
            out["traj_prefix_mode"] = "history"
        elif vel is not None:
            out["traj_prefix_mode"] = "velocity"
        return out



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("input_video", "animate_pose_video", "animate_face_video", "animate_inpaint_video", "animate_mask_video"))

    def process(self, pipe: WanVideoPipeline, input_video, animate_pose_video, animate_face_video, animate_inpaint_video, animate_mask_video):
        if input_video is None:
            return {}
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[:len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[:len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[:len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[:len(input_video) - 4]
        return {"animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video}


class WanVideoPostUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, animate_pose_video, tiled, tile_size, tile_stride):
        if animate_pose_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(animate_pose_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoPostUnit_AnimateFacePixelValues(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("animate_face_video", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(inputs_shared["animate_face_video"])
        inputs_nega["face_pixel_values"] = torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_inpaint_video", "animate_mask_video", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        
    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def process(self, pipe: WanVideoPipeline, animate_inpaint_video, animate_mask_video, input_image, tiled, tile_size, tile_stride):
        if animate_inpaint_video is None or animate_mask_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(bg_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape
        
        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(ref_pixel_values, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=torch.bfloat16, device=pipe.device)
        
        mask_pixel_values = 1 - pipe.preprocess_video(animate_mask_video, max_value=1, min_value=0)
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(mask_pixel_values, size=(lat_h, lat_w), mode='nearest')
        mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b t c h w", b=1)[:,:,0]
        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, 0, mask_pixel_values=mask_pixel_values, device=pipe.device)
        
        y_reft = torch.concat([msk_reft, y_reft]).to(dtype=torch.bfloat16, device=pipe.device)
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class WanVideoUnit_LongCatVideo(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("longcat_video",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, longcat_video):
        if longcat_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        longcat_video = pipe.preprocess_video(longcat_video)
        longcat_latents = pipe.vae.encode(longcat_video, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"longcat_latents": longcat_latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value

def model_fn_wan2_2_5b_longcat(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    """
    Wan2.2 5B 模型的 LongCat 版本
    集成了 Self-Attention 和 Cross-Attention 的条件帧优化
    """
    B, C, T, H, W = latents.shape
    
    # 1. 注入历史潜在向量
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    
    
    # 3. 调用 DiT 模型
    output = dit(
        latents,
        timestep,
        context,
        num_cond_latents=num_cond_latents,  # 传递条件帧数量
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    
    output = -output
    output = output.to(latents.dtype)
    return output

def model_fn_wan_video(
    dit: WanModel,
    motion_controller = None,
    vace = None,
    animate_adapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    vace_context = None,
    vace_scale = 1.0,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    fuse_vae_embedding_in_latents: bool = False,
    traj_tokens: Optional[torch.Tensor] = None,
    trajectory_head: Optional[TrajectoryHead] = None,
    return_traj_pred: bool = False,
    pipe=None,
    traj_postprocess: bool = True,
    target_fps: Optional[float] = None,
    **kwargs,
):

    # if longcat_latents is not None and isinstance(dit, WanModel):

    #     return model_fn_wan2_2_5b_longcat(
    #         dit=dit,
    #         latents=latents,
    #         timestep=timestep,
    #         context=context,
    #         longcat_latents=longcat_latents,
    #         use_gradient_checkpointing=use_gradient_checkpointing,
    #         use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    #     ) 
    
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # LongCat-Video
    if isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    # if dit.seperated_timestep and fuse_vae_embedding_in_latents:
    #     timestep = torch.concat([
    #         torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
    #         torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
    #     ]).flatten()
    #     t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
    #     if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
    #         t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
    #         t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
    #         t = t_chunks[get_sequence_parallel_rank()]
    #     t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    # else:
    #     t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    #     t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
        B, C, T, H, W = latents.shape
    
    # 1. 注入历史潜在向量
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0

    diffusion_timestep = timestep
    traj_rope_total_latent_frames = max(
        int(kwargs.get("traj_rope_total_latent_frames", latents.shape[2])),
        int(latents.shape[2]),
    )

    if num_cond_latents > 0:
        timestep = torch.concat([
            torch.zeros((num_cond_latents, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - num_cond_latents, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    dit_dtype = next(dit.parameters()).dtype

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    if t_mod.dtype != dit_dtype:
        t_mod = t_mod.to(dit_dtype)
    context = dit.text_embedding(context)
    if context.dtype != dit_dtype:
        context = context.to(dit_dtype)

    x = latents
    if x.dtype != dit_dtype:
        x = x.to(dit_dtype)
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        if y.dtype != dit_dtype:
            y = y.to(dit_dtype)
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        if clip_feature.dtype != dit_dtype:
            clip_feature = clip_feature.to(dit_dtype)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
    
    # Camera control
    x = dit.patchify(x)
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    traj_len = 0
    if traj_tokens is not None:
        traj_len = traj_tokens.shape[1]
        traj_has_vel = bool(kwargs.get("traj_has_vel", False))
        if ("traj_has_vel" not in kwargs) and (not traj_has_vel):
            traj_has_vel = kwargs.get("ego_vel", None) is not None
        traj_prefix_len = int(kwargs.get("traj_prefix_len", 1 if traj_has_vel else 0))
        traj_prefix_mode = kwargs.get("traj_prefix_mode", None)
        if traj_prefix_mode is None and traj_has_vel:
            traj_prefix_mode = "velocity"
        if traj_tokens.dtype != dit_dtype:
            traj_tokens = traj_tokens.to(dit_dtype)
        # Append planning tokens to the DiT sequence so attention can exchange
        # information between video patches and future trajectory points.
        x = torch.concat([x, traj_tokens], dim=1)
        if len(t_mod.shape) == 4:
            use_partitioned_traj_timestep = bool(
                pipe is not None and getattr(pipe, "use_mixed_latent_attention_mask", False)
            )
            if not use_partitioned_traj_timestep:
                # Preserve the DriveVA checkpoint's original time modulation.
                t_traj = t_mod[:, :1].expand(t_mod.shape[0], traj_len, 6, dit.dim)
            else:
                motion_mod = None
                if motion_bucket_id is not None and motion_controller is not None:
                    motion_mod = motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim)).unsqueeze(1)

                def _build_traj_token_t_mod(step_value: torch.Tensor) -> torch.Tensor:
                    step_mod = dit.time_projection(
                        dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, step_value))
                    ).unflatten(1, (6, dit.dim)).unsqueeze(1)
                    if motion_mod is not None:
                        step_mod = step_mod + motion_mod
                    if step_mod.dtype != dit_dtype:
                        step_mod = step_mod.to(dit_dtype)
                    return step_mod

                traj_t_parts = []
                if traj_prefix_len > 0:
                    prefix_step = (
                        torch.zeros_like(diffusion_timestep)
                        if num_cond_latents > 0
                        else diffusion_timestep
                    )
                    prefix_t_mod = _build_traj_token_t_mod(prefix_step)
                    traj_t_parts.append(prefix_t_mod.expand(prefix_t_mod.shape[0], traj_prefix_len, 6, dit.dim))
                future_traj_len = max(traj_len - traj_prefix_len, 0)
                if future_traj_len > 0:
                    future_t_mod = _build_traj_token_t_mod(diffusion_timestep)
                    traj_t_parts.append(future_t_mod.expand(future_t_mod.shape[0], future_traj_len, 6, dit.dim))
                t_traj = (
                    torch.cat(traj_t_parts, dim=1)
                    if len(traj_t_parts) > 0
                    else t_mod[:, :0]
                )
            t_mod = torch.cat([t_mod, t_traj], dim=1)
    else:
        traj_prefix_len = 0

    if x.dtype != dit_dtype:
        x = x.to(dit_dtype)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    if traj_len > 0:
        traj_has_vel = bool(kwargs.get("traj_has_vel", False))
        if ("traj_has_vel" not in kwargs) and (not traj_has_vel):
            traj_has_vel = kwargs.get("ego_vel", None) is not None
        traj_prefix_len = int(kwargs.get("traj_prefix_len", 1 if traj_has_vel else 0))
        traj_prefix_mode = kwargs.get("traj_prefix_mode", None)
        if traj_prefix_mode is None and traj_has_vel:
            traj_prefix_mode = "velocity"

        f_rope = traj_rope_total_latent_frames
        # Align trajectory tokens to future frames in the same f-scale as RoPE.
        start_frame = max(num_cond_latents, 0)
        end_frame = max(f_rope - 1, 0)
        if end_frame < start_frame:
            end_frame = start_frame

        traj_main_len = max(traj_len - traj_prefix_len, 0)
        traj_pos_parts = []
        if traj_prefix_len > 0:
            if traj_prefix_mode == "history":
                hist_end = max(start_frame - 1, 0)
                if traj_prefix_len == 1:
                    prefix_pos = torch.tensor([hist_end], device=freqs.device, dtype=torch.float32)
                else:
                    prefix_pos = torch.linspace(0, hist_end, traj_prefix_len, device=freqs.device, dtype=torch.float32)
            else:
                prefix_pos = torch.full((traj_prefix_len,), float(start_frame), device=freqs.device, dtype=torch.float32)
            traj_pos_parts.append(prefix_pos)
        if traj_main_len > 0:
            traj_pos = torch.linspace(start_frame, end_frame, traj_main_len, device=freqs.device, dtype=torch.float32)
            traj_pos_parts.append(traj_pos)
        if traj_pos_parts:
            traj_pos = torch.cat(traj_pos_parts, dim=0)
        else:
            traj_pos = torch.tensor([], device=freqs.device, dtype=torch.float32)
        dim = freqs.shape[-1] * 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=freqs.device, dtype=torch.float32) / dim))
        traj_freqs = torch.outer(traj_pos, inv_freq)
        traj_freqs = torch.polar(torch.ones_like(traj_freqs), traj_freqs).to(dtype=freqs.dtype)
        traj_freqs = traj_freqs.view(traj_len, 1, -1)
        freqs = torch.cat([freqs, traj_freqs], dim=0)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_x = x[:, :-traj_len] if traj_len > 0 else x
        vace_freqs = freqs[:-traj_len] if traj_len > 0 else freqs
        vace_hints = vace(
            vace_x, vace_context, context, t_mod, vace_freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    sequence_partition = None
    if pipe is not None and hasattr(pipe, "_build_mixed_sequence_partition"):
        sequence_partition = pipe._build_mixed_sequence_partition(
            num_video_tokens=int(f * h * w),
            num_cond_tokens=int(num_cond_latents * h * w),
            traj_len=int(traj_len),
            traj_prefix_len=int(traj_prefix_len),
        )
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            sequence_partition = None
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, sequence_partition=sequence_partition)
            return custom_forward
        # Count tokens that belong to clean conditioned history frames.
        num_cond_tokens = 0
        if num_cond_latents > 0:
            # Each latent frame contributes h * w video patch tokens.
            num_cond_tokens = num_cond_latents * h * w
        for block_id, block in enumerate(dit.blocks):
            # Block
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs, sequence_partition=sequence_partition)
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if traj_len > 0:
                    pad = current_vace_hint.new_zeros(
                        (current_vace_hint.shape[0], traj_len, current_vace_hint.shape[2])
                    )
                    current_vace_hint = torch.cat([current_vace_hint, pad], dim=1)
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)

    gathered = False
    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1 and traj_len > 0:
        x = get_sp_group().all_gather(x, dim=1)
        x = x[:, :-pad_shape] if pad_shape > 0 else x
        gathered = True

    traj_pred = None
    if traj_len > 0:
        traj_out = x[:, -traj_len:]
        x = x[:, :-traj_len]
        if return_traj_pred and trajectory_head is not None:
            # Trajectory head outputs noise prediction (same space as video eps).
            # Postprocess is handled outside via scheduler updates; keep raw prediction here.
            traj_pred = trajectory_head(traj_out)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1 and not gathered:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    x = dit.unpatchify(x, (f, h, w))
    if return_traj_pred:
        return {"video": x, "traj": traj_pred}
    return x


def model_fn_longcat_video(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    longcat_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
):
    if longcat_latents is not None:
        latents[:, :, :longcat_latents.shape[2]] = longcat_latents
        num_cond_latents = longcat_latents.shape[2]
    else:
        num_cond_latents = 0
    context = context.unsqueeze(0)
    encoder_attention_mask = torch.any(context != 0, dim=-1)[:, 0].to(torch.int64)
    output = dit(
        latents,
        timestep,
        context,
        encoder_attention_mask,
        num_cond_latents=num_cond_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    output = -output
    output = output.to(latents.dtype)
    return output


