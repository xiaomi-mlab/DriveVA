<div align="center">

<h1>DriveVA</h1>
<h3>[🎉ECCV 26] DriveVA: Video Action Models are Zero-Shot Drivers</h3>

<p>
Mengmeng Liu<sup>1</sup>, Diankun Zhang<sup>2</sup>, Jiuming Liu<sup>3</sup>,<br>
Jianfeng Cui<sup>2</sup>, Hongwei Xie<sup>2</sup>, Guang Chen<sup>2</sup>,
Hangjun Ye<sup>2</sup>,<br>
Michael Ying Yang<sup>4</sup>, Francesco Nex<sup>1</sup>,
Hao Cheng<sup>1</sup>
</p>

<p>
<sup>1</sup>University of Twente&nbsp;&nbsp;
<sup>2</sup>Xiaomi EV&nbsp;&nbsp;
<sup>3</sup>University of Cambridge&nbsp;&nbsp;
<sup>4</sup>University of Bath
</p>

<a href="https://link.springer.com/chapter/10.1007/978-3-032-37718-0_19">
  <img src="https://img.shields.io/badge/Paper-ECCV_2026-b31b1b.svg" alt="DriveVA ECCV 2026 paper">
</a>
<a href="https://huggingface.co/mengmengliu1998/DriveVA">
  <img src="https://img.shields.io/badge/Model-Hugging%20Face-ffcc4d.svg?logo=huggingface" alt="DriveVA model">
</a>

</div>

## News

- **`September 12th, 2026`:** The [ECCV 2026 proceedings version of DriveVA](https://link.springer.com/chapter/10.1007/978-3-032-37718-0_19) is published on Springer Nature Link.
- **`July 20th, 2026`:** DriveVA NavSIM training code released.
- **`July 10th, 2026`:** DriveVA inference code and checkpoint released.
- **`June 20th, 2026`:** DriveVA is accepted by ECCV 2026🎉🎉🎉!
- **`Apr. 5th, 2026`:** DriveVA is released on [arXiv](https://arxiv.org/abs/2604.04198).

## Table of Contents

- [News](#news)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Qualitative Results](#qualitative-results)
- [Installation](#installation)
- [Model Preparation](#model-preparation)
- [NavSIM v1 Training](#navsim-v1-training)
- [NavSIM v1 Evaluation](#navsim-v1-evaluation)
- [nuScenes Evaluation](#nuscenes-evaluation)
- [Bench2Drive Evaluation](#bench2drive-evaluation)
- [All Datasets](#all-datasets)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Introduction

DriveVA investigates whether large video generation models can serve as
generalizable video action models for autonomous driving. Instead of predicting
future visual imagination and ego trajectories with loosely coupled modules,
DriveVA jointly decodes future video latents and action tokens in a shared
generative process. The model inherits motion and physical priors from
pretrained video generation backbones, uses a DiT-based decoder for unified
video-trajectory prediction, and applies video continuation for longer and more
consistent rollouts.

DriveVA reports a 90.9 PDM score on NAVSIM and demonstrates zero-shot transfer
to nuScenes and Bench2Drive, reducing average L2 error and collision rate over
the prior world-model-based planner baseline reported in the paper.

<div align="center">
<b>Unified video-trajectory rollout for planning.</b>
<img src="assets/driveva_teaser.png" />

<b>Overall pipeline of DriveVA.</b>
<img src="assets/driveva_pipeline.png" />
</div>

## Qualitative Results

<div align="center">
<b>Zero-shot video-trajectory consistency comparison on nuScenes.</b>
<img src="assets/driveva_zero_shot_compare.png" />

<b>Predicted video imaginations and corresponding trajectories.</b>
<img src="assets/driveva_qualitative.png" />
</div>

## Installation

Recommended environment:

- Linux
- Python 3.10
- CUDA-capable GPU
- Git

```bash
git clone https://github.com/xiaomi-mlab/DriveVA.git
cd DriveVA

export REPO_ROOT="$(pwd)"

apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
python -m pip install "setuptools==60.2.0"
python -m pip install -e .
```

`python -m pip install -e .` does not install nuPlan, FlashAttention, or fetch
external runtime repositories. Use the official repositories as installation
references when these dependencies are needed:

- FlashAttention: https://github.com/Dao-AILab/flash-attention
- nuPlan devkit: https://github.com/motional/nuplan-devkit

This repository does not vendor the nuScenes devkit. Clone the official
nuScenes devkit repository into `third_party/nuscenes-devkit` before running
nuScenes inference:

```bash
git clone https://github.com/nutonomy/nuscenes-devkit.git third_party/nuscenes-devkit
```

The launchers add `third_party/nuscenes-devkit/python-sdk` to `PYTHONPATH`
automatically after the directory exists.

## Model Preparation

Download the Wan2.2 base model and DriveVA checkpoint to the default paths:

```bash
python -m pip install -U "huggingface_hub[cli]"

mkdir -p models/Wan-AI checkpoints

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir models/Wan-AI/Wan2.2-TI2V-5B

huggingface-cli download mengmengliu1998/DriveVA DriveVA.safetensors \
  --local-dir checkpoints

mv checkpoints/DriveVA.safetensors checkpoints/pdms90_9.safetensors
```

The released DriveVA checkpoint is hosted at
https://huggingface.co/mengmengliu1998/DriveVA and is saved as
`checkpoints/pdms90_9.safetensors` by the commands above.

The Wan2.2 model directory should contain the text encoder, VAE, and diffusion
checkpoint files required by the Wan2.2 TI2V 5B runtime. Override the defaults
with `LOCAL_MODEL_PATH` and `FULL_CKPT` when needed.

## NavSIM v1 Training

Set the NAVSIM v1 trainval paths and Wan2.2 model directory:

```bash
export NAVSIM_LOG_PATH=/path/to/navsim_logs/trainval
export SENSOR_BLOBS_PATH=/path/to/nuplan/sensor_blobs
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NUPLAN_DATA_ROOT=/path/to/navsim/dataset
export LOCAL_MODEL_PATH=$PWD/models
export FULL_CKPT=$PWD/checkpoints/pdms90_9.safetensors

bash examples/wanvideo/driveva_train/scripts/train_dist_multi_node_navsim.sh
```

To evaluate saved checkpoints on NavSIM, nuScenes, and Bench2Drive:

```bash
bash examples/wanvideo/driveva_train/scripts/train_w_infer.sh
```

`train_w_infer.sh` watches `OUTPUT_PATH` and evaluates both raw and EMA
checkpoints by default. Use `EVAL_CKPT_KIND=ema` or `EVAL_CKPT_KIND=raw` to
evaluate only one checkpoint kind.

Common overrides include `OUTPUT_PATH`, `NUM_EPOCHS`, `SAVE_STEPS`, `LR`,
`NUM_HISTORY_FRAMES`, `NUM_FUTURE_FRAMES`, `TRAINABLE_MODELS`, `USE_EMA`,
`SAVE_EMA`, and `SMOKE_TEST=1`.

## NavSIM v1 Evaluation

Set the dataset paths, NuPlan paths, and checkpoint:

```bash
export NAVSIM_LOG_PATH=/path/to/navsim_logs/test
export NAVSIM_SENSOR_BLOBS_PATH=/path/to/sensor_blobs/test
export NAVSIM_METRIC_CACHE_PATH=/path/to/metric_cache
export NUPLAN_MAPS_ROOT=/path/to/nuplan/maps
export NUPLAN_DATA_ROOT=/path/to/navsim/dataset
export FULL_CKPT=$PWD/checkpoints/pdms90_9.safetensors
```

`NAVSIM_METRIC_CACHE_PATH` should point to metric caches generated with the
official NAVSIM v1.1 metric-caching workflow. See the NAVSIM v1.1 branch and
`scripts/evaluation/run_metric_caching.sh` for the reference command:

- https://github.com/autonomousvision/navsim/tree/v1.1
- https://github.com/autonomousvision/navsim/blob/v1.1/scripts/evaluation/run_metric_caching.sh

The official v1.1 script generates the cache with this pattern:

```bash
TRAIN_TEST_SPLIT=navtest
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=$TRAIN_TEST_SPLIT \
  cache.cache_path=$CACHE_PATH
```

Run evaluation:

```bash
bash examples/wanvideo/driveva_infer/scripts/eval_navsim_v1.sh
```

Small check:

```bash
export SMOKE_TEST=1
export MAX_EVAL_TOKENS=64
bash examples/wanvideo/driveva_infer/scripts/eval_navsim_v1.sh
```

Common overrides include `NUM_INFERENCE_STEPS`, `CFG_SCALE`, `SEED`,
`MAX_EVAL_TOKENS`, `SAVE_VIZ`, `HEIGHT`, `WIDTH`,
`TRAJECTORY_CONDITION_MODE`, `TRAFFIC_AGENTS_POLICY`, `SCENE_FILTER_YAML`, and
`NUM_EVAL_SHARDS`.

## nuScenes Evaluation

```bash
export NUSCENES_DATAROOT=/path/to/nuscenes
export FULL_CKPT=$PWD/checkpoints/pdms90_9.safetensors
export LOCAL_MODEL_PATH=$PWD/models

bash examples/wanvideo/driveva_infer/scripts/infer_nuscenes.sh
```

Set `NUSCENES_EVAL_POLICY_ANNO_JSON` when you need the same validation token
filter as a reference run. If the file is not found, the launcher evaluates the
unfiltered nuScenes validation split.

## Bench2Drive Evaluation

```bash
export B2D_DATA_ROOT=/path/to/bench2drive
export B2D_ANN_FILE=/path/to/b2d_infos_val.pkl
export FULL_CKPT=$PWD/checkpoints/pdms90_9.safetensors

bash examples/wanvideo/driveva_infer/scripts/infer_bench2drive.sh
```

For the common repository layout, the defaults are:

```text
B2D_DATA_ROOT=$REPO_ROOT/data/bench2drive
B2D_ANN_FILE=$REPO_ROOT/data/infos/b2d_infos_val.pkl
OUTPUT_DIR=$REPO_ROOT/outputs/bench2drive
```

The `data/infos/*.pkl` annotation files are expected to be processed metadata
from ORION's Bench2Drive preprocessing results. Use the official Bench2Drive
repository and ORION repository as references for the dataset layout and
processed info files:

- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- ORION: https://github.com/xiaomi-mlab/Orion

## All Datasets

Run all three launchers from one entry point:

```bash
export FULL_CKPT=$PWD/checkpoints/pdms90_9.safetensors
export LOCAL_MODEL_PATH=$PWD/models

bash examples/wanvideo/driveva_infer/scripts/infer_all.sh
```

`infer_all.sh` runs NavSIM, nuScenes, and Bench2Drive sequentially by default
and writes to `outputs/all_infer/{navsim_v1,nuscenes,bench2drive}`. Select
datasets with `RUN_NAVSIM=0`, `RUN_NUSCENES=0`, or `RUN_B2D=0`.
Per-dataset overrides include `NAVSIM_OUTPUT_DIR`, `NUSCENES_OUTPUT_DIR`,
`B2D_OUTPUT_DIR`, `NAVSIM_CONFIG`, `NUSCENES_CONFIG`, and `B2D_CONFIG`.

To run them in parallel, set `INFER_ALL_MODE=parallel` and assign GPUs per
dataset, for example:

```bash
export NAVSIM_CUDA_VISIBLE_DEVICES=0
export NUSCENES_CUDA_VISIBLE_DEVICES=1
export B2D_CUDA_VISIBLE_DEVICES=2
INFER_ALL_MODE=parallel bash examples/wanvideo/driveva_infer/scripts/infer_all.sh
```

## Configuration

Launchers read defaults from `examples/wanvideo/driveva_infer/configs/`. Use
`CONFIG=/path/to/config.yaml` for another config file. Distributed execution is
inferred from standard environment variables such as `CUDA_VISIBLE_DEVICES`,
`GPUS`, `GPUS_PER_NODE`, `NNODES`, `RANK`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT`.

## Acknowledgments

DriveVA builds on and is inspired by the following open-source projects and
benchmarks: Wan2.2, NAVSIM, nuScenes, nuPlan, Bench2Drive, FlashAttention,
and Diffusers.

- DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio
- Wan2.2: https://github.com/Wan-Video/Wan2.2
- NAVSIM: https://github.com/autonomousvision/navsim
- nuScenes devkit: https://github.com/nutonomy/nuscenes-devkit
- nuPlan devkit: https://github.com/motional/nuplan-devkit
- Bench2Drive: https://github.com/Thinklab-SJTU/Bench2Drive
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- Diffusers: https://github.com/huggingface/diffusers

## Citation

If DriveVA is useful for your research or applications, please consider citing:

```bibtex
@inproceedings{liu2026driveva,
  title={Driveva: Video action models are zero-shot drivers},
  author={Liu, Mengmeng and Zhang, Diankun and Liu, Jiuming and Cui, Jianfeng and Xie, Hongwei and Chen, Guang and Ye, Hangjun and Yang, Michael Ying and Nex, Francesco and Cheng, Hao},
  booktitle={European Conference on Computer Vision},
  pages={315--335},
  year={2026},
  organization={Springer}
}
```
