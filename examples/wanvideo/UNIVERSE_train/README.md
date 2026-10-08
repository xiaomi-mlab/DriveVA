# UNIVERSE Training

This directory contains the NavSIM v1 training entry point for UNIVERSE. For
the model overview, paper figures, checkpoint download, and inference entry
points, see the [UNIVERSE documentation](../../../UNIVERSE.md).

## Data and Model Paths

```bash
export NAVSIM_LOG_PATH=/path/to/navsim_v1.1/navsim_logs/trainval
export SENSOR_BLOBS_PATH=/path/to/nuplan/dataset/nuplan-v1.1/sensor_blobs
export NUPLAN_MAPS_ROOT=/path/to/nuplan/dataset/maps
export NUPLAN_DATA_ROOT=/path/to/navsim_v1.1/dataset
export LOCAL_MODEL_PATH=$PWD/models
```

`FULL_CKPT` is optional. When omitted, training initializes from the configured
Wan2.2 base model. Set it to initialize from a full UNIVERSE-compatible
checkpoint:

```bash
export FULL_CKPT=$PWD/checkpoints/UNIVERSE.safetensors
```

## Launch

Single-node or environment-configured distributed training:

```bash
bash examples/wanvideo/UNIVERSE_train/scripts/train_navsim_v1.sh
```

Multi-node launcher:

```bash
bash examples/wanvideo/UNIVERSE_train/scripts/train_dist_multi_node_navsim.sh
```

Training followed by inference on enabled datasets:

```bash
bash examples/wanvideo/UNIVERSE_train/scripts/train_w_infer.sh
```

The default configuration is
`examples/wanvideo/UNIVERSE_train/configs/navsim_v1.yaml`. Environment
variables override YAML values. Checkpoints and logs are written to
`outputs/universe/train_navsim_v1` by default.

## Distributed Settings

The launchers read standard distributed environment variables:

```bash
export GPUS_PER_NODE=8
export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29600
```

For a short training smoke run, use:

```bash
SMOKE_TEST=1 AUTO_EVAL=0 \
  bash examples/wanvideo/UNIVERSE_train/scripts/train_navsim_v1.sh
```

Automatic evaluation is enabled by default in the released training config.
Control it with `AUTO_EVAL`, `RUN_NAVSIM`, `RUN_NUSCENES`, and `RUN_B2D`.
