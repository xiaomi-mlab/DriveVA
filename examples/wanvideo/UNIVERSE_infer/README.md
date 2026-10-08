# UNIVERSE Inference

This directory contains UNIVERSE inference entry points for NavSIM v1,
nuScenes, and Bench2Drive. For the architecture, paper figures, training guide,
and checkpoint download, see the
[UNIVERSE documentation](../../../UNIVERSE.md).

## Common Paths

The launchers use these defaults:

```bash
export LOCAL_MODEL_PATH=$PWD/models
export FULL_CKPT=$PWD/checkpoints/UNIVERSE.safetensors
```

The released YAML configurations use three flow-matching sampling steps.
Environment variables override all YAML defaults.

## NavSIM v1

```bash
export NAVSIM_LOG_PATH=/path/to/navsim_v1.1/navsim_logs/test
export NAVSIM_SENSOR_BLOBS_PATH=/path/to/navsim_v1.1/sensor_blobs/test
export NAVSIM_METRIC_CACHE_PATH=/path/to/navsim_v1.1/metric_cache
export NUPLAN_MAPS_ROOT=/path/to/nuplan/dataset/maps
export NUPLAN_DATA_ROOT=/path/to/navsim_v1.1/dataset

bash examples/wanvideo/UNIVERSE_infer/scripts/eval_navsim_v1.sh
```

The default output is `outputs/universe/navsim_v1`. Set `SAVE_VIZ=1` to save
visualizations.

## nuScenes

Clone the official nuScenes devkit as described in the main
[installation guide](../../../README.md#installation), then run:

```bash
export NUSCENES_DATAROOT=/path/to/nuscenes
export NUSCENES_VERSION=v1.0-trainval
export NUSCENES_SPLIT=val

bash examples/wanvideo/UNIVERSE_infer/scripts/infer_nuscenes.sh
```

The default output is `outputs/universe/nuscenes`. The optional
`POLICY_ANNO_JSON` controls policy-token filtering.

## Bench2Drive

```bash
export B2D_DATA_ROOT=/path/to/bench2drive
export B2D_ANN_FILE=/path/to/b2d_infos_val.pkl

bash examples/wanvideo/UNIVERSE_infer/scripts/infer_bench2drive.sh
```

The default output is `outputs/universe/bench2drive`.

## All Datasets

Run all enabled datasets sequentially:

```bash
bash examples/wanvideo/UNIVERSE_infer/scripts/infer_all.sh
```

Select datasets with `RUN_NAVSIM`, `RUN_NUSCENES`, and `RUN_B2D`. For parallel
execution, set `INFER_ALL_MODE=parallel` and assign per-dataset visible GPUs
with `NAVSIM_CUDA_VISIBLE_DEVICES`, `NUSCENES_CUDA_VISIBLE_DEVICES`, and
`B2D_CUDA_VISIBLE_DEVICES`.

Configuration files are stored in `configs/`. Generated results from
`infer_all.sh` are written below `outputs/universe/all_infer` unless
`INFER_ALL_OUTPUT_ROOT` is overridden.
