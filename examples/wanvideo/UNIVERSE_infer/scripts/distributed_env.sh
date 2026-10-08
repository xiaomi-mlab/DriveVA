#!/usr/bin/env bash
set -euo pipefail

UNIVERSE_setup_distributed_env() {
  if [ -n "${MLP_ROLE_INDEX:-}" ]; then export RANK="${RANK:-$MLP_ROLE_INDEX}"; fi
  if [ -n "${MLP_WORKER_0_HOST:-}" ]; then export MASTER_ADDR="${MASTER_ADDR:-$MLP_WORKER_0_HOST}"; fi
  if [ -n "${MLP_WORKER_0_PORT:-}" ]; then export MASTER_PORT="${MASTER_PORT:-$MLP_WORKER_0_PORT}"; fi
  if [ -n "${MLP_WORKER_NUM:-}" ]; then export NNODES="${NNODES:-$MLP_WORKER_NUM}"; fi
  if [ -n "${MLP_WORKER_GPU:-}" ]; then export GPUS_PER_NODE="${GPUS_PER_NODE:-$MLP_WORKER_GPU}"; fi
  if [ -n "${GPUS:-}" ]; then export GPUS_PER_NODE="${GPUS_PER_NODE:-$GPUS}"; fi
  if [ -n "${NUM_GPUS:-}" ]; then export GPUS_PER_NODE="${GPUS_PER_NODE:-$NUM_GPUS}"; fi

  if [ -z "${GPUS_PER_NODE:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
      GPUS_PER_NODE=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
    elif command -v nvidia-smi >/dev/null 2>&1; then
      GPUS_PER_NODE=$(nvidia-smi -L | wc -l | tr -d ' ')
    else
      GPUS_PER_NODE=1
    fi
  fi

  NUM_NODES="${NUM_NODES:-${NNODES:-1}}"
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29600}"
  NODE_RANK="${NODE_RANK:-${RANK:-0}}"

  for kv in "GPUS_PER_NODE=${GPUS_PER_NODE}" "NUM_NODES=${NUM_NODES}" "MASTER_PORT=${MASTER_PORT}" "NODE_RANK=${NODE_RANK}"; do
    name="${kv%%=*}"
    value="${kv#*=}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
      echo "[UNIVERSE] ${name} must be an integer, got '${value}'" >&2
      exit 1
    fi
  done
  for kv in "GPUS_PER_NODE=${GPUS_PER_NODE}" "NUM_NODES=${NUM_NODES}" "MASTER_PORT=${MASTER_PORT}"; do
    name="${kv%%=*}"
    value="${kv#*=}"
    if [ "${value}" -lt 1 ]; then
      echo "[UNIVERSE] ${name} must be >= 1, got '${value}'" >&2
      exit 1
    fi
  done
  if [ "${NODE_RANK}" -ge "${NUM_NODES}" ]; then
    echo "[UNIVERSE] NODE_RANK must be < NUM_NODES, got ${NODE_RANK} >= ${NUM_NODES}" >&2
    exit 1
  fi

  TOTAL_PROCESSES=$((NUM_NODES * GPUS_PER_NODE))

  export GPUS_PER_NODE NUM_NODES MASTER_ADDR MASTER_PORT NODE_RANK TOTAL_PROCESSES
  export TORCH_NCCL_ENABLE_TIMING="${TORCH_NCCL_ENABLE_TIMING:-1}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
  export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-8}"
  export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
  export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-13}"

  echo "[UNIVERSE] distributed: nodes=${NUM_NODES} node_rank=${NODE_RANK} gpus_per_node=${GPUS_PER_NODE} total_processes=${TOTAL_PROCESSES} master=${MASTER_ADDR}:${MASTER_PORT}"
}

UNIVERSE_launch_torchrun() {
  local entrypoint="$1"
  shift

  if [ -z "${TOTAL_PROCESSES:-}" ]; then
    UNIVERSE_setup_distributed_env
  fi

  if [ "${NUM_NODES}" -gt 1 ]; then
    torchrun \
      --nnodes "${NUM_NODES}" \
      --node_rank "${NODE_RANK}" \
      --nproc_per_node "${GPUS_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${entrypoint}" "$@"
  else
    torchrun \
      --standalone \
      --nproc_per_node "${GPUS_PER_NODE}" \
      "${entrypoint}" "$@"
  fi
}
