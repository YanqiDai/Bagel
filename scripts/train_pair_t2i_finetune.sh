#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAGEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BAGEL_ROOT}"
export PYTHONPATH="${BAGEL_ROOT}"

# ---------- Required paths ----------
model_path="/mnt/yanqi/models/BAGEL-7B-MoT"

# ---------- Training args ----------
num_workers=1
total_steps=500
warmup_steps=50
lr=1e-5
log_every=1
wandb_runid="pair_t2i_$(date +%Y%m%d_%H%M%S)"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr="127.0.0.1" \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/pair_t2i_finetune.yaml \
  --model_path "${model_path}" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume_from "${model_path}" \
  --visual_und False \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema True \
  --log_every "${log_every}" \
  --lr "${lr}" \
  --total_steps "${total_steps}" \
  --warmup_steps "${warmup_steps}" \
  --num_workers "${num_workers}" \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --wandb_runid "${wandb_runid}"
