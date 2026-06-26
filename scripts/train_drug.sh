#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/InertialAR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

NUM_GPUS="${NUM_GPUS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-128}"
DATA_ROOT="${DATA_ROOT:-./data}"
DATASET="${DATASET:-Drug}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/drug}"

mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  InertialAR/train_ddp.py \
  --dist \
  --model_3d InertialAR \
  --max_len 183 \
  --n_layer 6 \
  --n_layer_diffusion 6 \
  --n_head 24 \
  --n_embd 768 \
  --num_workers "${NUM_WORKERS}" \
  --dataset_type drug \
  --node_class 119 \
  --position_pad_token 0 \
  --position_eos_token 0 \
  --data_root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --cls_token_num 1 \
  --class_dropout_prob 0.0 \
  --use_qk_layernorm \
  --use_layernorm_atom_type \
  --use_layernorm_position \
  --loss_weight_pos 0.7 \
  --max_epochs 250 \
  --learning_rate 2e-4 \
  --weight_decay 1e-3 \
  --dropout 0.0 \
  --batch_size "${BATCH_SIZE_PER_GPU}" \
  --seed 42 \
  --amp_dtype bf16 \
  --recycle 1 \
  --lr_scheduler CosineLRSchedule \
  --grad_norm_clip 1.0 \
  --warmup_ratio 0.06 \
  --min_lr 1e-9 \
  --save_start_epoch 50 \
  --save_interval_epoch 25 \
  --log_interval 20000 \
  --ema_decay 0.999 \
  --apply_selective_rope selective \
  --scale 1 \
  --rope_theta 100.0 \
  --max_distance 10 \
  --RBF_num_sigma 1 \
  --EPS 1e-8 \
  --dim_diffmlp 1536 \
  --layers_diffmlp 10 \
  --num_t_samples 4 \
  --P_mean -1.2 \
  --P_std 1.2 \
  --sigma_data 2.3 \
  --loss_type per_atom \
  --output_model_dir "${OUTPUT_DIR}"
