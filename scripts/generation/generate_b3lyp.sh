#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/InertialAR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi

DATA_ROOT="${DATA_ROOT:-./data}"
DATASET="${DATASET:-B3LYP_17M}"
CKPT_PATH="${CKPT_PATH:-./ckpt/B3LYP}"
DEVICE="${DEVICE:-auto}"

NUM_GENERATE="${NUM_GENERATE:-10000}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
EPOCH="${EPOCH:-1000}"
SEED="${SEED:-42}"
AUTO_EVAL="${AUTO_EVAL:-0}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-b3lyp}"

mkdir -p "${CKPT_PATH}"
MARKER_FILE="$(mktemp)"
touch "${MARKER_FILE}"
cleanup() {
  rm -f "${MARKER_FILE}"
}
trap cleanup EXIT

"${PYTHON}" InertialAR/generate_b3lyp.py \
  --cond_id -1 \
  --cond_pos 0 \
  --num_generate "${NUM_GENERATE}" \
  --batch_size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --data_root "${DATA_ROOT}" \
  --dataset "${DATASET}" \
  --ckpt_path "${CKPT_PATH}" \
  --epoch "${EPOCH}" \
  --model_3d InertialAR \
  --node_class 119 \
  --max_len 126 \
  --n_layer 6 \
  --n_layer_diffusion 6 \
  --n_head 24 \
  --n_embd 768 \
  --cls_token_num 1 \
  --class_dropout_prob 0.0 \
  --ema_decay 0.999 \
  --use_qk_layernorm \
  --use_layernorm_atom_type \
  --use_layernorm_position \
  --loss_weight_pos 0.8 \
  --dropout 0.0 \
  --recycle 1 \
  --cfg_scale 1.0 \
  --top_k 0 \
  --top_p 1.0 \
  --temperature 1.0 \
  --num_steps 120 \
  --S_churn 0 \
  --S_min 0.1 \
  --S_max 50 \
  --S_noise 1 \
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
  --sigma_data 1.8 \
  --loss_type per_atom

if [[ "${AUTO_EVAL}" == "1" ]]; then
  GENERATED_NPZ="$(find "${CKPT_PATH}" -maxdepth 1 -type f -name "generated_epoch_${EPOCH}_*.npz" -newer "${MARKER_FILE}" | sort | tail -n 1)"
  if [[ -z "${GENERATED_NPZ}" ]]; then
    echo "Error: no generated NPZ found under ${CKPT_PATH}" >&2
    exit 1
  fi
  DATASET_NAME="${EVAL_DATASET_NAME}" SEED="${SEED}" bash scripts/evaluation/evaluate_generated.sh "${GENERATED_NPZ}"
fi
