#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    PYTHON="python3"
  fi
fi

SEED="${SEED:-42}"
SUMMARY_JSON="${SUMMARY_JSON:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: DATASET_NAME=<qm9|drug|b3lyp|geom> $0 <generated.npz|generated.txt> [output.txt]" >&2
  exit 2
fi

if [[ -z "${DATASET_NAME:-}" ]]; then
  echo "Error: set DATASET_NAME to one of: qm9, drug, b3lyp, geom" >&2
  exit 2
fi

case "${DATASET_NAME}" in
  qm9|drug|b3lyp|b3lyp_17m|geom) ;;
  *)
    echo "Error: unsupported DATASET_NAME='${DATASET_NAME}'" >&2
    echo "Supported values: qm9, drug, b3lyp, b3lyp_17m, geom" >&2
    exit 2
    ;;
esac

INPUT_PATH="$1"
OUTPUT_TEXT="${2:-}"

run_eval() {
  if [[ -n "${SUMMARY_JSON}" ]]; then
    "${PYTHON}" -m InertialAR.evaluation.eval "$@" --summary_json "${SUMMARY_JSON}"
  else
    "${PYTHON}" -m InertialAR.evaluation.eval "$@"
  fi
}

if [[ "${INPUT_PATH}" == *.npz ]]; then
  if [[ -n "${OUTPUT_TEXT}" ]]; then
    run_eval \
      --input_npz "${INPUT_PATH}" \
      --output_text "${OUTPUT_TEXT}" \
      --dataset_name "${DATASET_NAME}" \
      --seed "${SEED}"
  else
    run_eval \
      --input_npz "${INPUT_PATH}" \
      --dataset_name "${DATASET_NAME}" \
      --seed "${SEED}"
  fi
else
  run_eval \
    --input_path "${INPUT_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --seed "${SEED}"
fi
