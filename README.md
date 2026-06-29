# InertialAR: Autoregressive 3D Molecule Generation with Inertial Frames

[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-green.svg)](https://openreview.net/pdf?id=GUu3Bi41Fm)

This is the official repository for the paper **[InertialAR: Autoregressive 3D Molecule Generation with Inertial Frames](https://openreview.net/pdf?id=GUu3Bi41Fm)**, accepted at **ICML 2026**.

## Setup

See [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) for the recommended Python 3.11, PyTorch 2.3.0, and FlashAttention 2.6.2 environment. In short:

```bash
conda create -n InertialTransformer python=3.11 -y
conda activate InertialTransformer

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r env.txt
python -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
python -m pip install -e .
```

## Data and Checkpoints

Preprocessed data and released checkpoints are hosted at
[Haoruili46/InertialAR](https://huggingface.co/Haoruili46/InertialAR).
Download them into the repository root with:

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Haoruili46/InertialAR",
    repo_type="model",
    allow_patterns=["data/**", "ckpt/**"],
    local_dir=".",
)
PY
```

See [docs/DATA_AND_CHECKPOINTS.md](docs/DATA_AND_CHECKPOINTS.md) for
dataset-only and checkpoint-only download commands.

After download, the public scripts expect the following layout:

```text
data/
  QM9/processed/
  Drug/processed/
  B3LYP_17M/processed/
ckpt/
  QM9/epoch_2000.pt
  Drug/epoch_250.pt
  B3LYP/epoch_1000.pt
```

Set `DATA_ROOT` or `CKPT_PATH` to override these defaults.

## Generation

The generation scripts reproduce the released checkpoint settings:

```bash
bash scripts/generation/generate_qm9.sh
bash scripts/generation/generate_drug.sh
bash scripts/generation/generate_b3lyp.sh
```

Useful overrides:

```bash
NUM_GENERATE=100 BATCH_SIZE=10 CKPT_PATH=./ckpt/QM9 bash scripts/generation/generate_qm9.sh
```

## Training

The training scripts expose the released model hyperparameters and use `torchrun`:

```bash
bash scripts/train_qm9.sh
bash scripts/train_drug.sh
bash scripts/train_b3lyp.sh
```

Set `NUM_GPUS`, `BATCH_SIZE_PER_GPU`, `DATA_ROOT`, or `OUTPUT_DIR` to adapt them to your machine.

## Generation Evaluation

Generated `.npz` files can be evaluated with EDM-style stability and RDKit validity metrics:

```bash
DATASET_NAME=qm9 bash scripts/evaluation/evaluate_generated.sh path/to/generated.npz
DATASET_NAME=drug bash scripts/evaluation/evaluate_generated.sh path/to/generated.npz
DATASET_NAME=b3lyp bash scripts/evaluation/evaluate_generated.sh path/to/generated.npz
```

Evaluation writes:

- `*_eval_report.txt`
- `*_summary.json`
- `*_unique_smiles.csv`
- `*_log_smiles.csv`

The public evaluation reports stability, validity, and uniqueness. Novelty against the training set is not evaluated. Drug and B3LYP use GEOM-style atom and bond rules; QM9 uses QM9-style rules.

Generation and evaluation are separate by default. To run evaluation immediately after generation:

```bash
AUTO_EVAL=1 bash scripts/generation/generate_qm9.sh
AUTO_EVAL=1 bash scripts/generation/generate_drug.sh
AUTO_EVAL=1 bash scripts/generation/generate_b3lyp.sh
```

## TODO

- [x] **Code Upload**: Initial upload of the core codebase.
- [x] **Pipeline Instructions**: Add environment, training, generation, and evaluation entrypoints.
- [ ] **Code Reorganization**: Clean up and refactor the code for better readability and maintainability.
- [ ] **Provide Checkpoints**: Upload and provide links to the pre-trained model weights (ckpt).
