# Environment Setup

This guide provides the recommended environment for reproducing InertialAR
training and evaluation.

The project has been smoke-tested with:

- Python 3.11
- CUDA 12.x
- PyTorch 2.3.0
- torchvision 0.18.0
- torchaudio 2.3.0
- FlashAttention 2.6.2

## 1. Create a Conda Environment

```bash
conda create -n InertialTransformer python=3.11 -y
conda activate InertialTransformer
```

## 2. Install Python Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r env.txt
```

`env.txt` pins the PyTorch 2.3.0 stack and the remaining Python dependencies.

## 3. Install FlashAttention

Install the Linux x86_64 wheel for Python 3.11, CUDA 12.x, and PyTorch 2.3:

```bash
python -m pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.2/flash_attn-2.6.2+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

## 4. Install InertialAR in Editable Mode

```bash
python -m pip install -e .
```

## 5. Verify the Environment

```bash
python - <<'PY'
import torch
import flash_attn
import rdkit
import torch_geometric

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("environment ok")
PY
```

For GPU training, `cuda available` should be `True`.
