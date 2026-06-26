# Data and Checkpoints

The released preprocessed datasets and checkpoints are hosted in the
[Haoruili46/InertialAR](https://huggingface.co/Haoruili46/InertialAR)
Hugging Face repository.

## Download Everything

Run the following commands from the repository root:

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

This downloads both the preprocessed datasets and the checkpoints.

If you already have the Hugging Face `hf` CLI installed, the equivalent command
is:

```bash
hf download Haoruili46/InertialAR \
  --repo-type model \
  --include "data/**" \
  --include "ckpt/**" \
  --local-dir .
```

## Download Only Checkpoints

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Haoruili46/InertialAR",
    repo_type="model",
    allow_patterns=["ckpt/**"],
    local_dir=".",
)
PY
```

Expected checkpoint layout:

```text
ckpt/
  QM9/epoch_2000.pt
  Drug/epoch_250.pt
  B3LYP/epoch_1000.pt
```

Equivalent `hf` CLI command:

```bash
hf download Haoruili46/InertialAR \
  --repo-type model \
  --include "ckpt/**" \
  --local-dir .
```

## Download Only Data

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Haoruili46/InertialAR",
    repo_type="model",
    allow_patterns=["data/**"],
    local_dir=".",
)
PY
```

Expected data layout:

```text
data/
  QM9/
    raw/
    processed/
  Drug/
    processed/
  B3LYP_17M/
    processed/
```

Equivalent `hf` CLI command:

```bash
hf download Haoruili46/InertialAR \
  --repo-type model \
  --include "data/**" \
  --local-dir .
```

The training and generation scripts use `DATA_ROOT=./data` and the checkpoint
folders under `./ckpt` by default. Override `DATA_ROOT` or `CKPT_PATH` if you
store the assets somewhere else.
