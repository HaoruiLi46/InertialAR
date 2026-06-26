"""Public generation evaluation entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from InertialAR.evaluation.eval_edm import (
    convert_npz_to_text,
    invariant_seq_for_edm_eval,
    main as eval_main,
    run_evaluation,
)

__all__ = ["convert_npz_to_text", "invariant_seq_for_edm_eval", "run_evaluation", "main"]


def main():
    return eval_main()


if __name__ == "__main__":
    main()
