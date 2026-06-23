"""Lazy dataset exports.

Some dataset implementations require optional packages that are not needed by
other training paths. Keep package-level imports lightweight so QM9 training
does not fail because an unrelated dataset dependency is missing.
"""

from importlib import import_module

_DATASET_EXPORTS = {
    "DatasetInertialSeqQM9": ".dataset_qm9",
    "DatasetInertialSeqDrug": ".dataset_drug",
    "DatasetInertialSeqB3LYP": ".dataset_b3lyp",
}

__all__ = sorted(_DATASET_EXPORTS)


def __getattr__(name):
    if name not in _DATASET_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_DATASET_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
