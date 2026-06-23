from .config import InertialARConfig

__all__ = ["InertialAR", "InertialARConfig"]


def __getattr__(name):
    if name == "InertialAR":
        from .model import InertialAR

        return InertialAR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
