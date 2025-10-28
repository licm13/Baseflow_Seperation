"""Public package interface for the baseflow separation toolkit."""

from pathlib import Path

from . import comparision, methods, param_estimate, separation, utils
from .comparision import *  # noqa: F401,F403
from .methods import *  # noqa: F401,F403
from .param_estimate import *  # noqa: F401,F403
from .separation import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403

example = Path(__file__).resolve().parent / "example.csv"

__all__ = [  # pragma: no cover - 动态导出的集合
    *getattr(comparision, "__all__", []),
    *getattr(methods, "__all__", []),
    *getattr(param_estimate, "__all__", []),
    *getattr(separation, "__all__", []),
    *getattr(utils, "__all__", []),
    "example",
]
