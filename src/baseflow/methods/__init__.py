"""Collection of individual baseflow separation method implementations."""

from .Boughton import *  # noqa: F401,F403
from .CM import *  # noqa: F401,F403
from .Chapman import *  # noqa: F401,F403
from .Eckhardt import *  # noqa: F401,F403
from .EWMA import *  # noqa: F401,F403
from .Fixed import *  # noqa: F401,F403
from .Furey import *  # noqa: F401,F403
from .LH import *  # noqa: F401,F403
from .Local import *  # noqa: F401,F403
from .Slide import *  # noqa: F401,F403
from .UKIH import *  # noqa: F401,F403
from .Willems import *  # noqa: F401,F403

__all__ = [
    "Boughton",
    "CM",
    "Chapman",
    "Eckhardt",
    "EWMA",
    "Fixed",
    "Furey",
    "LH",
    "Local",
    "Slide",
    "UKIH",
    "Willems",
]
