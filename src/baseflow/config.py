"""Configuration module for baseflow separation parameters.

This module centralizes all parameter ranges and default values used
across different baseflow separation methods.
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import numpy.typing as npt


@dataclass
class MethodConfig:
    """Configuration for a specific baseflow separation method.

    Attributes:
        param_range: Array defining the search space for parameter calibration
        description: Human-readable description of the method
        requires_area: Whether the method requires drainage area information
        requires_recession_coef: Whether the method requires recession coefficient
    """
    param_range: npt.NDArray[np.float64] | None = None
    description: str = ""
    requires_area: bool = False
    requires_recession_coef: bool = False


# Default parameter ranges for each method
DEFAULT_PARAM_RANGES: Dict[str, MethodConfig] = {
    "Boughton": MethodConfig(
        param_range=np.arange(0.0001, 0.1, 0.0001),
        description="Boughton method with parameter C (recession constant)",
        requires_recession_coef=True,
    ),
    "Furey": MethodConfig(
        param_range=np.arange(0.01, 10, 0.01),
        description="Furey method with parameter A (scaling factor)",
        requires_recession_coef=True,
    ),
    "Eckhardt": MethodConfig(
        param_range=np.arange(0.001, 1, 0.001),
        description="Eckhardt two-parameter digital filter with BFImax",
        requires_recession_coef=True,
    ),
    "EWMA": MethodConfig(
        param_range=np.arange(0.0001, 0.1, 0.0001),
        description="Exponential Weighted Moving Average filter with smoothing parameter",
        requires_recession_coef=False,
    ),
    "Willems": MethodConfig(
        param_range=np.arange(0.001, 1, 0.001),
        description="Willems method with parameter w (weighting factor)",
        requires_recession_coef=True,
    ),
    "Chapman": MethodConfig(
        description="Chapman digital filter",
        requires_recession_coef=True,
    ),
    "CM": MethodConfig(
        description="Combined method (Chapman variant)",
        requires_recession_coef=True,
    ),
    "LH": MethodConfig(
        description="Lyne-Hollick digital filter",
        requires_recession_coef=False,
    ),
    "UKIH": MethodConfig(
        description="UK Institute of Hydrology method",
        requires_recession_coef=False,
    ),
    "Local": MethodConfig(
        description="Local minimum method (HYSEP)",
        requires_area=True,
        requires_recession_coef=False,
    ),
    "Fixed": MethodConfig(
        description="Fixed interval method (HYSEP)",
        requires_area=True,
        requires_recession_coef=False,
    ),
    "Slide": MethodConfig(
        description="Sliding interval method (HYSEP)",
        requires_area=True,
        requires_recession_coef=False,
    ),
}


# All available methods
ALL_METHODS = [
    "UKIH",
    "Local",
    "Fixed",
    "Slide",
    "LH",
    "Chapman",
    "CM",
    "Boughton",
    "Furey",
    "Eckhardt",
    "EWMA",
    "Willems",
]


@dataclass
class SeparationConfig:
    """Global configuration for baseflow separation workflows.

    Attributes:
        default_ice_period: Default frozen period definition [(month, day), (month, day)]
        lh_beta: Default beta parameter for Lyne-Hollick filter
        strict_baseflow_quantile: Quantile threshold for identifying major flow events
        recession_period_min_length: Minimum length (days) for recession period identification
        recession_strict_percentile: Percentile for strict recession coefficient estimation
    """
    default_ice_period: tuple = field(default_factory=lambda: ([11, 1], [3, 31]))
    lh_beta: float = 0.925
    strict_baseflow_quantile: float = 0.9
    recession_period_min_length: int = 10
    recession_strict_percentile: float = 0.05


# Global configuration instance
GLOBAL_CONFIG = SeparationConfig()


def get_param_range(method: str) -> npt.NDArray[np.float64] | None:
    """Get parameter calibration range for a given method.

    Args:
        method: Name of the baseflow separation method

    Returns:
        Array of parameter values to search over, or None if no calibration needed

    Raises:
        ValueError: If method name is not recognized
    """
    if method not in DEFAULT_PARAM_RANGES:
        raise ValueError(
            f"Unknown method '{method}'. Available methods: {list(DEFAULT_PARAM_RANGES.keys())}"
        )
    return DEFAULT_PARAM_RANGES[method].param_range


def update_param_range(
    method: str,
    start: float,
    stop: float,
    step: float
) -> None:
    """Update the parameter search range for a specific method.

    Args:
        method: Name of the method to update
        start: Start of parameter range
        stop: End of parameter range (exclusive)
        step: Step size for parameter grid

    Example:
        >>> update_param_range("Eckhardt", 0.01, 0.99, 0.01)
    """
    if method not in DEFAULT_PARAM_RANGES:
        raise ValueError(f"Unknown method '{method}'")
    DEFAULT_PARAM_RANGES[method].param_range = np.arange(start, stop, step)
