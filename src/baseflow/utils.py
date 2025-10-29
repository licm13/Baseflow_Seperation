"""Utility helpers for data cleaning, masking and numerical routines.

This module provides general-purpose utility functions for:
- Streamflow data cleaning and validation
- Frozen period detection and masking
- Numerical operations (moving averages, range generation)
- Geographic coordinate transformations
- Method name formatting and validation
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit

__all__ = [
    "clean_streamflow",
    "exist_ice",
    "moving_average",
    "multi_arange",
    "geo2imagexy",
    "format_method",
]


def clean_streamflow(
    series: pd.Series,
) -> Tuple[pd.DatetimeIndex, npt.NDArray[np.float64]]:
    """Clean and validate streamflow time series data.

    Removes missing values (NaN, Inf) and converts negative flows to positive.

    Args:
        series: Pandas Series with DatetimeIndex and streamflow values

    Returns:
        Tuple of (cleaned dates, cleaned flow values)

    Note:
        - Negative values are converted to absolute values
        - Missing/infinite values are removed
        - Returned arrays have matching lengths
    """
    date, Q = series.index, series.values.astype(float)
    has_value = np.isfinite(Q)
    date, Q = date[has_value], np.abs(Q[has_value])
    return date, Q


def exist_ice(
    date: Optional[pd.DatetimeIndex],
    ice_period: Optional[Union[npt.NDArray[np.bool_], Tuple[List[int], List[int]]]]
) -> Optional[npt.NDArray[np.bool_]]:
    """Create boolean mask for frozen (ice-covered) periods.

    Generates a mask identifying dates during frozen periods when the
    groundwater-baseflow relationship may be invalidated by ice cover.

    Args:
        date: DatetimeIndex for the time series
        ice_period: Frozen period specification. Can be:
            - Boolean array of length 12 (True for frozen months)
            - Tuple: [(start_month, start_day), (end_month, end_day)]
            - None: Returns None (no masking)

    Returns:
        Boolean array (True = frozen) matching date length, or None

    Examples:
        >>> # Northern hemisphere winter: Nov 1 to Mar 31
        >>> ice = exist_ice(dates, ([11, 1], [3, 31]))
        >>>
        >>> # Monthly mask (Dec, Jan, Feb frozen)
        >>> months = np.array([False]*11 + [True] + [True]*2 + [False]*9)
        >>> ice = exist_ice(dates, months)

    Note:
        - Handles cross-year periods (e.g., Nov-Mar spans calendar year boundary)
        - Used to exclude frozen periods from recession analysis
    """
    if (date is None) or (ice_period is None):
        return None

    # Case 1: Monthly boolean mask
    if isinstance(ice_period, np.ndarray):
        return np.isin(date.month, np.where(ice_period)[0] + 1)

    # Case 2: Date range specification
    beg, end = ice_period

    # Check if period crosses year boundary
    if (end[0] > beg[0]) or ((end[0] == beg[0]) & (end[1] > beg[1])):
        # Within-year period (e.g., May to September)
        ice = (
            ((date.month > beg[0]) & (date.month < end[0]))
            | ((date.month == beg[0]) & (date.day >= beg[1]))
            | ((date.month == end[0]) & (date.day <= end[1]))
        )
    else:
        # Cross-year period (e.g., November to March)
        ice = (
            ((date.month > beg[0]) | (date.month < end[0]))
            | ((date.month == beg[0]) & (date.day >= beg[1]))
            | ((date.month == end[0]) & (date.day <= end[1]))
        )
    return ice


def moving_average(
    x: npt.NDArray[np.float64],
    w: int
) -> npt.NDArray[np.float64]:
    """Calculate moving average using convolution.

    Applies a simple moving average filter with equal weights.

    Args:
        x: Input array
        w: Window width (number of points to average)

    Returns:
        Smoothed array (length = len(x) - 2*(w-1))

    Note:
        - Uses np.convolve for efficient computation
        - Edge handling: Returns only fully-overlapping windows
        - Output is shorter than input by 2*(w-1) points
    """
    res = np.convolve(x, np.ones(w)) / w
    return res[w - 1 : -w + 1]


@njit
def multi_arange(
    starts: npt.NDArray[np.int64],
    stops: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Efficiently concatenate multiple np.arange() outputs.

    Numba-compiled function to create an array containing concatenated
    ranges [starts[i], stops[i]) for all i.

    Args:
        starts: Array of range start values (inclusive)
        stops: Array of range stop values (exclusive)

    Returns:
        Concatenated array of all ranges

    Example:
        >>> multi_arange(np.array([0, 5, 10]), np.array([3, 8, 12]))
        array([0, 1, 2, 5, 6, 7, 10, 11])

    Note:
        - JIT-compiled with Numba for performance
        - Much faster than using np.concatenate([np.arange(s, e) for s, e in ...])
        - Used in recession period identification
    """
    pos = 0
    cnt = np.sum(stops - starts, dtype=np.int64)
    res = np.zeros((cnt,), dtype=np.int64)

    for i in range(starts.size):
        num = stops[i] - starts[i]
        res[pos : pos + num] = np.arange(starts[i], stops[i])
        pos += num

    return res


def geo2imagexy(
    x: float,
    y: float
) -> Tuple[int, int]:
    """Convert geographic coordinates to raster image indices.

    Transforms lon/lat coordinates to col/row indices for the included
    permafrost data raster (0.5° resolution, global extent).

    Args:
        x: Longitude in decimal degrees (-180 to 180)
        y: Latitude in decimal degrees (-90 to 90)

    Returns:
        Tuple of (column, row) indices for raster access

    Example:
        >>> col, row = geo2imagexy(-120.5, 45.2)
        >>> ice_data = thawed_mask[:, row, col]

    Note:
        - Assumes 0.5° grid resolution
        - Origin at (-180, 90) in lon/lat space
        - Used for extracting frozen period data from global permafrost raster
    """
    # Affine transformation matrix for 0.5° grid
    a = np.array([[0.5, 0.0], [0.0, -0.5]])
    b = np.array([x - -180, y - 90])
    col, row = np.linalg.solve(a, b) - 0.5
    return np.round(col).astype(int), np.round(row).astype(int)


def format_method(method: Union[str, List[str]]) -> List[str]:
    """Normalize method name specification to list format.

    Converts various method specifications into a consistent list format
    for internal processing.

    Args:
        method: Method specification. Can be:
            - "all": Expands to all 12 available methods
            - Single method name string (e.g., "LH")
            - List of method names (returned as-is)

    Returns:
        List of method names

    Example:
        >>> format_method("all")
        ['UKIH', 'Local', 'Fixed', ..., 'Willems']
        >>> format_method("Eckhardt")
        ['Eckhardt']
        >>> format_method(["LH", "Chapman"])
        ['LH', 'Chapman']

    Available methods:
        - Graphical: UKIH, Local, Fixed, Slide
        - Digital filters: LH, Chapman, CM, Eckhardt, EWMA, Willems
        - Parameterized: Boughton, Furey

    Note:
        No validation is performed on method names.
        Invalid names will cause errors in separation functions.
    """
    if method == "all":
        method = [
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
    elif isinstance(method, str):
        method = [method]
    return method
