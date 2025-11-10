"""Centralized, optimized LH (Lyne-Hollick) filter implementation.

This module provides a unified LH filter that uses numba JIT acceleration when
available and falls back to pure Python when numba is not installed.

The LH filter is one of the most widely used baseflow separation methods,
using recursive digital filtering to separate high-frequency (quickflow)
and low-frequency (baseflow) components.

Reference:
    Lyne, V., & Hollick, M. (1979). Stochastic time-variable rainfall-runoff
    modelling. Institute of Engineers Australia National Conference. (pp. 89-93).
    Perth.
"""

import numpy as np
import numpy.typing as npt

__all__ = ["lh_filter"]

# Try to import numba for JIT acceleration
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define a no-op decorator when numba is not available
    def njit(func):
        return func


if HAS_NUMBA:
    @njit
    def _lh_filter_numba(
        Q: npt.NDArray[np.float64],
        beta: float = 0.925,
        return_exceed: bool = False
    ) -> npt.NDArray[np.float64]:
        """Numba-accelerated LH digital filter implementation.

        Uses two-pass filtering strategy:
        1. Forward pass: compute initial baseflow
        2. Backward pass: filter again on initial results for temporal symmetry

        Args:
            Q: Flow array
            beta: Filter parameter, default 0.925
            return_exceed: Whether to return exceedance count (baseflow > flow)

        Returns:
            If return_exceed=False: baseflow array
            If return_exceed=True: baseflow array (with exceed count appended)

        Note:
            - Physical constraint: b[i] ≤ Q[i] (baseflow cannot exceed total flow)
        """
        # Initialize result array
        if return_exceed:
            b = np.zeros(Q.shape[0] + 1)  # Last element stores exceed count
        else:
            b = np.zeros(Q.shape[0])

        # ========================================================================
        # First pass: forward filtering
        # ========================================================================
        b[0] = Q[0]  # Initial value: assume initial baseflow equals initial flow

        for i in range(Q.shape[0] - 1):
            # LH recursion formula: current baseflow = β * previous baseflow + (1-β)/2 * (current flow + previous flow)
            b[i + 1] = beta * b[i] + (1 - beta) / 2 * (Q[i] + Q[i + 1])

            # Physical constraint: baseflow cannot exceed total flow
            if b[i + 1] > Q[i + 1]:
                b[i + 1] = Q[i + 1]
                if return_exceed:
                    b[-1] += 1  # Record exceed count (for calibration evaluation)

        # ========================================================================
        # Second pass: backward filtering (operates on first pass results)
        # ========================================================================
        b1 = np.copy(b)  # Save first pass results

        for i in range(Q.shape[0] - 2, -1, -1):  # Start from second-to-last element
            # Backward recursion formula: similar to forward, but reversed
            b[i] = beta * b[i + 1] + (1 - beta) / 2 * (b1[i + 1] + b1[i])

            # Physical constraint: backward filtered baseflow also cannot exceed first pass result
            if b[i] > b1[i]:
                b[i] = b1[i]
                if return_exceed:
                    b[-1] += 1

        return b


def _lh_filter_python(
    Q: npt.NDArray[np.float64],
    beta: float = 0.925,
    return_exceed: bool = False
) -> npt.NDArray[np.float64]:
    """Pure Python LH digital filter implementation (fallback when numba unavailable).

    This implements the same algorithm as the numba version but without JIT compilation.
    Used as a fallback when numba is not installed.

    Args:
        Q: Flow array
        beta: Filter parameter, default 0.925
        return_exceed: Whether to return exceedance count (baseflow > flow)

    Returns:
        If return_exceed=False: baseflow array
        If return_exceed=True: baseflow array (with exceed count appended)
    """
    # Initialize result array
    if return_exceed:
        b = np.zeros(Q.shape[0] + 1)
    else:
        b = np.zeros(Q.shape[0])

    # First pass: forward filtering
    b[0] = Q[0]
    for i in range(Q.shape[0] - 1):
        b[i + 1] = beta * b[i] + (1 - beta) / 2 * (Q[i] + Q[i + 1])
        if b[i + 1] > Q[i + 1]:
            b[i + 1] = Q[i + 1]
            if return_exceed:
                b[-1] += 1

    # Second pass: backward filtering
    b1 = np.copy(b)
    for i in range(Q.shape[0] - 2, -1, -1):
        b[i] = beta * b[i + 1] + (1 - beta) / 2 * (b1[i + 1] + b1[i])
        if b[i] > b1[i]:
            b[i] = b1[i]
            if return_exceed:
                b[-1] += 1

    return b


def lh_filter(
    flow: npt.NDArray[np.float64],
    beta: float = 0.925,
    return_exceed: bool = False
) -> npt.NDArray[np.float64]:
    """Apply LH (Lyne-Hollick) digital filter for baseflow separation.

    This is the main entry point for the centralized LH filter implementation.
    It automatically uses numba-accelerated version when available, otherwise
    falls back to pure Python.

    The LH filter uses a two-pass recursive digital filter to separate baseflow
    from total streamflow, ensuring temporal symmetry in the results.

    Args:
        flow: Streamflow time series array
        beta: Filter parameter (0 < beta < 1). Default is 0.925 as recommended
            by Nathan & McMahon (1990). Larger values produce smoother baseflow.
        return_exceed: If True, appends the count of exceedances (times when
            baseflow exceeded total flow) to the end of the result array.
            Useful for calibration diagnostics.

    Returns:
        Baseflow time series array. If return_exceed=True, the last element
        contains the exceedance count.

    Example:
        >>> import numpy as np
        >>> flow = np.array([10, 15, 12, 8, 6, 7, 9, 11])
        >>> baseflow = lh_filter(flow, beta=0.925)
        >>> print(baseflow)

    Note:
        - The filter enforces the physical constraint that baseflow cannot
          exceed total streamflow at any time step.
        - Uses numba JIT compilation when numba is installed for better performance.
        - Falls back to pure Python when numba is not available.
    """
    if HAS_NUMBA:
        return _lh_filter_numba(flow, beta=beta, return_exceed=return_exceed)
    else:
        return _lh_filter_python(flow, beta=beta, return_exceed=return_exceed)
