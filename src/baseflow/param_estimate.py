"""Parameter estimation helpers for baseflow separation methods.

This module provides automatic parameter estimation functions used by
baseflow separation algorithms. Key capabilities include:
- Recession coefficient estimation from streamflow data
- Automatic parameter calibration using grid search with Nash-Sutcliffe Efficiency
- Recession period identification
- Maximum Baseflow Index (BFI) calculation
"""

from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import njit, prange

from .utils import moving_average, multi_arange

__all__ = [
    "recession_coefficient",
    "param_calibrate",
    "param_calibrate_jit",
    "recession_period",
    "maxmium_BFI",
    "Backward",
]


def recession_coefficient(
    Q: npt.NDArray[np.float64],
    strict: npt.NDArray[np.bool_]
) -> float:
    """Estimate recession coefficient from streamflow data.

    This function calculates the recession coefficient 'a' using the method
    described in Eckhardt (2008), focusing on strict baseflow periods to
    ensure accurate estimation.

    Args:
        Q: Streamflow time series array
        strict: Boolean mask identifying strict baseflow periods

    Returns:
        Recession coefficient 'a' (typically 0.9-0.995 for daily data)

    Reference:
        Eckhardt, K. (2008). A comparison of baseflow indices, which were
        calculated with baseflow separation methods. Journal of Hydrology,
        352(1-2), 168-173.

    Note:
        - Uses the 5th percentile of -dQ/Q ratios during baseflow recession
        - Higher values indicate slower baseflow recession
        - Returns exp(-1/K) where K is the recession time constant
    """
    # Calculate flow and derivative at center points
    cQ, dQ = Q[1:-1], (Q[2:] - Q[:-2]) / 2
    mask = strict[1:-1]
    if not np.any(mask):
        mask = np.ones_like(mask, dtype=bool)

    cQ, dQ = cQ[mask], dQ[mask]

    if cQ.size == 0 or dQ.size == 0:
        return 0.95

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(-dQ, cQ, out=np.full_like(dQ, np.nan), where=cQ != 0)

    finite = np.isfinite(ratios)
    if not np.any(finite):
        return 0.95

    ratios = ratios[finite]
    cQ = cQ[finite]
    dQ = dQ[finite]

    # Use np.percentile for robust 5th percentile calculation
    if ratios.size < 10:
        return 0.95
    perc5 = np.percentile(ratios, 5)
    # Find the index of the value closest to the 5th percentile
    idx = np.argmin(np.abs(ratios - perc5))
    K = -cQ[idx] / dQ[idx]
    if not np.isfinite(K) or K == 0:
        return 0.95

    a_value = float(np.exp(-1 / K))
    if not np.isfinite(a_value):
        return 0.95

    return float(np.clip(a_value, 0.5, 0.995))


def param_calibrate(
    param_range: npt.NDArray[np.float64],
    method: Callable,
    Q: npt.NDArray[np.float64],
    b_LH: npt.NDArray[np.float64],
    a: float
) -> float:
    """Calibrate method-specific parameter using grid search.

    This function performs automatic parameter calibration by testing each value
    in the parameter range and selecting the one that minimizes a composite loss
    function based on Nash-Sutcliffe Efficiency (NSE) for both recession and
    non-recession periods, plus a penalty for baseflow exceeding streamflow.

    Args:
        param_range: Array of parameter values to test
        method: Baseflow separation method function (must accept return_exceed=True)
        Q: Streamflow time series array
        b_LH: Baseline LH filter results
        a: Recession coefficient

    Returns:
        Optimal parameter value that minimizes the loss function

    Note:
        - Uses Numba JIT compilation for fast grid search
        - Automatically identifies recession periods for targeted evaluation
        - Loss function balances fit quality and physical constraints
        - Parallel execution across parameter candidates
    """
    idx_rec = recession_period(Q)
    idx_oth = np.full(Q.shape[0], True)
    idx_oth[idx_rec] = False

    Q_sum = float(np.sum(Q))
    bfi_target = float(np.sum(b_LH) / (Q_sum + 1e-10))

    return param_calibrate_jit(
        param_range,
        method,
        Q,
        b_LH,
        a,
        idx_rec,
        idx_oth,
        bfi_target,
        Q_sum,
    )


@njit(parallel=True)
def param_calibrate_jit(
    param_range: npt.NDArray[np.float64],
    method: Callable,
    Q: npt.NDArray[np.float64],
    b_LH: npt.NDArray[np.float64],
    a: float,
    idx_rec: npt.NDArray[np.int64],
    idx_oth: npt.NDArray[np.bool_],
    bfi_target: float,
    Q_sum: float,
) -> float:
    """Numba-accelerated parameter calibration with parallel grid search.

    Internal JIT-compiled function for fast parameter optimization. Uses a
    composite loss function combining:
    1. NSE on log-transformed flows during recession periods
    2. NSE on log-transformed flows during non-recession periods
    3. Penalty for baseflow exceeding streamflow

    Args:
        param_range: Array of parameter values to test
        method: Baseflow separation method (Numba-compatible)
        Q: Streamflow array
        b_LH: LH filter baseline
        a: Recession coefficient
        idx_rec: Indices of recession periods
        idx_oth: Boolean mask for non-recession periods

    Returns:
        Parameter value with minimum loss

    Note:
        This function is JIT-compiled with parallel execution for performance.
        Do not call directly; use param_calibrate() instead.
    """
    logQ = np.log1p(Q)
    loss = np.zeros(param_range.shape)

    # Parallel grid search over parameter range
    for i in prange(param_range.shape[0]):
        p = param_range[i]
        b_exceed = method(Q, b_LH, a, p, return_exceed=True)
        baseflow = b_exceed[:-1]
        f_exd, logb = b_exceed[-1] / Q.shape[0], np.log1p(baseflow)

        # NSE for recession part (log-transformed)
        Q_obs, Q_sim = logQ[idx_rec], logb[idx_rec]
        SS_res = np.sum(np.square(Q_obs - Q_sim))
        SS_tot = np.sum(np.square(Q_obs - np.mean(Q_obs)))
        NSE_rec = (1 - SS_res / (SS_tot + 1e-10)) - 1e-10

        # NSE for non-recession part (log-transformed)
        Q_obs, Q_sim = logQ[idx_oth], logb[idx_oth]
        SS_res = np.sum(np.square(Q_obs - Q_sim))
        SS_tot = np.sum(np.square(Q_obs - np.mean(Q_obs)))
        NSE_oth = (1 - SS_res / (SS_tot + 1e-10)) - 1e-10

        # Composite loss: balance recession fit, overall fit, physical constraints, and BFI realism
        bfi_candidate = np.sum(baseflow) / (Q_sum + 1e-10)
        bfi_penalty = np.abs(bfi_candidate - bfi_target)
        loss[i] = 1 - (1 - (1 - NSE_rec) / (1 - NSE_oth)) * (1 - f_exd) + bfi_penalty

    return param_range[np.argmin(loss)]


def recession_period(Q: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Identify recession periods in streamflow time series.

    Detects periods of sustained flow decrease (recessions) that are at least
    10 days long, and returns indices corresponding to the latter 40% of each
    recession period (where baseflow dominates).

    Args:
        Q: Streamflow time series array

    Returns:
        Array of indices corresponding to late recession periods

    Algorithm:
        1. Apply 3-day moving average smoothing
        2. Identify continuous decreasing segments
        3. Keep segments ≥10 days long
        4. Return indices from 60% point to end of each segment

    Note:
        Late recession periods are assumed to be dominated by baseflow,
        making them ideal for parameter calibration and evaluation.
    """
    idx_dec = np.zeros(Q.shape[0] - 1, dtype=np.int64)
    Q_ave = moving_average(Q, 3)
    idx_dec[1:-1] = (Q_ave[:-1] - Q_ave[1:]) > 0

    # Find start and end of recession segments
    idx_beg = np.where(idx_dec[:-1] - idx_dec[1:] == -1)[0] + 1
    idx_end = np.where(idx_dec[:-1] - idx_dec[1:] == 1)[0] + 1

    # Keep only segments ≥10 days
    idx_keep = (idx_end - idx_beg) >= 10
    idx_beg = idx_beg[idx_keep]
    idx_end = idx_end[idx_keep]

    # Use latter 40% of each recession period
    duration = idx_end - idx_beg
    idx_beg = idx_beg + np.ceil(duration * 0.6).astype(np.int64)

    return multi_arange(idx_beg, idx_end)


def maxmium_BFI(
    Q: npt.NDArray[np.float64],
    b_LH: npt.NDArray[np.float64],
    a: float,
    date: Optional[pd.DatetimeIndex] = None
) -> float:
    """Calculate maximum Baseflow Index (BFI) from backward-pass recession analysis.

    Estimates the upper bound of BFI by applying backward recession analysis
    and selecting the maximum annual BFI value. Useful for constraining the
    BFImax parameter in Eckhardt filter.

    Args:
        Q: Streamflow time series array
        b_LH: LH filter baseline results
        a: Recession coefficient
        date: Optional DatetimeIndex for accurate annual aggregation.
               If None, assumes continuous 365-day years.

    Returns:
        Maximum BFI value (capped at 0.9, or long-term mean if exceeded)

    Reference:
        Eckhardt, K. (2005). How to construct recursive digital filters for
        baseflow separation. Hydrological Processes, 19(2), 507-515.

    Note:
        - Typical BFImax ranges: 0.25 (ephemeral) to 0.80 (perennial)
        - Values >0.9 are replaced with long-term mean BFI for stability
    """
    b = Backward(Q, b_LH, a)

    if date is None:
        # Use fixed 365-day years
        idx_end = b.shape[0] // 365 * 365
        annual_b = np.mean(b[:idx_end].reshape(-1, 365), axis=1)
        annual_Q = np.mean(Q[:idx_end].reshape(-1, 365), axis=1)
        annual_BFI = annual_b / annual_Q
    else:
        # Use calendar years based on DatetimeIndex
        idx_year = date.year - date.year.min()
        counts = np.bincount(idx_year)
        idx_valid = counts > 0
        annual_b = np.bincount(idx_year, weights=b)[idx_valid] / counts[idx_valid]
        annual_Q = np.bincount(idx_year, weights=Q)[idx_valid] / counts[idx_valid]
        annual_BFI = annual_b / annual_Q

    # Select maximum annual BFI, with fallback to long-term mean if >0.9
    BFI_max = np.max(annual_BFI)
    BFI_max = BFI_max if BFI_max < 0.9 else np.sum(annual_b) / np.sum(annual_Q)
    return BFI_max


@njit
def Backward(
    Q: npt.NDArray[np.float64],
    b_LH: npt.NDArray[np.float64],
    a: float
) -> npt.NDArray[np.float64]:
    """Apply backward recession analysis to estimate baseflow.

    Propagates baseflow backward in time using the recession equation:
    b[t-1] = b[t] / a, subject to physical constraints (baseflow ≤ streamflow).

    Args:
        Q: Streamflow time series array
        b_LH: LH filter results (used for initialization)
        a: Recession coefficient

    Returns:
        Baseflow array estimated from backward recession

    Algorithm:
        1. Initialize last value from LH filter
        2. Propagate backward: b[t-1] = b[t] / a
        3. Apply constraints: b[t] ≤ Q[t] and handle zero flows

    Note:
        Numba-compiled for performance. Backward pass complements forward
        filtering by emphasizing recession characteristics.
    """
    b = np.zeros(Q.shape[0])
    b[-1] = b_LH[-1]

    # Backward propagation with recession equation
    for i in range(Q.shape[0] - 1, 0, -1):
        b[i - 1] = b[i] / a

        # Handle zero flow case
        if b[i] == 0:
            b[i - 1] = Q[i - 1]

        # Physical constraint: baseflow cannot exceed streamflow
        if b[i - 1] > Q[i - 1]:
            b[i - 1] = Q[i - 1]

    return b
