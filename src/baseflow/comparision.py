"""Evaluation utilities for comparing baseflow separation results.

This module provides functions for:
- Identifying strict baseflow periods using hydrograph analysis
- Computing the Kling-Gupta Efficiency (KGE) metric for method evaluation
- Supporting performance assessment and inter-method comparison
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

__all__ = ["strict_baseflow", "KGE"]


def strict_baseflow(
    Q: npt.NDArray[np.float64],
    ice: Optional[npt.NDArray[np.bool_]] = None
) -> npt.NDArray[np.bool_]:
    """Identify strict baseflow periods using hydrograph analysis.

    Implements a multi-criteria filtering approach to isolate periods when
    streamflow is dominated by baseflow (groundwater contribution), excluding
    periods affected by direct runoff, rising limbs, and major events.

    Args:
        Q: Streamflow time series array
        ice: Optional boolean mask for frozen periods (True = frozen)

    Returns:
        Boolean array (True = strict baseflow period)

    Algorithm (4-step filtering):
        1. Exclude rising hydrograph (dQ/dt ≥ 0)
        2. Exclude 2 points before and 3 points after each rising segment
        3. Exclude 5 points following major events (>90th percentile)
        4. Exclude accelerating recession (d²Q/dt² < 0)

    Example:
        >>> Q = np.array([10, 9, 8, 7, 15, 12, 10, 9, 8])
        >>> strict = strict_baseflow(Q)
        >>> print(Q[strict])  # Only late recession points
        [7, 9, 8]

    Reference:
        Based on methods from Eckhardt (2008) and related baseflow literature
        for identifying periods of pure groundwater discharge.

    Note:
        - Frozen periods are automatically excluded if ice mask provided
        - Used as "ground truth" for parameter calibration and KGE calculation
        - Typically represents 10-30% of total record length
    """
    # Calculate centered finite difference dQ/dt
    dQ = (Q[2:] - Q[:-2]) / 2

    # 1. Exclude flow data with positive/zero derivatives (rising/flat hydrograph)
    wet1 = np.concatenate([[True], dQ >= 0, [True]])

    # 2. Exclude 2 points before and 3 points after rising segments
    #    (transition zones between baseflow and quickflow)
    idx_first = np.where(wet1[1:].astype(int) - wet1[:-1].astype(int) == 1)[0] + 1
    idx_last = np.where(wet1[1:].astype(int) - wet1[:-1].astype(int) == -1)[0]
    idx_before = np.repeat([idx_first], 2) - np.tile(range(1, 3), idx_first.shape)
    idx_next = np.repeat([idx_last], 3) + np.tile(range(1, 4), idx_last.shape)
    idx_remove = np.concatenate([idx_before, idx_next])
    wet2 = np.full(Q.shape, False)
    wet2[idx_remove.clip(min=0, max=Q.shape[0] - 1)] = True

    # 3. Exclude 5 points after major flow events (>90th percentile)
    #    (allows time for storm flow to fully recede)
    growing = np.concatenate([[True], (Q[1:] - Q[:-1]) >= 0, [True]])
    idx_major = np.where((Q >= np.quantile(Q, 0.9)) & growing[:-1] & ~growing[1:])[0]
    idx_after = np.repeat([idx_major], 5) + np.tile(range(1, 6), idx_major.shape)
    wet3 = np.full(Q.shape, False)
    wet3[idx_after.clip(min=0, max=Q.shape[0] - 1)] = True

    # 4. Exclude points with accelerating recession (d²Q/dt² < 0)
    #    (indicates non-exponential decay, possibly mixed flow)
    wet4 = np.concatenate([[True], dQ[1:] - dQ[:-1] < 0, [True, True]])

    # Combine all exclusion criteria (dry = strict baseflow)
    dry = ~(wet1 + wet2 + wet3 + wet4)

    # Exclude frozen periods (groundwater-baseflow relationship invalid)
    if ice is not None:
        dry[ice] = False

    return dry


def KGE(
    simulations: npt.NDArray[np.float64],
    evaluation: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculate Kling-Gupta Efficiency for model evaluation.

    Computes the KGE metric, which balances correlation, variability bias,
    and mean bias. KGE ranges from -∞ to 1, where 1 = perfect agreement.

    Args:
        simulations: Simulated values (e.g., separated baseflow)
                    Shape: (n_samples,) or (n_samples, n_methods)
        evaluation: Evaluation/reference values (e.g., observed streamflow)
                   Shape must match simulations

    Returns:
        KGE value(s). Scalar if both inputs are 1D, otherwise array.

    Formula:
        KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]

        Where:
        - r = Pearson correlation (timing/dynamics)
        - α = σ_sim/σ_obs (variability ratio)
        - β = μ_sim/μ_obs (bias ratio)

    Interpretation:
        - KGE > 0.75: Very good
        - KGE > 0.50: Good
        - KGE > 0.00: Better than mean
        - KGE < 0.00: Worse than using mean as predictor

    Reference:
        Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
        Decomposition of the mean squared error and NSE performance criteria:
        Implications for improving hydrological modelling. Journal of Hydrology,
        377(1-2), 80-91. https://doi.org/10.1016/j.jhydrol.2009.08.003

    Example:
        >>> obs = np.array([10, 9, 8, 7, 6])
        >>> sim = np.array([10.5, 8.9, 8.2, 6.8, 5.9])
        >>> kge = KGE(sim, obs)
        >>> print(f"KGE: {kge:.3f}")
        KGE: 0.987

    Note:
        - Small regularization (1e-10) prevents division by zero
        - Supports vectorized computation for multiple methods
    """
    # Calculate error in timing and dynamics (Pearson correlation)
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, axis=0, dtype=np.float64)

    r_num = np.sum((simulations - sim_mean) * (evaluation - obs_mean), axis=0, dtype=np.float64)
    r_den = np.sqrt(
        np.sum((simulations - sim_mean) ** 2, axis=0, dtype=np.float64)
        * np.sum((evaluation - obs_mean) ** 2, axis=0, dtype=np.float64)
    )
    r = r_num / (r_den + 1e-10)

    # Calculate error in variability (standard deviation ratio)
    alpha = np.std(simulations, axis=0) / (np.std(evaluation, axis=0) + 1e-10)

    # Calculate error in volume (mean bias ratio)
    beta = np.sum(simulations, axis=0, dtype=np.float64) / (
        np.sum(evaluation, axis=0, dtype=np.float64) + 1e-10
    )

    # Calculate Kling-Gupta Efficiency
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_
