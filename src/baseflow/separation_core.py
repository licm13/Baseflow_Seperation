"""Core utilities for baseflow separation workflows.

This module provides reusable components for single-station and multi-station
baseflow separation, including:
- Flow DataFrame preparation
- Baseline LH computation (cached to avoid redundant calculations)
- Per-station worker functions
- Multi-station batch processing with optional parallelization

These utilities are used by the high-level separation API to reduce code
duplication and improve performance.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from .methods.lh_core import lh_filter
from .utils import clean_streamflow

__all__ = [
    "prepare_flow_df",
    "compute_baseline_lh_for_df",
    "process_station_worker",
    "run_multi_station",
]


def prepare_flow_df(series: pd.Series) -> Tuple[pd.DatetimeIndex, npt.NDArray[np.float64]]:
    """Prepare a flow time series for baseflow separation.

    Cleans the streamflow data and ensures proper datetime indexing.

    Args:
        series: Time series of streamflow data

    Returns:
        Tuple of (datetime index, cleaned flow array)
    """
    return clean_streamflow(series)


def compute_baseline_lh_for_df(
    Q: npt.NDArray[np.float64],
    beta: float = 0.925
) -> npt.NDArray[np.float64]:
    """Compute baseline LH filter for a flow array.

    This is a convenience wrapper around lh_filter that can be used
    when only the baseline LH is needed (not exceedance counts).

    Args:
        Q: Flow array
        beta: Filter parameter for LH (default 0.925)

    Returns:
        Baseline LH filter result
    """
    return lh_filter(Q, beta=beta, return_exceed=False)


def process_station_worker(
    station_id: str,
    flow_series: pd.Series,
    separation_func: Callable,
    area: Optional[float] = None,
    ice: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """Worker function for processing a single station.

    This function is designed to be used in both sequential and parallel
    processing contexts.

    Args:
        station_id: Station identifier
        flow_series: Streamflow time series for this station
        separation_func: Function to perform the separation (e.g., single())
        area: Optional drainage area
        ice: Optional ice period specification
        **kwargs: Additional arguments to pass to separation_func

    Returns:
        Dictionary containing:
        - 'station_id': The station identifier
        - 'success': Boolean indicating success/failure
        - 'result': Separation results if successful, None otherwise
        - 'error': Error message if failed, None otherwise
    """
    try:
        result = separation_func(
            flow_series,
            area=area,
            ice=ice,
            **kwargs
        )
        return {
            'station_id': station_id,
            'success': True,
            'result': result,
            'error': None
        }
    except Exception as e:
        return {
            'station_id': station_id,
            'success': False,
            'result': None,
            'error': str(e)
        }


def run_multi_station(
    df: pd.DataFrame,
    df_sta: Optional[pd.DataFrame],
    method: List[str],
    separation_func: Callable,
    return_bfi: bool = False,
    return_kge: bool = False,
    parallel: bool = False,
    max_workers: Optional[int] = None,
    thawed_data: Optional[npt.NDArray] = None,
    geo2imagexy_func: Optional[Callable] = None,
) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run baseflow separation for multiple stations.

    This is a core utility function that handles the iteration over stations,
    optional parallel processing, and result aggregation.

    Args:
        df: DataFrame with streamflow data (index: dates, columns: station IDs)
        df_sta: Optional DataFrame with station metadata
        method: List of method names to apply
        separation_func: Function to call for each station (typically single())
        return_bfi: Whether to calculate BFI
        return_kge: Whether to calculate KGE
        parallel: Whether to use parallel processing (experimental)
        max_workers: Maximum number of parallel workers (None = use default)
        thawed_data: Optional pre-loaded thawed period data
        geo2imagexy_func: Optional function to convert geo coordinates to image xy

    Returns:
        Tuple of (results_dict, bfi_df, kge_df) where:
        - results_dict: Dict mapping method names to baseflow DataFrames
        - bfi_df: BFI DataFrame if return_bfi=True, else None
        - kge_df: KGE DataFrame if return_kge=True, else None
    """
    # Create DataFrames to store results
    dfs = {m: pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float)
           for m in method}

    df_bfi = None
    df_kge = None
    if return_bfi:
        df_bfi = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)
    if return_kge:
        df_kge = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)

    # Load thawed data if needed and not provided
    if thawed_data is None and df_sta is not None and geo2imagexy_func is not None:
        with np.load(Path(__file__).parent / "thawed.npz") as f:
            thawed_data = f["thawed"]

    # Helper to extract station metadata
    def get_station_params(s: str) -> Tuple[Optional[float], Optional[Any]]:
        area, ice = None, None
        if df_sta is not None:
            to_num = lambda col: (
                pd.to_numeric(df_sta.loc[s, col], errors="coerce")
                if col in df_sta.columns
                else np.nan
            )
            if np.isfinite(to_num("area")):
                area = to_num("area")
            if (thawed_data is not None and geo2imagexy_func is not None and
                np.isfinite(to_num("lon")) and np.isfinite(to_num("lat"))):
                c, r = geo2imagexy_func(to_num("lon"), to_num("lat"))
                ice_arr = ~thawed_data[:, r, c]
                ice = ([11, 1], [3, 31]) if ice_arr.all() else ice_arr
        return area, ice

    # Process stations
    if parallel and max_workers != 1:
        # Parallel processing (experimental)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for s in df.columns:
                area, ice = get_station_params(s)
                future = executor.submit(
                    process_station_worker,
                    s,
                    df[s],
                    separation_func,
                    area=area,
                    ice=ice,
                    method=method,
                    return_kge=return_kge
                )
                futures[future] = s

            # Collect results with progress bar
            for future in tqdm(futures, total=len(futures)):
                s = futures[future]
                try:
                    worker_result = future.result()
                    if worker_result['success']:
                        b, KGEs = worker_result['result']
                        for m in method:
                            dfs[m].loc[b.index, s] = b[m]
                        if return_bfi:
                            df_bfi.loc[s] = b.sum() / df.loc[b.index, s].abs().sum()
                        if return_kge:
                            df_kge.loc[s] = KGEs
                    else:
                        print(f"\nFailed to separate baseflow for station {s}")
                except Exception:
                    print(f"\nFailed to separate baseflow for station {s}")
    else:
        # Sequential processing (default)
        for s in tqdm(df.columns, total=df.shape[1]):
            try:
                area, ice = get_station_params(s)
                b, KGEs = separation_func(
                    df[s],
                    ice=ice,
                    area=area,
                    method=method,
                    return_kge=return_kge
                )
                # Write into already created dataframe
                for m in method:
                    dfs[m].loc[b.index, s] = b[m]
                if return_bfi:
                    df_bfi.loc[s] = b.sum() / df.loc[b.index, s].abs().sum()
                if return_kge:
                    df_kge.loc[s] = KGEs
            except Exception:
                print(f"\nFailed to separate baseflow for station {s}")

    return dfs, df_bfi, df_kge
