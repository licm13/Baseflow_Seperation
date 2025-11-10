"""High-level APIs for running baseflow separation workflows.

This module provides user-facing functions for applying baseflow separation
algorithms to hydrological time series data. It supports both single-station
and multi-station batch processing with automatic parameter calibration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from .comparision import KGE, strict_baseflow
from .config import ALL_METHODS, get_param_range
from .methods import (
    Boughton,
    Chapman,
    CM,
    Eckhardt,
    EWMA,
    Fixed,
    Furey,
    LH,
    Local,
    Slide,
    UKIH,
    Willems,
)
from .methods.lh_core import lh_filter
from .param_estimate import param_calibrate, recession_coefficient
from .separation_core import compute_baseline_lh_for_df
from .utils import clean_streamflow, exist_ice, format_method, geo2imagexy

__all__ = ["single", "separation"]


def single(
    series: pd.Series,
    area: Optional[float] = None,
    ice: Optional[Union[npt.NDArray[np.bool_], Tuple[List[int], List[int]]]] = None,
    method: Union[str, List[str]] = "all",
    return_kge: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Perform baseflow separation for a single hydrological station.

    This function applies one or more baseflow separation algorithms to a streamflow
    time series. It automatically handles data cleaning, parameter estimation, and
    optional performance evaluation using the Kling-Gupta Efficiency (KGE) metric.

    Args:
        series: Time series of streamflow data with DatetimeIndex
        area: Drainage area in km² (required for HYSEP-based methods: Local, Fixed, Slide)
        ice: Frozen period specification. Can be:
            - Boolean array matching series length (True = frozen)
            - Tuple of [(start_month, start_day), (end_month, end_day)]
            - None to skip ice period masking
        method: Method name(s) to apply. Options:
            - "all": Apply all 12 available methods
            - Single method name: e.g., "LH", "Eckhardt"
            - List of method names: e.g., ["LH", "Chapman", "Eckhardt"]
        return_kge: Whether to calculate KGE scores against strict baseflow

    Returns:
        A tuple containing:
            - DataFrame with baseflow series for each method (index: dates, columns: methods)
            - Series of KGE scores for each method (if return_kge=True), otherwise None

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2010-01-01', periods=365, freq='D')
        >>> flow = pd.Series(np.random.lognormal(2, 1, 365), index=dates)
        >>> baseflow, kge_scores = single(flow, area=1000, method=["LH", "Eckhardt"])
        >>> print(kge_scores)

    Note:
        Methods requiring specific inputs:
        - Local, Fixed, Slide: Require 'area' parameter
        - Chapman, CM, Boughton, Furey, Eckhardt, Willems: Require recession coefficient
          (automatically estimated from data)
    """
    date, Q = clean_streamflow(series)
    method = format_method(method)

    # Convert ice_period specification to boolean array
    if not isinstance(ice, np.ndarray) or ice.shape[0] == 12:
        ice = exist_ice(date, ice)

    # Identify strict baseflow periods for evaluation
    strict = strict_baseflow(Q, ice)

    # Estimate recession coefficient if needed by any selected method
    if any(m in ["Chapman", "CM", "Boughton", "Furey", "Eckhardt", "Willems"] for m in method):
        a = recession_coefficient(Q, strict)

    # Compute baseline LH filter (used by many methods) using centralized implementation
    b_LH = compute_baseline_lh_for_df(Q)

    # Initialize results DataFrame
    b = pd.DataFrame(np.nan, index=date, columns=method)

    # Apply each selected method
    for m in method:
        if m == "UKIH":
            b[m] = UKIH(Q, b_LH)

        elif m == "Local":
            b[m] = Local(Q, b_LH, area)

        elif m == "Fixed":
            b[m] = Fixed(Q, area)

        elif m == "Slide":
            b[m] = Slide(Q, area)

        elif m == "LH":
            b[m] = b_LH

        elif m == "Chapman":
            b[m] = Chapman(Q, b_LH, a)

        elif m == "CM":
            b[m] = CM(Q, b_LH, a)

        elif m == "Boughton":
            param_range = get_param_range("Boughton")
            C = param_calibrate(param_range, Boughton, Q, b_LH, a)
            b[m] = Boughton(Q, b_LH, a, C)

        elif m == "Furey":
            param_range = get_param_range("Furey")
            A = param_calibrate(param_range, Furey, Q, b_LH, a)
            b[m] = Furey(Q, b_LH, a, A)

        elif m == "Eckhardt":
            param_range = get_param_range("Eckhardt")
            BFImax = param_calibrate(param_range, Eckhardt, Q, b_LH, a)
            b[m] = Eckhardt(Q, b_LH, a, BFImax)

        elif m == "EWMA":
            param_range = get_param_range("EWMA")
            e = param_calibrate(param_range, EWMA, Q, b_LH, 0)
            b[m] = EWMA(Q, b_LH, 0, e)

        elif m == "Willems":
            param_range = get_param_range("Willems")
            w = param_calibrate(param_range, Willems, Q, b_LH, a)
            b[m] = Willems(Q, b_LH, a, w)

    # Calculate performance metrics if requested
    if return_kge:
        KGEs = pd.Series(
            KGE(b[strict].values, np.repeat(Q[strict], len(method)).reshape(-1, len(method))),
            index=b.columns,
        )
        return b, KGEs
    else:
        return b, None


def separation(
    df: pd.DataFrame,
    df_sta: Optional[pd.DataFrame] = None,
    method: Union[str, List[str]] = "all",
    return_bfi: bool = False,
    return_kge: bool = False,
) -> Union[
    Dict[str, pd.DataFrame],
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame],
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame],
]:
    """Perform baseflow separation for multiple hydrological stations.

    This function processes streamflow data from multiple stations in parallel,
    applying selected baseflow separation methods to each station. It optionally
    uses station metadata (area, coordinates) to improve separation accuracy and
    compute performance metrics.

    Args:
        df: DataFrame with streamflow data (index: dates, columns: station IDs)
        df_sta: Optional DataFrame with station metadata (index: station IDs).
            Supported columns:
            - 'area': Drainage area in km² (for HYSEP methods)
            - 'lon', 'lat': Coordinates for frozen period detection
        method: Method name(s) to apply (see `single()` for options)
        return_bfi: Whether to calculate Baseflow Index (BFI) for each station
        return_kge: Whether to calculate KGE scores for each station

    Returns:
        Depending on flags, returns:
        - dfs: Dict mapping method names to baseflow DataFrames (same shape as input df)
        - df_bfi: DataFrame of BFI values (stations × methods) if return_bfi=True
        - df_kge: DataFrame of KGE scores (stations × methods) if return_kge=True

    Example:
        >>> # Create multi-station streamflow data
        >>> dates = pd.date_range('2010-01-01', periods=365)
        >>> stations = ['Station_A', 'Station_B', 'Station_C']
        >>> flow_data = pd.DataFrame(
        ...     np.random.lognormal(2, 1, (365, 3)),
        ...     index=dates,
        ...     columns=stations
        ... )
        >>>
        >>> # Station metadata
        >>> station_info = pd.DataFrame({
        ...     'area': [1000, 1500, 800],
        ...     'lon': [-120.5, -119.2, -121.0],
        ...     'lat': [45.2, 44.8, 46.1]
        ... }, index=stations)
        >>>
        >>> # Run separation
        >>> results, bfi, kge = separation(
        ...     flow_data,
        ...     df_sta=station_info,
        ...     method=["LH", "Eckhardt"],
        ...     return_bfi=True,
        ...     return_kge=True
        ... )
        >>> print(f"Methods applied: {list(results.keys())}")
        >>> print(f"BFI summary:\\n{bfi}")
        >>> print(f"KGE summary:\\n{kge}")

    Note:
        - Progress is displayed via tqdm progress bar
        - Stations that fail processing will print an error message and be skipped
        - Frozen period detection uses global permafrost data (included in package)
    """
    # Internal worker function for processing a single station
    def sep_work(s: str) -> None:
        try:
            # read area, longitude, latitude from df_sta
            area, ice = None, None
            to_num = lambda col: (
                pd.to_numeric(df_sta.loc[s, col], errors="coerce")
                if (df_sta is not None) and (col in df_sta.columns)
                else np.nan
            )
            if np.isfinite(to_num("area")):
                area = to_num("area")
            if np.isfinite(to_num("lon")):
                c, r = geo2imagexy(to_num("lon"), to_num("lat"))
                ice = ~thawed[:, r, c]
                ice = ([11, 1], [3, 31]) if ice.all() else ice
            # separate baseflow for station S
            b, KGEs = single(df[s], ice=ice, area=area, method=method, return_kge=return_kge)
            # write into already created dataframe
            for m in method:
                dfs[m].loc[b.index, s] = b[m]
            if return_bfi:
                df_bfi.loc[s] = b.sum() / df.loc[b.index, s].abs().sum()
            if return_kge:
                df_kge.loc[s] = KGEs
        except BaseException:
            print("\nFailed to separate baseflow for station {sta}".format(sta=s))
            pass

    # convert index to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # thawed months from https://doi.org/10.5194/essd-9-133-2017
    with np.load(Path(__file__).parent / "thawed.npz") as f:
        thawed = f["thawed"]

    # create df to store baseflow
    method = format_method(method)
    dfs = {m: pd.DataFrame(np.nan, index=df.index, columns=df.columns, dtype=float) for m in method}

    # create df to store BFI and KGE
    if return_bfi:
        df_bfi = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)
    if return_kge:
        df_kge = pd.DataFrame(np.nan, index=df.columns, columns=method, dtype=float)

    # run separation for each column
    for s in tqdm(df.columns, total=df.shape[1]):
        sep_work(s)

    # return result
    if return_bfi and return_kge:
        return dfs, df_bfi, df_kge
    if return_bfi and not return_kge:
        return dfs, df_bfi
    if not return_bfi and return_kge:
        return dfs, df_kge
    return dfs
