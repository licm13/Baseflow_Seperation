"""Shared utilities for batch-processing scripts.

该模块集中管理批处理脚本的公共配置、面积查询逻辑和结果写入操作，
避免三个批处理脚本间的重复代码，确保参数解释一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from baseflow.separation import single

DEFAULT_METHODS: Tuple[str, ...] = (
    "LH",
    "Chapman",
    "CM",
    "Boughton",
    "Furey",
    "Eckhardt",
    "EWMA",
    "Willems",
    "UKIH",
    "Fixed",
    "Local",
    "Slide",
)


@dataclass(slots=True)
class BatchConfig:
    """用户可自定义的批处理配置。"""

    input_dir: Path
    output_dir: Path
    info_file: Path
    methods: Sequence[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    default_area: float = 150.0
    id_columns: Tuple[str, str] = ("id8", "id5")
    filename_suffix: str = ".csv"

    def ensure_paths(self) -> None:
        self.input_dir = self.input_dir.expanduser()
        self.output_dir = self.output_dir.expanduser()
        self.info_file = self.info_file.expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)


def load_station_info(info_path: Path) -> pd.DataFrame:
    """Load the station attribute table and normalise ID columns to strings."""

    info_df = pd.read_excel(info_path)
    for col in ["id8", "id5"]:
        if col in info_df.columns:
            info_df[col] = info_df[col].astype(str).str.replace(".0", "", regex=False)
    return info_df


def lookup_area(site_id: str, info_df: pd.DataFrame, config: BatchConfig) -> float:
    """Return the basin area associated with *site_id*.

    若匹配失败，返回配置中的默认面积并记录日志使用者需自行处理。
    """

    matches: List[pd.Series] = []
    for column in config.id_columns:
        if column in info_df.columns:
            subset = info_df[info_df[column] == site_id]
            if not subset.empty:
                matches.append(subset.iloc[0])
    if not matches:
        raise KeyError(site_id)

    row = matches[0]
    area = row.get("area", np.nan)
    area2 = row.get("area2", np.nan)
    if np.isnan(area) and np.isnan(area2):
        raise ValueError(site_id)
    if np.isnan(area):
        return float(area2)
    if np.isnan(area2):
        return float(area)
    return float((area + area2) / 2)


def iter_input_files(config: BatchConfig) -> Iterable[Path]:
    """Yield input data files whose stem is numeric (站点编号)."""

    suffix = config.filename_suffix
    for path in sorted(config.input_dir.glob(f"*{suffix}")):
        if path.stem.isdigit():
            yield path


def run_single_site(
    filepath: Path,
    info_df: pd.DataFrame,
    config: BatchConfig,
    missing_area: List[str],
) -> Dict[str, float | str]:
    """Execute the baseflow separation pipeline for a single site."""

    site_id = filepath.stem
    df = read_timeseries(filepath)

    try:
        area = lookup_area(site_id, info_df, config)
    except (KeyError, ValueError):
        area = config.default_area
        missing_area.append(site_id)

    if area <= 0:
        raise RuntimeError(f"非法面积 {area} km²，站点 {site_id}")

    df["R"] = df["Q"] * 86400 / area
    series = pd.Series(df["R"].values, index=df["date"])
    output_df = pd.DataFrame({"time": df["date"], "Q": df["Q"], "R": df["R"]})
    kge_record: Dict[str, float | str] = {"site": site_id}

    for method in tqdm(config.methods, desc=f"    ➤ {site_id} 方法计算中", leave=False, unit="方法"):
        try:
            b, kge = single(series, area=area, method=[method], return_kge=True)
            output_df[method] = b[method].values
            kge_record[method] = float(kge[method])
        except Exception:  # noqa: BLE001 - 保留现场供人工排查
            output_df[method] = np.nan
            kge_record[method] = np.nan

    output_csv = config.output_dir / f"baseflow_separation_{site_id}.csv"
    output_df.to_csv(output_csv, index=False)
    return kge_record


def read_timeseries(filepath: Path) -> pd.DataFrame:
    """Read daily discharge file and standardise column names."""

    if filepath.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    if {"year", "month", "day"}.issubset(df.columns):
        df = df.rename(columns={"Q": "Q"})
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    else:
        df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "Q"]].copy()
    return df
