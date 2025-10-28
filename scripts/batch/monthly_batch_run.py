"""Monthly-resolution batch processor for legacy 60-16 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from tqdm import tqdm

from .common import (
    BatchConfig,
    DEFAULT_METHODS,
    iter_input_files,
    load_station_info,
    run_single_site,
)

DEFAULT_INPUT_DIR = Path(r"Z:/Runoff_Flood/China_runoff/daily_Q_60_16")
DEFAULT_OUTPUT_DIR = Path(r"Z:/LCM/Baseflow_seperation/daily60_16_outputs")
DEFAULT_INFO_FILE = Path(r"Z:/Runoff_Flood/China_runoff/info2.xlsx")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process 1960-2016 station records.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--info-file", type=Path, default=DEFAULT_INFO_FILE)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="需要执行的基流分割方法列表。",
    )
    parser.add_argument(
        "--default-area",
        type=float,
        default=150.0,
        help="当站点缺少面积信息时使用的默认面积 (km²)。",
    )
    return parser.parse_args(argv)


def run_batch(config: BatchConfig) -> None:
    info_df = load_station_info(config.info_file)
    missing_area: List[str] = []
    kge_records: List[dict[str, float | str]] = []

    files = list(iter_input_files(config))
    for idx, filepath in enumerate(tqdm(files, desc="📦 处理60-16站点", unit="站点")):
        site_id = filepath.stem
        try:
            record = run_single_site(filepath, info_df, config, missing_area)
            kge_records.append(record)
            print(
                f"✅ {site_id} 完成 ({idx + 1}/{len(files)}) -> {config.output_dir / f'baseflow_separation_{site_id}.csv'}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"❌ 站点 {site_id} 处理失败：{exc}")

    if kge_records:
        pd.DataFrame(kge_records).to_csv(
            config.output_dir / "baseflow_kge_all_sites.csv", index=False
        )
    if missing_area:
        pd.DataFrame(missing_area, columns=["site_id"]).to_csv(
            config.output_dir / "missing_area_sites.csv", index=False
        )
        print(f"⚠️ 共 {len(missing_area)} 个站点缺失面积信息。")
    else:
        print("✅ 所有站点均匹配到面积信息。")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = BatchConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        info_file=args.info_file,
        methods=args.methods,
        default_area=args.default_area,
        filename_suffix=".csv",
    )
    config.ensure_paths()
    run_batch(config)


if __name__ == "__main__":  # pragma: no cover
    main()
