"""Minimal example demonstrating the high-level API of the baseflow package."""

from __future__ import annotations

from pathlib import Path
import sys

# Prefer repo/src over site-packages
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir if (src_dir / "baseflow").exists() else repo_root))

import pandas as pd
import baseflow

OUTPUT_DIR = Path("./data/example_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"使用示例数据: {baseflow.example}")
    df = pd.read_csv(baseflow.example, index_col=0, parse_dates=True)

    # 构造站点信息表：示例数据中包含两个站点
    station_info = pd.DataFrame(
        data=[[30, -28.4, 659], [-109.4, 33, 1611]],
        index=df.columns,
        columns=["lon", "lat", "area"],
    )

    # 调用高层 API，同时获取基流序列与 KGE 指标
    dfs, df_kge = baseflow.separation(df, station_info, return_kge=True)

    for method, result in dfs.items():
        output_path = OUTPUT_DIR / f"example_{method}.csv"
        result.to_csv(output_path)
        print(f"✅ {method:<10} -> {output_path}")

    print("\n最佳方法 (KGE 最高)：")
    print(df_kge.idxmax(axis=1))


if __name__ == "__main__":  # pragma: no cover
    main()
