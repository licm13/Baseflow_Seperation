#!/usr/bin/env python
"""专业的基流分割命令行工具 (使用 Click 框架)。

本工具提供完整的命令行界面用于执行基流分割任务,包括:
- 单站点分割: 对单个流量时间序列进行分割
- 批处理模式: 对多个站点同时进行分割
- 参数配置: 自定义参数范围和方法选择
- 结果导出: 支持多种输出格式(CSV, Excel, NetCDF)

使用示例:
    # 单站点分割
    $ python baseflow_cli.py run-single data/flow.csv --area 1000 --methods LH Eckhardt

    # 批处理模式
    $ python baseflow_cli.py run-batch data/multi_station.csv data/stations.csv \\
        --output-dir results/ --return-bfi --return-kge

    # 列出所有可用方法
    $ python baseflow_cli.py list-methods

Author: Baseflow Separation Team
License: MIT
"""

import sys
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd

# 添加 src 到路径(用于开发模式)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import separation, single
from baseflow.config import ALL_METHODS, DEFAULT_PARAM_RANGES, update_param_range


# ============================================================================
# CLI 主程序和命令组
# ============================================================================

@click.group()
@click.version_option(version="1.0.0", prog_name="基流分割工具")
def cli():
    """基流分割命令行工具集。

    这个工具提供多种命令用于执行基流分割分析。
    使用 --help 查看每个子命令的详细说明。
    """
    pass


# ============================================================================
# 子命令1: 运行单站点分割
# ============================================================================

@cli.command("run-single")
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--column",
    "-c",
    default=None,
    help="要处理的列名(如果 CSV 有多列)。默认使用第一列。"
)
@click.option(
    "--area",
    "-a",
    type=float,
    default=None,
    help="流域面积(km²)。HYSEP 方法(Local, Fixed, Slide)需要此参数。"
)
@click.option(
    "--methods",
    "-m",
    multiple=True,
    default=["all"],
    help="要应用的方法名称。可多次指定。例如: -m LH -m Eckhardt。默认: all"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("outputs"),
    help="输出目录路径。默认: ./outputs"
)
@click.option(
    "--return-kge/--no-kge",
    default=True,
    help="是否计算 KGE 性能指标。默认: True"
)
@click.option(
    "--ice-period",
    nargs=4,
    type=int,
    default=None,
    metavar="START_MONTH START_DAY END_MONTH END_DAY",
    help="冻结期定义: 开始月 开始日 结束月 结束日。例如: --ice-period 11 1 3 31"
)
@click.option(
    "--date-column",
    default=None,
    help="日期列名称(如果日期不是索引)。"
)
@click.option(
    "--date-format",
    default=None,
    help="日期格式字符串(例如: %%Y-%%m-%%d)。默认自动推断。"
)
def run_single(
    input_file: Path,
    column: Optional[str],
    area: Optional[float],
    methods: tuple,
    output_dir: Path,
    return_kge: bool,
    ice_period: Optional[tuple],
    date_column: Optional[str],
    date_format: Optional[str],
):
    """运行单站点基流分割。

    读取单个流量时间序列文件,应用选定的基流分割方法,
    并将结果(基流时间序列和性能指标)保存到输出目录。

    INPUT_FILE: 输入 CSV 文件路径,包含流量时间序列。
                应该有日期列(或日期索引)和至少一列流量数据。

    示例:

        \b
        # 使用 LH 和 Eckhardt 方法分割单个站点
        $ python baseflow_cli.py run-single data/station_001.csv \\
            --area 1200 \\
            --methods LH --methods Eckhardt \\
            --output-dir results/station_001

        \b
        # 使用所有方法,指定冻结期
        $ python baseflow_cli.py run-single data/flow.csv \\
            --methods all \\
            --ice-period 11 1 3 31 \\
            --output-dir results/
    """
    click.echo(f"{'='*70}")
    click.echo(f"基流分割 - 单站点模式")
    click.echo(f"{'='*70}")

    # 步骤1: 读取输入数据
    click.echo(f"\n1. 读取输入文件: {input_file}")
    try:
        if date_column:
            df = pd.read_csv(input_file, parse_dates=[date_column])
            df = df.set_index(date_column)
        else:
            df = pd.read_csv(input_file, index_col=0, parse_dates=True)

        if date_format:
            df.index = pd.to_datetime(df.index, format=date_format)

        click.echo(f"   - 读取 {len(df)} 条记录")
        click.echo(f"   - 时间范围: {df.index[0]} 至 {df.index[-1]}")

    except Exception as e:
        click.echo(f"   错误: 无法读取文件: {e}", err=True)
        sys.exit(1)

    # 选择数据列
    if column:
        if column not in df.columns:
            click.echo(f"   错误: 列 '{column}' 不存在于文件中", err=True)
            click.echo(f"   可用列: {', '.join(df.columns)}", err=True)
            sys.exit(1)
        series = df[column]
    else:
        series = df.iloc[:, 0]

    click.echo(f"   - 使用列: {series.name}")
    click.echo(f"   - 流量范围: {series.min():.2f} - {series.max():.2f} m³/s")

    # 步骤2: 处理方法参数
    method_list = list(methods) if methods != ("all",) else "all"
    if method_list == "all":
        click.echo(f"\n2. 将应用所有 {len(ALL_METHODS)} 种方法")
    else:
        click.echo(f"\n2. 将应用 {len(method_list)} 种方法: {', '.join(method_list)}")

    # 步骤3: 处理冻结期参数
    ice = None
    if ice_period:
        start_m, start_d, end_m, end_d = ice_period
        ice = ([start_m, start_d], [end_m, end_d])
        click.echo(f"\n3. 冻结期: {start_m}/{start_d} - {end_m}/{end_d}")
    else:
        click.echo(f"\n3. 未指定冻结期(将跳过冻结期处理)")

    # 步骤4: 运行基流分割
    click.echo(f"\n4. 执行基流分割...")
    try:
        with click.progressbar(
            length=1,
            label="处理中",
            show_eta=True
        ) as bar:
            baseflow_df, kge_scores = single(
                series,
                area=area,
                ice=ice,
                method=method_list,
                return_kge=return_kge
            )
            bar.update(1)

        click.echo(f"   ✓ 分割完成")

    except Exception as e:
        click.echo(f"   ✗ 分割失败: {e}", err=True)
        sys.exit(1)

    # 步骤5: 显示结果摘要
    click.echo(f"\n5. 结果摘要:")
    for method in baseflow_df.columns:
        bfi = baseflow_df[method].sum() / series.sum()
        kge = kge_scores[method] if return_kge else None
        kge_str = f", KGE: {kge:.3f}" if kge is not None else ""
        click.echo(f"   - {method:12s}: BFI = {bfi:.3f}{kge_str}")

    # 步骤6: 保存结果
    click.echo(f"\n6. 保存结果到: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存基流时间序列
    baseflow_file = output_dir / "baseflow.csv"
    baseflow_df.to_csv(baseflow_file)
    click.echo(f"   - 基流时间序列: {baseflow_file}")

    # 保存 KGE 分数
    if return_kge:
        kge_file = output_dir / "kge_scores.csv"
        kge_scores.to_csv(kge_file, header=["KGE"])
        click.echo(f"   - KGE 分数: {kge_file}")

    # 保存 BFI
    bfi_series = baseflow_df.sum() / series.sum()
    bfi_file = output_dir / "bfi.csv"
    bfi_series.to_csv(bfi_file, header=["BFI"])
    click.echo(f"   - BFI 值: {bfi_file}")

    click.echo(f"\n{'='*70}")
    click.echo(f"✓ 单站点分割完成")
    click.echo(f"{'='*70}\n")


# ============================================================================
# 子命令2: 运行批处理分割
# ============================================================================

@cli.command("run-batch")
@click.argument(
    "flow_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "station_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--methods",
    "-m",
    multiple=True,
    default=["all"],
    help="要应用的方法名称。可多次指定。默认: all"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("outputs"),
    help="输出目录路径。默认: ./outputs"
)
@click.option(
    "--return-bfi/--no-bfi",
    default=True,
    help="是否计算 BFI。默认: True"
)
@click.option(
    "--return-kge/--no-kge",
    default=True,
    help="是否计算 KGE。默认: True"
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["csv", "excel", "netcdf"], case_sensitive=False),
    default="csv",
    help="输出文件格式。默认: csv"
)
def run_batch(
    flow_file: Path,
    station_file: Optional[Path],
    methods: tuple,
    output_dir: Path,
    return_bfi: bool,
    return_kge: bool,
    output_format: str,
):
    """运行批处理基流分割(多站点)。

    读取多站点流量数据和站点元数据,对所有站点应用基流分割,
    并生成汇总报告。

    FLOW_FILE: 流量数据 CSV 文件(索引=日期, 列=站点ID)

    STATION_FILE: 可选的站点元数据 CSV 文件(索引=站点ID)
                  应包含列: area, lon, lat

    示例:

        \b
        # 批处理模式,计算 BFI 和 KGE
        $ python baseflow_cli.py run-batch \\
            data/multi_station_flow.csv \\
            data/station_metadata.csv \\
            --methods LH --methods Eckhardt \\
            --output-dir results/batch/ \\
            --return-bfi --return-kge

        \b
        # 仅流量数据,无站点元数据
        $ python baseflow_cli.py run-batch \\
            data/flow_data.csv \\
            --methods all \\
            --output-format excel
    """
    click.echo(f"{'='*70}")
    click.echo(f"基流分割 - 批处理模式")
    click.echo(f"{'='*70}")

    # 步骤1: 读取流量数据
    click.echo(f"\n1. 读取流量数据: {flow_file}")
    try:
        df_flow = pd.read_csv(flow_file, index_col=0, parse_dates=True)
        click.echo(f"   - {df_flow.shape[0]} 天 × {df_flow.shape[1]} 个站点")
        click.echo(f"   - 时间范围: {df_flow.index[0]} 至 {df_flow.index[-1]}")
    except Exception as e:
        click.echo(f"   错误: {e}", err=True)
        sys.exit(1)

    # 步骤2: 读取站点元数据(可选)
    df_sta = None
    if station_file:
        click.echo(f"\n2. 读取站点元数据: {station_file}")
        try:
            df_sta = pd.read_csv(station_file, index_col=0)
            click.echo(f"   - {len(df_sta)} 个站点")
            if "area" in df_sta.columns:
                click.echo(f"   - 包含流域面积信息")
            if "lon" in df_sta.columns and "lat" in df_sta.columns:
                click.echo(f"   - 包含坐标信息(用于冻结期检测)")
        except Exception as e:
            click.echo(f"   警告: 无法读取站点文件: {e}", err=True)
            df_sta = None
    else:
        click.echo(f"\n2. 未提供站点元数据(将使用默认参数)")

    # 步骤3: 处理方法参数
    method_list = list(methods) if methods != ("all",) else "all"
    if method_list == "all":
        click.echo(f"\n3. 将应用所有 {len(ALL_METHODS)} 种方法")
    else:
        click.echo(f"\n3. 将应用 {len(method_list)} 种方法: {', '.join(method_list)}")

    # 步骤4: 运行批处理
    click.echo(f"\n4. 执行批处理分割...")
    click.echo(f"   (进度条将显示在下方)")

    try:
        result = separation(
            df_flow,
            df_sta=df_sta,
            method=method_list,
            return_bfi=return_bfi,
            return_kge=return_kge
        )

        # 解析返回值
        if return_bfi and return_kge:
            dfs, df_bfi, df_kge = result
        elif return_bfi:
            dfs, df_bfi = result
            df_kge = None
        elif return_kge:
            dfs, df_kge = result
            df_bfi = None
        else:
            dfs = result
            df_bfi, df_kge = None, None

        click.echo(f"\n   ✓ 批处理完成")

    except Exception as e:
        click.echo(f"\n   ✗ 批处理失败: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 步骤5: 保存结果
    click.echo(f"\n5. 保存结果到: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存每种方法的基流时间序列
    for method_name, baseflow_df in dfs.items():
        if output_format == "csv":
            file_path = output_dir / f"baseflow_{method_name}.csv"
            baseflow_df.to_csv(file_path)
        elif output_format == "excel":
            file_path = output_dir / f"baseflow_{method_name}.xlsx"
            baseflow_df.to_excel(file_path)
        elif output_format == "netcdf":
            file_path = output_dir / f"baseflow_{method_name}.nc"
            baseflow_df.to_xarray().to_netcdf(file_path)

        click.echo(f"   - {method_name}: {file_path}")

    # 保存 BFI 和 KGE 汇总
    if df_bfi is not None:
        bfi_file = output_dir / f"bfi_summary.{output_format if output_format != 'netcdf' else 'csv'}"
        if output_format == "excel":
            df_bfi.to_excel(bfi_file)
        else:
            df_bfi.to_csv(bfi_file)
        click.echo(f"   - BFI 汇总: {bfi_file}")

    if df_kge is not None:
        kge_file = output_dir / f"kge_summary.{output_format if output_format != 'netcdf' else 'csv'}"
        if output_format == "excel":
            df_kge.to_excel(kge_file)
        else:
            df_kge.to_csv(kge_file)
        click.echo(f"   - KGE 汇总: {kge_file}")

    # 生成摘要报告
    report_file = output_dir / "summary_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("基流分割批处理 - 摘要报告\n")
        f.write("="*70 + "\n\n")

        f.write(f"输入数据:\n")
        f.write(f"  - 流量文件: {flow_file}\n")
        f.write(f"  - 站点文件: {station_file or '无'}\n")
        f.write(f"  - 站点数量: {df_flow.shape[1]}\n")
        f.write(f"  - 时间跨度: {df_flow.shape[0]} 天\n\n")

        f.write(f"应用的方法:\n")
        for method in (method_list if method_list != "all" else ALL_METHODS):
            f.write(f"  - {method}\n")
        f.write("\n")

        if df_bfi is not None:
            f.write(f"BFI 统计摘要:\n")
            f.write(df_bfi.describe().to_string())
            f.write("\n\n")

        if df_kge is not None:
            f.write(f"KGE 统计摘要:\n")
            f.write(df_kge.describe().to_string())
            f.write("\n\n")

            f.write(f"最佳方法(按平均 KGE):\n")
            mean_kge = df_kge.mean().sort_values(ascending=False)
            for i, (method, kge) in enumerate(mean_kge.items(), 1):
                f.write(f"  {i}. {method}: {kge:.3f}\n")

    click.echo(f"   - 摘要报告: {report_file}")

    click.echo(f"\n{'='*70}")
    click.echo(f"✓ 批处理完成")
    click.echo(f"{'='*70}\n")


# ============================================================================
# 子命令3: 列出所有可用方法
# ============================================================================

@cli.command("list-methods")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="显示详细信息(参数范围、依赖关系)"
)
def list_methods(verbose: bool):
    """列出所有可用的基流分割方法。

    示例:

        \b
        $ python baseflow_cli.py list-methods

        \b
        # 显示详细信息
        $ python baseflow_cli.py list-methods --verbose
    """
    click.echo(f"\n可用的基流分割方法 (共 {len(ALL_METHODS)} 种):\n")
    click.echo(f"{'='*70}")

    for i, method in enumerate(ALL_METHODS, 1):
        config = DEFAULT_PARAM_RANGES.get(method)
        if config:
            click.echo(f"{i:2d}. {method:12s} - {config.description}")

            if verbose:
                click.echo(f"    依赖:")
                click.echo(f"      流域面积: {'是' if config.requires_area else '否'}")
                click.echo(f"      衰退系数: {'是' if config.requires_recession_coef else '否'}")
                click.echo(f"      参数标定: {'是' if config.param_range is not None else '否'}")

                if config.param_range is not None:
                    click.echo(f"    参数范围:")
                    click.echo(f"      长度: {len(config.param_range)}")
                    click.echo(f"      范围: [{config.param_range.min():.4f}, {config.param_range.max():.4f}]")
                click.echo()

    click.echo(f"{'='*70}\n")


# ============================================================================
# 子命令4: 配置参数范围
# ============================================================================

@cli.command("config-param")
@click.argument("method", type=str)
@click.argument("start", type=float)
@click.argument("stop", type=float)
@click.argument("step", type=float)
def config_param(method: str, start: float, stop: float, step: float):
    """配置方法的参数搜索范围。

    METHOD: 方法名称(例如: Eckhardt)

    START: 参数范围起始值

    STOP: 参数范围结束值(不包含)

    STEP: 参数步长

    示例:

        \b
        # 为 Eckhardt 设置粗粒度参数范围(加速标定)
        $ python baseflow_cli.py config-param Eckhardt 0.01 0.99 0.01

        \b
        # 为 Boughton 设置精细参数范围
        $ python baseflow_cli.py config-param Boughton 0.0001 0.1 0.00001
    """
    click.echo(f"\n配置参数范围:")
    click.echo(f"  方法: {method}")
    click.echo(f"  范围: [{start}, {stop})")
    click.echo(f"  步长: {step}")

    try:
        update_param_range(method, start, stop, step)
        n_values = len(np.arange(start, stop, step))
        click.echo(f"\n✓ 更新成功")
        click.echo(f"  参数点数: {n_values}")
        click.echo(f"\n注意: 这个配置仅在当前 Python 会话中有效。")
        click.echo(f"      如需永久修改,请编辑 src/baseflow/config.py\n")

    except Exception as e:
        click.echo(f"\n✗ 更新失败: {e}\n", err=True)
        sys.exit(1)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    cli()
