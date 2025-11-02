"""真实世界基流分割工作流程示例。

本脚本展示一个完整的科研/业务工作流程:
1. 从外部配置文件读取参数
2. 加载真实(或示例)流量数据和站点信息
3. 执行批量基流分割
4. 生成可视化图表
5. 导出结果和生成报告

这个示例模拟了实际生产环境中的使用场景,适合作为
自己项目的起点模板。

Author: Baseflow Separation Team
Date: 2025
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml

# 配置matplotlib中文字体
def configure_chinese_font():
    """配置matplotlib中文字体显示"""
    # 尝试多种中文字体，按优先级排序
    chinese_fonts = [
        'SimHei',           # Windows 黑体
        'Microsoft YaHei',  # Windows 微软雅黑
        'SimSun',           # Windows 宋体
        'PingFang SC',      # macOS 苹方
        'Hiragino Sans GB', # macOS 冬青黑体
        'WenQuanYi Micro Hei', # Linux 文泉驿微米黑
        'Noto Sans CJK SC', # Linux Noto字体
        'DejaVu Sans',      # 通用fallback
        'Arial Unicode MS', # 通用fallback
        'sans-serif'        # 最终fallback
    ]
    
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置默认字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 10

# 配置中文字体
configure_chinese_font()

# 添加 src 到路径(用于开发模式)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import separation
from baseflow.config import update_param_range

# 忽略警告以保持输出清洁
warnings.filterwarnings("ignore")


# ============================================================================
# 步骤1: 配置管理 - 从 YAML 文件读取配置
# ============================================================================

def load_config(config_file: Path) -> dict:
    """从 YAML 配置文件加载参数。

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    print(f"{'='*70}")
    print(f"步骤1: 加载配置文件")
    print(f"{'='*70}")

    if not config_file.exists():
        print(f"\n配置文件不存在: {config_file}")
        print(f"将使用默认配置...")
        return get_default_config()

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        print(f"\n✓ 成功加载配置: {config_file}")
        print(f"\n配置内容:")
        print(yaml.dump(config, default_flow_style=False, allow_unicode=True))

        return config

    except Exception as e:
        print(f"\n✗ 配置文件加载失败: {e}")
        print(f"将使用默认配置...")
        return get_default_config()


def get_default_config() -> dict:
    """获取默认配置。

    Returns:
        默认配置字典
    """
    return {
        "data": {
            "flow_file": "data/example.csv",  # 流量数据文件
            "station_file": None,  # 站点信息文件(可选)
            "date_column": None,  # 日期列名(如果不是索引)
        },
        "methods": {
            "selected": ["LH", "Eckhardt", "Chapman", "UKIH"],  # 要使用的方法
            "parameter_ranges": {  # 自定义参数范围(可选)
                "Eckhardt": {"start": 0.01, "stop": 0.99, "step": 0.01},
            },
        },
        "output": {
            "directory": "outputs/real_world_workflow",  # 输出目录
            "save_timeseries": True,  # 保存基流时间序列
            "save_metrics": True,  # 保存性能指标
            "generate_report": True,  # 生成摘要报告
            "generate_plots": True,  # 生成可视化图表
        },
        "processing": {
            "return_bfi": True,  # 计算 BFI
            "return_kge": True,  # 计算 KGE
        },
    }


# ============================================================================
# 步骤2: 数据加载
# ============================================================================

def load_data(config: dict) -> tuple:
    """加载流量数据和站点信息。

    Args:
        config: 配置字典

    Returns:
        (流量DataFrame, 站点信息DataFrame)
    """
    print(f"\n{'='*70}")
    print(f"步骤2: 加载数据")
    print(f"{'='*70}")

    # 2.1 加载流量数据
    # ----------------------
    flow_file = Path(config["data"]["flow_file"])
    print(f"\n2.1 加载流量数据: {flow_file}")

    if not flow_file.exists():
        print(f"\n流量文件不存在: {flow_file}")
        print(f"将生成合成数据作为演示...")
        return generate_demo_data()

    try:
        date_col = config["data"].get("date_column")
        if date_col:
            df_flow = pd.read_csv(flow_file, parse_dates=[date_col])
            df_flow = df_flow.set_index(date_col)
        else:
            df_flow = pd.read_csv(flow_file, index_col=0, parse_dates=True)

        print(f"   ✓ 成功加载")
        print(f"   - 形状: {df_flow.shape[0]} 天 × {df_flow.shape[1]} 个站点")
        print(f"   - 时间范围: {df_flow.index[0]} 至 {df_flow.index[-1]}")
        print(f"   - 站点ID: {', '.join(df_flow.columns[:5].tolist())}" +
              (f" (及其他 {df_flow.shape[1]-5} 个)" if df_flow.shape[1] > 5 else ""))

    except Exception as e:
        print(f"   ✗ 加载失败: {e}")
        print(f"   将生成合成数据作为演示...")
        return generate_demo_data()

    # 2.2 加载站点信息(可选)
    # ----------------------
    station_file = config["data"].get("station_file")
    df_sta = None

    if station_file:
        station_file = Path(station_file)
        print(f"\n2.2 加载站点信息: {station_file}")

        if station_file.exists():
            try:
                df_sta = pd.read_csv(station_file, index_col=0)
                print(f"   ✓ 成功加载")
                print(f"   - 站点数: {len(df_sta)}")
                print(f"   - 可用字段: {', '.join(df_sta.columns)}")

                # 检查关键字段
                if "area" in df_sta.columns:
                    print(f"   - 包含流域面积(用于 HYSEP 方法)")
                if "lon" in df_sta.columns and "lat" in df_sta.columns:
                    print(f"   - 包含坐标(用于冻结期检测)")

            except Exception as e:
                print(f"   ✗ 加载失败: {e}")
                df_sta = None
        else:
            print(f"   站点文件不存在: {station_file}")
    else:
        print(f"\n2.2 未提供站点信息文件(将使用默认参数)")

    return df_flow, df_sta


def generate_demo_data() -> tuple:
    """生成演示用的合成数据。

    Returns:
        (流量DataFrame, 站点信息DataFrame)
    """
    print(f"\n正在生成演示数据...")

    from baseflow.synthetic_data import create_test_dataframe

    df_flow, _, df_sta = create_test_dataframe(
        n_days=730,  # 2年数据
        n_stations=3,
        start_date="2020-01-01",
        random_seed=42
    )

    print(f"   ✓ 演示数据生成完成")
    print(f"   - {df_flow.shape[0]} 天 × {df_flow.shape[1]} 个站点")

    return df_flow, df_sta


# ============================================================================
# 步骤3: 参数配置
# ============================================================================

def configure_parameters(config: dict):
    """配置方法参数范围。

    Args:
        config: 配置字典
    """
    print(f"\n{'='*70}")
    print(f"步骤3: 配置参数范围")
    print(f"{'='*70}\n")

    param_ranges = config["methods"].get("parameter_ranges", {})

    if not param_ranges:
        print("使用默认参数范围(未自定义)")
        return

    for method, params in param_ranges.items():
        try:
            update_param_range(
                method,
                params["start"],
                params["stop"],
                params["step"]
            )
            n_values = int((params["stop"] - params["start"]) / params["step"])
            print(f"✓ {method}: [{params['start']}, {params['stop']}), "
                  f"步长={params['step']}, 共 {n_values} 个参数点")

        except Exception as e:
            print(f"✗ {method}: 配置失败 - {e}")


# ============================================================================
# 步骤4: 执行基流分割
# ============================================================================

def run_separation(df_flow: pd.DataFrame, df_sta: pd.DataFrame, config: dict) -> tuple:
    """执行批量基流分割。

    Args:
        df_flow: 流量数据
        df_sta: 站点信息
        config: 配置字典

    Returns:
        (基流字典, BFI DataFrame, KGE DataFrame)
    """
    print(f"\n{'='*70}")
    print(f"步骤4: 执行基流分割")
    print(f"{'='*70}\n")

    methods = config["methods"]["selected"]
    return_bfi = config["processing"]["return_bfi"]
    return_kge = config["processing"]["return_kge"]

    print(f"应用的方法: {', '.join(methods)}")
    print(f"计算 BFI: {'是' if return_bfi else '否'}")
    print(f"计算 KGE: {'是' if return_kge else '否'}")
    print(f"\n开始处理...(进度条显示在下方)\n")

    try:
        result = separation(
            df_flow,
            df_sta=df_sta,
            method=methods,
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

        print(f"\n✓ 基流分割完成")

        return dfs, df_bfi, df_kge

    except Exception as e:
        print(f"\n✗ 基流分割失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# 步骤5: 生成可视化
# ============================================================================

def generate_plots(dfs: dict, df_flow: pd.DataFrame, df_bfi: pd.DataFrame,
                   df_kge: pd.DataFrame, output_dir: Path):
    """生成可视化图表。

    Args:
        dfs: 基流字典
        df_flow: 原始流量数据
        df_bfi: BFI DataFrame
        df_kge: KGE DataFrame
        output_dir: 输出目录
    """
    print(f"\n{'='*70}")
    print(f"步骤5: 生成可视化图表")
    print(f"{'='*70}\n")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 5.1 时间序列图(选择第一个站点)
    # ----------------------
    print("5.1 生成时间序列图...")
    station = df_flow.columns[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_flow.index, df_flow[station], 'k-', label='总流量',
            alpha=0.6, linewidth=1)

    for method, baseflow_df in dfs.items():
        ax.plot(baseflow_df.index, baseflow_df[station],
                label=f'基流 ({method})', alpha=0.8, linewidth=1.5)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('流量 (m³/s)', fontsize=12)
    ax.set_title(f'基流分割结果 - {station}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plot_file = plots_dir / "timeseries.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ {plot_file}")

    # 5.2 BFI 对比图
    # ----------------------
    if df_bfi is not None:
        print("5.2 生成 BFI 对比图...")

        fig, ax = plt.subplots(figsize=(12, 6))
        df_bfi.plot(kind='bar', ax=ax, rot=45)

        ax.set_xlabel('站点', fontsize=12)
        ax.set_ylabel('基流指数 (BFI)', fontsize=12)
        ax.set_title('不同方法的 BFI 对比', fontsize=14, fontweight='bold')
        ax.legend(title='方法', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plot_file = plots_dir / "bfi_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {plot_file}")

    # 5.3 KGE 性能图
    # ----------------------
    if df_kge is not None:
        print("5.3 生成 KGE 性能图...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 热力图
        im = ax1.imshow(df_kge.T.values, cmap='RdYlGn', aspect='auto',
                        vmin=-0.4, vmax=1.0)
        ax1.set_xticks(range(len(df_kge.index)))
        ax1.set_xticklabels(df_kge.index, rotation=45, ha='right')
        ax1.set_yticks(range(len(df_kge.columns)))
        ax1.set_yticklabels(df_kge.columns)
        ax1.set_xlabel('站点', fontsize=12)
        ax1.set_ylabel('方法', fontsize=12)
        ax1.set_title('KGE 热力图', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='KGE')

        # 箱线图
        df_kge.boxplot(ax=ax2, rot=45)
        ax2.axhline(y=0.75, color='g', linestyle='--', alpha=0.5, label='优秀')
        ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='良好')
        ax2.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='差')
        ax2.set_xlabel('方法', fontsize=12)
        ax2.set_ylabel('KGE', fontsize=12)
        ax2.set_title('KGE 分布', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3, axis='y')

        plot_file = plots_dir / "kge_performance.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ {plot_file}")


# ============================================================================
# 步骤6: 导出结果
# ============================================================================

def export_results(dfs: dict, df_bfi: pd.DataFrame, df_kge: pd.DataFrame,
                   output_dir: Path, config: dict):
    """导出基流时间序列和性能指标。

    Args:
        dfs: 基流字典
        df_bfi: BFI DataFrame
        df_kge: KGE DataFrame
        output_dir: 输出目录
        config: 配置字典
    """
    print(f"\n{'='*70}")
    print(f"步骤6: 导出结果")
    print(f"{'='*70}\n")

    # 6.1 导出基流时间序列
    # ----------------------
    if config["output"]["save_timeseries"]:
        print("6.1 导出基流时间序列...")
        ts_dir = output_dir / "baseflow_timeseries"
        ts_dir.mkdir(parents=True, exist_ok=True)

        for method, baseflow_df in dfs.items():
            file_path = ts_dir / f"baseflow_{method}.csv"
            baseflow_df.to_csv(file_path)
            print(f"   ✓ {file_path}")

    # 6.2 导出性能指标
    # ----------------------
    if config["output"]["save_metrics"]:
        print("\n6.2 导出性能指标...")
        metrics_dir = output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if df_bfi is not None:
            bfi_file = metrics_dir / "bfi.csv"
            df_bfi.to_csv(bfi_file)
            print(f"   ✓ BFI: {bfi_file}")

        if df_kge is not None:
            kge_file = metrics_dir / "kge.csv"
            df_kge.to_csv(kge_file)
            print(f"   ✓ KGE: {kge_file}")


# ============================================================================
# 步骤7: 生成报告
# ============================================================================

def generate_report(dfs: dict, df_flow: pd.DataFrame, df_bfi: pd.DataFrame,
                    df_kge: pd.DataFrame, output_dir: Path, config: dict):
    """生成 Markdown 格式的摘要报告。

    Args:
        dfs: 基流字典
        df_flow: 原始流量数据
        df_bfi: BFI DataFrame
        df_kge: KGE DataFrame
        output_dir: 输出目录
        config: 配置字典
    """
    print(f"\n{'='*70}")
    print(f"步骤7: 生成摘要报告")
    print(f"{'='*70}\n")

    report_file = output_dir / "REPORT.md"

    with open(report_file, "w", encoding="utf-8") as f:
        # 标题
        f.write("# 基流分割分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # 1. 数据概况
        f.write("## 1. 数据概况\n\n")
        f.write(f"- **站点数量**: {df_flow.shape[1]}\n")
        f.write(f"- **时间跨度**: {df_flow.shape[0]} 天 ({df_flow.index[0].date()} 至 {df_flow.index[-1].date()})\n")
        f.write(f"- **数据完整性**: {(~df_flow.isna()).sum().sum() / df_flow.size * 100:.1f}%\n\n")

        # 2. 应用的方法
        f.write("## 2. 应用的方法\n\n")
        methods = config["methods"]["selected"]
        for i, method in enumerate(methods, 1):
            f.write(f"{i}. **{method}**\n")
        f.write("\n")

        # 3. BFI 摘要
        if df_bfi is not None:
            f.write("## 3. 基流指数 (BFI) 摘要\n\n")
            f.write("### 3.1 统计摘要\n\n")
            f.write("```\n")
            f.write(df_bfi.describe().to_string())
            f.write("\n```\n\n")

            f.write("### 3.2 各站点 BFI\n\n")
            f.write(df_bfi.to_markdown())
            f.write("\n\n")

            # 最佳方法(按 BFI 稳定性)
            bfi_std = df_bfi.std()
            f.write(f"### 3.3 方法稳定性(标准差越小越稳定)\n\n")
            for method, std in bfi_std.sort_values().items():
                f.write(f"- **{method}**: {std:.4f}\n")
            f.write("\n")

        # 4. KGE 摘要
        if df_kge is not None:
            f.write("## 4. KGE 性能摘要\n\n")
            f.write("### 4.1 统计摘要\n\n")
            f.write("```\n")
            f.write(df_kge.describe().to_string())
            f.write("\n```\n\n")

            f.write("### 4.2 各站点 KGE\n\n")
            f.write(df_kge.to_markdown())
            f.write("\n\n")

            # 方法排名
            mean_kge = df_kge.mean().sort_values(ascending=False)
            f.write("### 4.3 方法排名(按平均 KGE)\n\n")
            for i, (method, kge) in enumerate(mean_kge.items(), 1):
                performance = "🟢 优秀" if kge > 0.75 else "🟡 良好" if kge > 0.5 else "🔴 一般"
                f.write(f"{i}. **{method}**: {kge:.3f} {performance}\n")
            f.write("\n")

        # 5. 建议
        f.write("## 5. 建议\n\n")
        if df_kge is not None:
            best_method = df_kge.mean().idxmax()
            best_kge = df_kge.mean().max()
            f.write(f"- **推荐方法**: {best_method} (平均 KGE = {best_kge:.3f})\n")

        f.write("- 基于您的流域特征,可以进一步调整参数范围以优化性能\n")
        f.write("- 建议对比多种方法的结果,选择最符合实际水文过程的方法\n")
        f.write("- 可以使用 `examples/interactive_visualization.py` 进行交互式探索\n\n")

        # 6. 参考资料
        f.write("## 6. 相关文件\n\n")
        f.write("- 基流时间序列: `baseflow_timeseries/`\n")
        f.write("- 性能指标: `metrics/`\n")
        f.write("- 可视化图表: `plots/`\n")
        f.write("- 配置文件: 参见项目根目录的 `config.yml`\n\n")

        f.write("---\n\n")
        f.write("*本报告由基流分割工具自动生成*\n")

    print(f"✓ 报告已生成: {report_file}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主工作流程。"""
    print("\n")
    print("█" * 70)
    print(" " * 15 + "真实世界基流分割工作流程")
    print("█" * 70)
    print("\n")

    # 确定配置文件路径
    project_root = Path(__file__).parent.parent
    config_file = project_root / "config.yml"

    # 执行工作流程
    config = load_config(config_file)
    df_flow, df_sta = load_data(config)
    configure_parameters(config)
    dfs, df_bfi, df_kge = run_separation(df_flow, df_sta, config)

    # 输出目录
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成可视化
    if config["output"]["generate_plots"]:
        generate_plots(dfs, df_flow, df_bfi, df_kge, output_dir)

    # 导出结果
    export_results(dfs, df_bfi, df_kge, output_dir, config)

    # 生成报告
    if config["output"]["generate_report"]:
        generate_report(dfs, df_flow, df_bfi, df_kge, output_dir, config)

    # 完成
    print(f"\n{'='*70}")
    print(f"✓ 工作流程完成")
    print(f"{'='*70}")
    print(f"\n所有结果已保存到: {output_dir.absolute()}")
    print(f"\n主要输出:")
    print(f"  - 摘要报告: {output_dir / 'REPORT.md'}")
    print(f"  - 基流时间序列: {output_dir / 'baseflow_timeseries/'}")
    print(f"  - 性能指标: {output_dir / 'metrics/'}")
    print(f"  - 可视化图表: {output_dir / 'plots/'}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
