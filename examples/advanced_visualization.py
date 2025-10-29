"""Advanced visualization and analysis for baseflow separation.

This script provides comprehensive visualization tools including:
- Time series plots with multiple methods
- Statistical distribution analysis
- Error analysis and residuals
- Method comparison heatmaps
- Performance metrics visualization

Usage:
    python advanced_visualization.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseflow import single
from baseflow.synthetic_data import generate_streamflow


def create_comprehensive_plots(flow_series, baseflow_df, true_baseflow, kge_scores, output_dir=None):
    """Create comprehensive visualization of baseflow separation results.

    Args:
        flow_series: Original streamflow series
        baseflow_df: DataFrame with separated baseflow from multiple methods
        true_baseflow: True baseflow (for synthetic data)
        kge_scores: KGE performance scores
        output_dir: Directory to save plots (default: current directory)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    Q = flow_series.values
    dates = flow_series.index
    methods = baseflow_df.columns.tolist()

    # Create figure with complex layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ========== Plot 1: Full time series ==========
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, Q, 'k-', label='Streamflow', alpha=0.6, linewidth=1)
    ax1.plot(dates, true_baseflow, 'b-', label='True Baseflow', linewidth=2, alpha=0.8)

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, method in enumerate(methods[:5]):  # Plot top 5 methods
        ax1.plot(dates, baseflow_df[method], label=method,
                linewidth=1.5, alpha=0.7, color=colors[i])

    ax1.set_ylabel('Flow (m³/s)', fontsize=11, fontweight='bold')
    ax1.set_title('Baseflow Separation - Complete Time Series', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ========== Plot 2: Zoomed recession period ==========
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_start, zoom_end = len(Q)//3, len(Q)//3 + 60
    zoom_dates = dates[zoom_start:zoom_end]

    ax2.plot(zoom_dates, Q[zoom_start:zoom_end], 'k-', label='Streamflow', linewidth=2, alpha=0.7)
    ax2.plot(zoom_dates, true_baseflow[zoom_start:zoom_end], 'b--',
             label='True', linewidth=2, alpha=0.8)

    for i, method in enumerate(methods[:3]):
        ax2.plot(zoom_dates, baseflow_df[method].values[zoom_start:zoom_end],
                label=method, linewidth=1.5, alpha=0.7, color=colors[i])

    ax2.set_ylabel('Flow (m³/s)', fontsize=10, fontweight='bold')
    ax2.set_title('Recession Period Detail', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ========== Plot 3: Flow duration curves ==========
    ax3 = fig.add_subplot(gs[1, 1])
    exceedance = np.arange(1, len(Q) + 1) / len(Q) * 100

    ax3.semilogy(exceedance, np.sort(Q)[::-1], 'k-', label='Streamflow', linewidth=2)
    ax3.semilogy(exceedance, np.sort(true_baseflow)[::-1], 'b--',
                 label='True BF', linewidth=2)

    for i, method in enumerate(methods[:3]):
        bf_sorted = np.sort(baseflow_df[method].values)[::-1]
        ax3.semilogy(exceedance, bf_sorted, label=method,
                    linewidth=1.5, alpha=0.7, color=colors[i])

    ax3.set_xlabel('Exceedance Probability (%)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Flow (m³/s)', fontsize=10, fontweight='bold')
    ax3.set_title('Flow Duration Curves', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, which='both')

    # ========== Plot 4: KGE Scores ==========
    ax4 = fig.add_subplot(gs[1, 2])
    kge_values = [kge_scores[m] for m in methods]
    colors_bar = ['green' if k > 0.8 else 'orange' if k > 0.6 else 'red' for k in kge_values]

    bars = ax4.barh(methods, kge_values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax4.axvline(0.8, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent')
    ax4.axvline(0.6, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good')
    ax4.set_xlabel('KGE Score', fontsize=10, fontweight='bold')
    ax4.set_title('Method Performance (KGE)', fontsize=11, fontweight='bold')
    ax4.set_xlim(-0.5, 1.0)
    ax4.legend(fontsize=8, loc='lower right')
    ax4.grid(True, alpha=0.3, axis='x')

    # ========== Plot 5: BFI Comparison ==========
    ax5 = fig.add_subplot(gs[2, 0])
    true_bfi = true_baseflow.sum() / Q.sum()
    estimated_bfis = [baseflow_df[m].sum() / Q.sum() for m in methods]
    bfi_errors = [abs(bfi - true_bfi) for bfi in estimated_bfis]

    x_pos = np.arange(len(methods))
    ax5.bar(x_pos, estimated_bfis, alpha=0.7, label='Estimated', edgecolor='black')
    ax5.axhline(true_bfi, color='red', linestyle='--', linewidth=2, label='True BFI')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Baseflow Index', fontsize=10, fontweight='bold')
    ax5.set_title('BFI Comparison', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

    # ========== Plot 6: Error distribution ==========
    ax6 = fig.add_subplot(gs[2, 1])
    errors_data = []
    for method in methods[:5]:
        errors = baseflow_df[method].values - true_baseflow
        errors_data.append(errors)

    bp = ax6.boxplot(errors_data, labels=methods[:5], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax6.axhline(0, color='red', linestyle='--', linewidth=1)
    ax6.set_xticklabels(methods[:5], rotation=45, ha='right', fontsize=9)
    ax6.set_ylabel('Error (m³/s)', fontsize=10, fontweight='bold')
    ax6.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # ========== Plot 7: Cumulative baseflow ==========
    ax7 = fig.add_subplot(gs[2, 2])
    cumulative_true = np.cumsum(true_baseflow)
    ax7.plot(dates, cumulative_true, 'b-', label='True', linewidth=2)

    for i, method in enumerate(methods[:4]):
        cumulative = np.cumsum(baseflow_df[method].values)
        ax7.plot(dates, cumulative, label=method, linewidth=1.5,
                alpha=0.7, color=colors[i])

    ax7.set_ylabel('Cumulative Flow (m³)', fontsize=10, fontweight='bold')
    ax7.set_title('Cumulative Baseflow', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # ========== Plot 8: Scatter plots (True vs Estimated) ==========
    for idx, method in enumerate(methods[:3]):
        ax = fig.add_subplot(gs[3, idx])
        estimated = baseflow_df[method].values

        # Scatter plot
        ax.scatter(true_baseflow, estimated, alpha=0.5, s=10, color=colors[idx])

        # 1:1 line
        min_val = min(true_baseflow.min(), estimated.min())
        max_val = max(true_baseflow.max(), estimated.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

        # Linear regression
        z = np.polyfit(true_baseflow, estimated, 1)
        p = np.poly1d(z)
        ax.plot(true_baseflow, p(true_baseflow), 'g-', linewidth=2, alpha=0.7,
                label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

        # Statistics
        r2 = np.corrcoef(true_baseflow, estimated)[0, 1]**2
        rmse = np.sqrt(np.mean((estimated - true_baseflow)**2))

        ax.set_xlabel('True Baseflow (m³/s)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Estimated (m³/s)', fontsize=10, fontweight='bold')
        ax.set_title(f'{method}\nR²={r2:.3f}, RMSE={rmse:.2f}',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Save figure
    output_file = output_dir / 'advanced_visualization.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved comprehensive plot to: {output_file}")

    plt.close()


def create_performance_heatmap(flow_series, baseflow_df, true_baseflow, kge_scores, output_dir=None):
    """Create a heatmap showing method performance across different metrics.

    Args:
        flow_series: Original streamflow series
        baseflow_df: DataFrame with separated baseflow from multiple methods
        true_baseflow: True baseflow
        kge_scores: KGE performance scores
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    Q = flow_series.values
    methods = baseflow_df.columns.tolist()

    # Calculate various metrics
    metrics = {
        'KGE': [],
        'RMSE': [],
        'MAE': [],
        'R²': [],
        'BFI_Error': [],
        'Bias': []
    }

    true_bfi = true_baseflow.sum() / Q.sum()

    for method in methods:
        estimated = baseflow_df[method].values

        # KGE
        metrics['KGE'].append(kge_scores[method])

        # RMSE
        rmse = np.sqrt(np.mean((estimated - true_baseflow)**2))
        metrics['RMSE'].append(rmse)

        # MAE
        mae = np.mean(np.abs(estimated - true_baseflow))
        metrics['MAE'].append(mae)

        # R²
        r2 = np.corrcoef(true_baseflow, estimated)[0, 1]**2
        metrics['R²'].append(r2)

        # BFI Error
        estimated_bfi = estimated.sum() / Q.sum()
        bfi_error = abs(estimated_bfi - true_bfi)
        metrics['BFI_Error'].append(bfi_error)

        # Bias
        bias = np.mean(estimated - true_baseflow)
        metrics['Bias'].append(bias)

    # Create DataFrame for heatmap
    df_metrics = pd.DataFrame(metrics, index=methods)

    # Normalize metrics for visualization (0-1 scale)
    df_normalized = df_metrics.copy()
    for col in df_normalized.columns:
        if col in ['KGE', 'R²']:
            # Higher is better, already 0-1
            df_normalized[col] = df_normalized[col]
        else:
            # Lower is better, normalize and invert
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:
                df_normalized[col] = 1 - (df_normalized[col] - min_val) / (max_val - min_val)
            else:
                df_normalized[col] = 1.0

    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Actual values heatmap
    im1 = ax1.imshow(df_metrics.values, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(np.arange(len(df_metrics.columns)))
    ax1.set_yticks(np.arange(len(methods)))
    ax1.set_xticklabels(df_metrics.columns, fontsize=10, fontweight='bold')
    ax1.set_yticklabels(methods, fontsize=10)
    ax1.set_title('Performance Metrics (Actual Values)', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(df_metrics.columns)):
            text = ax1.text(j, i, f'{df_metrics.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im1, ax=ax1)

    # Normalized heatmap
    im2 = ax2.imshow(df_normalized.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(df_normalized.columns)))
    ax2.set_yticks(np.arange(len(methods)))
    ax2.set_xticklabels(df_normalized.columns, fontsize=10, fontweight='bold')
    ax2.set_yticklabels(methods, fontsize=10)
    ax2.set_title('Performance Metrics (Normalized 0-1, 1=Best)', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(df_normalized.columns)):
            text = ax2.text(j, i, f'{df_normalized.values[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()

    output_file = output_dir / 'performance_heatmap.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved performance heatmap to: {output_file}")

    plt.close()

    return df_metrics


def main():
    """Run advanced visualization example."""
    print("=" * 70)
    print("ADVANCED VISUALIZATION FOR BASEFLOW SEPARATION")
    print("=" * 70)

    # Generate test data
    print("\n1. Generating synthetic streamflow data...")
    Q, true_baseflow, quickflow = generate_streamflow(
        n_days=730,  # 2 years
        base_flow=20.0,
        seasonal_amplitude=8.0,
        n_storm_events=50,
        storm_intensity=70.0,
        bfi=0.65,
        random_seed=42
    )

    dates = pd.date_range('2020-01-01', periods=len(Q), freq='D')
    flow_series = pd.Series(Q, index=dates)

    print(f"   - Generated {len(Q)} days of data")
    print(f"   - Mean flow: {Q.mean():.2f} m³/s")
    print(f"   - True BFI: {true_baseflow.sum() / Q.sum():.3f}")

    # Apply all methods
    print("\n2. Applying baseflow separation methods...")
    baseflow_df, kge_scores = single(
        flow_series,
        area=2000.0,
        method="all",
        return_kge=True
    )

    print(f"   - Applied {len(baseflow_df.columns)} methods")

    # Create visualizations
    print("\n3. Creating advanced visualizations...")
    output_dir = Path(__file__).parent

    create_comprehensive_plots(flow_series, baseflow_df, true_baseflow, kge_scores, output_dir)
    df_metrics = create_performance_heatmap(flow_series, baseflow_df, true_baseflow, kge_scores, output_dir)

    # Print summary
    print("\n4. Performance Summary:")
    print(df_metrics.round(3).to_string())

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - advanced_visualization.png")
    print("  - performance_heatmap.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
