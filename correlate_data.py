'''
Correlates Boltz-2 predicted affinity against experimental IC50/EC50/Ki potency, per
receptor, and saves a scatter-plot PNG per receptor (one subplot per activity type).

Usage:
python correlate_data.py --input combined_results.csv --output-dir correlation_plots

Expects the output of `combine_results.py --input subtype_pairs.csv ...` as input: a table
with one or more `<receptor>_affinity_pred_value` columns (e.g. M1_affinity_pred_value,
M2_affinity_pred_value, ...) plus per-receptor `<receptor>_Ki_nm` / `<receptor>_IC50_nm` /
`<receptor>_EC50_nm` columns from the subtype-pairs reference merge (e.g. M1_Ki_nm pairs
with M1_affinity_pred_value, M2_Ki_nm with M2_affinity_pred_value, ...). If the input has
no reference columns, re-run combine_results.py with --input first.

For each receptor, plots predicted affinity (x) against each available experimental
activity type (y, log scale), and annotates the Spearman rank correlation (rho), its
p-value, and the number of paired data points. Pairs with fewer than --min-points valid
points are skipped (noted on the plot) rather than shown as a misleading correlation.
'''

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Activity types to plot, in the same order as filter_searched_results.py's subtype-pairs
# output columns (<receptor>_Ki_nm, <receptor>_IC50_nm, <receptor>_EC50_nm).
ACTIVITY_TYPES = ['Ki', 'IC50', 'EC50']

RECEPTOR_AFFINITY_PATTERN = re.compile(r'^(M\d+)_affinity_pred_value$')

# Colors from the project's validated default palette (dataviz skill): categorical
# slot 1 (blue) for the single-series marker, chart chrome/ink for everything else.
MARKER_COLOR = '#2a78d6'
CHART_SURFACE = '#fcfcfb'
GRIDLINE_COLOR = '#e1e0d9'
AXIS_COLOR = '#c3c2b7'
MUTED_TEXT_COLOR = '#898781'
PRIMARY_TEXT_COLOR = '#0b0b0b'


def parse_arguments():
    """Parse command line arguments for configuration options."""
    parser = argparse.ArgumentParser(
        description="Correlate Boltz-2 predicted affinity against experimental IC50/EC50/Ki, per receptor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        default='combined_results.csv',
        help='combine_results.py output CSV (must have been run with --input subtype_pairs.csv)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save the per-receptor correlation plot PNGs'
    )

    parser.add_argument(
        '--min-points',
        type=int,
        default=4,
        help='Minimum number of paired non-null data points required to plot a correlation'
    )

    return parser.parse_args()


def find_receptors(df: pd.DataFrame) -> list:
    """Find receptor prefixes (e.g. 'M1') that have an affinity_pred_value column,
    ordered numerically (M1, M2, ..., M10, ...)."""
    receptors = [m.group(1) for col in df.columns if (m := RECEPTOR_AFFINITY_PATTERN.match(col))]
    return sorted(receptors, key=lambda r: int(re.search(r'\d+', r).group()))


def compute_correlation(df: pd.DataFrame, x_col: str, y_col: str, min_points: int):
    """Return (x_values, y_values, rho, p_value, n) for the paired non-null rows.
    rho/p_value are None if there are fewer than min_points pairs, or if the values
    have no variability (Spearman undefined)."""
    paired = df[[x_col, y_col]].dropna()
    n = len(paired)
    if n < min_points:
        return paired[x_col].values, paired[y_col].values, None, None, n

    rho, p_value = spearmanr(paired[x_col], paired[y_col])
    if np.isnan(rho):
        return paired[x_col].values, paired[y_col].values, None, None, n

    return paired[x_col].values, paired[y_col].values, rho, p_value, n


def plot_receptor_correlations(df: pd.DataFrame, receptor: str, activity_types: list,
                                min_points: int, output_dir: str) -> bool:
    """Plot one figure (one subplot per activity type) for a single receptor's predicted
    affinity vs each available experimental activity value. Returns True if the figure
    was saved (at least one subplot had enough data to plot)."""
    x_col = f"{receptor}_affinity_pred_value"
    if x_col not in df.columns:
        logger.warning(f"{receptor}: no '{x_col}' column, skipping")
        return False

    fig, axes = plt.subplots(1, len(activity_types), figsize=(5 * len(activity_types), 4.5))
    if len(activity_types) == 1:
        axes = [axes]

    any_plotted = False
    for ax, activity_name in zip(axes, activity_types):
        y_col = f"{receptor}_{activity_name}_nm"
        ax.set_facecolor(CHART_SURFACE)

        if y_col not in df.columns:
            ax.set_title(activity_name, color=MUTED_TEXT_COLOR, fontsize=11)
            ax.text(0.5, 0.5, f"no {y_col} column", ha='center', va='center',
                     color=MUTED_TEXT_COLOR, transform=ax.transAxes)
            ax.set_axis_off()
            continue

        x_vals, y_vals, rho, p_value, n = compute_correlation(df, x_col, y_col, min_points)

        if rho is None:
            ax.set_title(activity_name, color=MUTED_TEXT_COLOR, fontsize=11)
            ax.text(0.5, 0.5, f"insufficient data (n={n})", ha='center', va='center',
                     color=MUTED_TEXT_COLOR, transform=ax.transAxes)
            ax.set_axis_off()
            logger.warning(
                f"{receptor} / {activity_name}: only {n} usable paired points "
                f"(need >= {min_points} with variability), skipping"
            )
            continue

        any_plotted = True
        ax.scatter(x_vals, y_vals, s=45, color=MARKER_COLOR, edgecolors=CHART_SURFACE,
                   linewidths=1, alpha=0.9)
        ax.set_yscale('log')
        ax.set_xlabel(f"{receptor} predicted affinity (affinity_pred_value)", color=MUTED_TEXT_COLOR)
        ax.set_ylabel(f"{activity_name} (nM, log scale)", color=MUTED_TEXT_COLOR)
        ax.set_title(f"{activity_name}: Spearman ρ = {rho:.2f} (n={n}, p={p_value:.3f})",
                     color=PRIMARY_TEXT_COLOR, fontsize=11)
        ax.grid(True, color=GRIDLINE_COLOR, linewidth=1)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(AXIS_COLOR)
        ax.tick_params(colors=MUTED_TEXT_COLOR)

        logger.info(f"{receptor} / {activity_name}: Spearman rho={rho:.3f}, p={p_value:.4f}, n={n}")

    if not any_plotted:
        plt.close(fig)
        return False

    fig.suptitle(f"{receptor}: predicted affinity vs experimental potency",
                 color=PRIMARY_TEXT_COLOR, fontsize=13)
    fig.tight_layout()

    output_path = Path(output_dir) / f"{receptor}_correlation.png"
    fig.savefig(output_path, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)
    logger.info(f"Saved {output_path}")
    return True


def main():
    """Main execution function."""
    args = parse_arguments()
    df = pd.read_csv(args.input)

    receptors = find_receptors(df)
    if not receptors:
        raise ValueError(f"No '<receptor>_affinity_pred_value' columns found in '{args.input}'")

    expected_columns = [f"{r}_{a}_nm" for r in receptors for a in ACTIVITY_TYPES]
    if not any(col in df.columns for col in expected_columns):
        raise ValueError(
            f"'{args.input}' has none of the expected <receptor>_{{Ki,IC50,EC50}}_nm columns -- "
            f"run combine_results.py with --input first"
        )

    logger.info(f"Found {len(receptors)} receptors: {receptors}")
    logger.info(f"Activity types to plot: {ACTIVITY_TYPES}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    plotted = 0
    for receptor in receptors:
        if plot_receptor_correlations(df, receptor, ACTIVITY_TYPES, args.min_points, args.output_dir):
            plotted += 1

    logger.info(f"Generated correlation plots for {plotted}/{len(receptors)} receptors")


if __name__ == "__main__":
    main()
