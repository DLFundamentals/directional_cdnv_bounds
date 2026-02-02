import csv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter
from scipy.stats import pearsonr


def set_border(g):
    for spine in ["top", "bottom", "left", "right"]:
        g.spines[spine].set_color("black")
        g.spines[spine].set_linewidth(1)


def load_cdnv_csv(csv_path):
    """Load CDNV and directional CDNV statistics from a CSV file.

    Expects at least columns: epoch, train_cdnv, train_dir_cdnv.
    Optionally also supports: val_cdnv, val_dir_cdnv.
    Rows with epoch == "last" are ignored.
    """

    epochs = []
    train_cdnv = []
    train_dir_cdnv = []
    # Validation columns (test) are read if present
    val_cdnv = []
    val_dir_cdnv = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch_str = row.get("epoch", "")
            if epoch_str == "last" or epoch_str == "":
                continue

            epochs.append(float(epoch_str))
            train_cdnv.append(float(row["train_cdnv"]))
            train_dir_cdnv.append(float(row["train_dir_cdnv"]))

            # Optional columns
            if "val_cdnv" in row and row["val_cdnv"] != "":
                val_cdnv.append(float(row["val_cdnv"]))
            if "val_dir_cdnv" in row and row["val_dir_cdnv"] != "":
                val_dir_cdnv.append(float(row["val_dir_cdnv"]))

    return (
        np.array(epochs),
        np.array(train_cdnv),
        np.array(train_dir_cdnv),
        np.array(val_cdnv) if val_cdnv else None,
        np.array(val_dir_cdnv) if val_dir_cdnv else None,
    )


def plot_cdnv(
    epochs,
    train_cdnv,
    train_dir_cdnv,
    val_cdnv=None,
    val_dir_cdnv=None,
    show_corr=True,
    output_path=None,
    figsize=(14, 11),
):
    """Plot train and test CDNV and directional CDNV vs. epochs.

    - X-axis: linear epochs
    - Y-axis: log-scale CDNV
    - Colors: CDNV (blue), Dir CDNV (red)
    - Styles: Train (solid with circle), Test (dashed with star)
    """

    sns.set_theme(
        style="whitegrid",
        font_scale=3.0,
        rc={"xtick.bottom": True, "ytick.left": True},
    )
    sns.set_context(rc={"patch.linewidth": 2.0})
    plt.figure(figsize=figsize)

    # Colors for the two metrics
    cdnv_color = "blue"      # CDNV
    dir_cdnv_color = "red"   # Dir CDNV

    # Sort by epoch to ensure monotonic x-axis ordering
    sort_idx = np.argsort(epochs)
    epochs = np.array(epochs, dtype=float)[sort_idx]
    train_cdnv = np.array(train_cdnv)[sort_idx]
    train_dir_cdnv = np.array(train_dir_cdnv)[sort_idx]

    if val_cdnv is not None and val_dir_cdnv is not None:
        val_cdnv = np.array(val_cdnv)[sort_idx]
        val_dir_cdnv = np.array(val_dir_cdnv)[sort_idx]

    # For plotting on a log-scale x-axis, we cannot place epoch 0 exactly.
    # We instead map epoch 0 to a small positive value (2^-1 = 0.5) so it
    # appears just to the left of epoch 1, and label it as "0".
    x_plot = epochs.copy()
    has_zero_epoch = np.any(x_plot == 0)
    if has_zero_epoch:
        x_plot[x_plot == 0] = 0.5

    # Pearson correlation between CDNV and dir-CDNV (train), if requested
    if show_corr:
        corr_train, p_train = pearsonr(train_cdnv, train_dir_cdnv)

    # Lines: CDNV and Dir CDNV over epochs, for train and (optionally) test
    sns.lineplot(
        x=x_plot,
        y=train_cdnv,
        marker="o",
        color=cdnv_color,
        linewidth=2.0,
        markeredgecolor="black",
    )
    sns.lineplot(
        x=x_plot,
        y=train_dir_cdnv,
        marker="o",
        color=dir_cdnv_color,
        linewidth=2.0,
        markeredgecolor="black",
    )

    if val_cdnv is not None and val_dir_cdnv is not None:
        sns.lineplot(
            x=x_plot,
            y=val_cdnv,
            marker="*",
            color=cdnv_color,
            linestyle="--",
            linewidth=2.0,
            markeredgecolor="black",
        )
        sns.lineplot(
            x=x_plot,
            y=val_dir_cdnv,
            marker="*",
            color=dir_cdnv_color,
            linestyle="--",
            linewidth=2.0,
            markeredgecolor="black",
        )

    ax = plt.gca()
    set_border(ax)

    # Legends
    metric_handles = [
        mlines.Line2D([], [], color=cdnv_color, linestyle="-", label="CDNV"),
        mlines.Line2D([], [], color=dir_cdnv_color, linestyle="-", label="Dir CDNV"),
    ]

    style_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", marker="o", label="Train", markersize=12),
        mlines.Line2D([], [], color="black", linestyle="--", marker="*", label="Test", markersize=15),
    ]

    # Place legends close together in the upper-right region without
    # overlapping the plotted curves too much.  
    legend1 = plt.legend(handles=metric_handles, loc="upper right", fontsize=30)
    ax.add_artist(legend1)
    plt.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.45, 0.99),  # slightly right from previous to balance spacing
        borderaxespad=0.3,
        fontsize=30,
    )

    # Optional correlation legend
    if show_corr:
        corr_handles = [
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="",
                label=rf"$\rho$, p-val (Train CDNV vs Dir CDNV): {corr_train:.2f}, {p_train:.2f}",
            ),
        ]
        legend2 = plt.legend(
            handles=corr_handles,
            loc="lower left",
            fontsize=25,
            handlelength=0,
            handletextpad=0.4,
        )
        legend2.get_frame().set_alpha(0.6)
        ax.add_artist(legend2)

    # X-axis handling (log2 scale with a special position for epoch 0).
    # x_plot already maps epoch 0 to 0.5; all other epochs remain unchanged.
    positive_x = x_plot[x_plot > 0]
    if positive_x.size > 0:
        min_x_exp = 0
        max_x_exp = 10
        x_exps = np.arange(min_x_exp, max_x_exp + 1)
        xticks = 2.0 ** x_exps

        if has_zero_epoch:
            xticks = np.concatenate(([0.5], xticks))

        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)

        def _x_log_formatter(x, _):
            if np.isclose(x, 0.5):
                return "0"
            if x <= 0:
                return ""
            exp = int(np.round(np.log2(x)))
            if not np.isclose(x, 2.0 ** exp):
                return ""
            if exp % 2 != 0:
                return ""
            if exp == 0:
                return "1"
            return rf"$2^{{{exp}}}$"

        ax.xaxis.set_major_formatter(FuncFormatter(_x_log_formatter))
    else:
        ax.set_xscale("log", base=2)

    all_vals = [train_cdnv, train_dir_cdnv]
    if val_cdnv is not None and val_dir_cdnv is not None:
        all_vals.extend([val_cdnv, val_dir_cdnv])

    all_vals = np.concatenate([v for v in all_vals if v is not None])
    positive_vals = all_vals[all_vals > 0]

    if positive_vals.size > 0:
        min_exp = int(np.floor(np.log2(positive_vals.min())))
        max_exp = int(np.ceil(np.log2(positive_vals.max())))
        exps = np.arange(min_exp, max_exp + 1)
        yticks = 2.0 ** exps

        ax.set_yscale("log", base=2)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"$2^{{{int(np.round(np.log2(y)))}}}$")
        )
    else:
        ax.set_yscale("log", base=2)
    plt.xlabel("Epoch")

    # Set x-limits compatible with log scale and cap at 2^10.
    if x_plot.size > 0:
        xmin = x_plot[x_plot > 0].min() if np.any(x_plot > 0) else 0.5
        # Ensure lower bound does not go below the epoch-0 pseudo position.
        if has_zero_epoch:
            xmin = min(xmin, 0.5)
        plt.xlim(xmin * 0.9, 2**10)
    plt.ylim(2**(-5),2**4)
    plt.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8, alpha=0.9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def plot_cdnv_from_csv(csv_path, output_path, figsize=(14, 11), show_corr=True):
    """Convenience wrapper: load from CSV and plot."""

    (
        epochs,
        train_cdnv,
        train_dir_cdnv,
        val_cdnv,
        val_dir_cdnv,
    ) = load_cdnv_csv(csv_path)

    plot_cdnv(
        epochs,
        train_cdnv,
        train_dir_cdnv,
        val_cdnv=val_cdnv,
        val_dir_cdnv=val_dir_cdnv,
        show_corr=show_corr,
        output_path=output_path,
        figsize=figsize,
    )


def plot_cdnv_from_two_csvs(
    csv_path_1,
    csv_path_2,
    output_path=None,
    figsize=(14, 11),
    show_corr=True,
    filename = None,
    labels=("10 Classes", "100 Classes"),
):
    """Load two CDNV CSVs and plot them together on the same axes.

    Formatting (scales, colors, fonts, etc.) matches ``plot_cdnv_from_csv``.
    """

    (
        epochs_1,
        train_cdnv_1,
        train_dir_cdnv_1,
        val_cdnv_1,
        val_dir_cdnv_1,
    ) = load_cdnv_csv(csv_path_1)

    (
        epochs_2,
        train_cdnv_2,
        train_dir_cdnv_2,
        val_cdnv_2,
        val_dir_cdnv_2,
    ) = load_cdnv_csv(csv_path_2)

    sns.set_theme(
        style="whitegrid",
        font_scale=3.0,
        rc={"xtick.bottom": True, "ytick.left": True},
    )
    sns.set_context(rc={"patch.linewidth": 2.0})
    plt.figure(figsize=figsize)

    cdnv_color = "blue"
    dir_cdnv_color = "red"

    # Sort each run by epoch to ensure monotonic ordering
    def _sort_by_epoch(epochs, *arrays):
        idx = np.argsort(epochs)
        sorted_epochs = np.array(epochs, dtype=float)[idx]
        sorted_arrays = [np.array(a)[idx] if a is not None and len(a) > 0 else None for a in arrays]
        return (sorted_epochs, *sorted_arrays)

    epochs_1, train_cdnv_1, train_dir_cdnv_1, val_cdnv_1, val_dir_cdnv_1 = _sort_by_epoch(
        epochs_1,
        train_cdnv_1,
        train_dir_cdnv_1,
        val_cdnv_1,
        val_dir_cdnv_1,
    )
    epochs_2, train_cdnv_2, train_dir_cdnv_2, val_cdnv_2, val_dir_cdnv_2 = _sort_by_epoch(
        epochs_2,
        train_cdnv_2,
        train_dir_cdnv_2,
        val_cdnv_2,
        val_dir_cdnv_2,
    )

    # Map epoch 0 to 0.5 on log2 scale for each run separately
    def _map_zero(epochs):
        epochs = epochs.copy()
        has_zero = np.any(epochs == 0)
        if has_zero:
            epochs[epochs == 0] = 0.5
        return epochs, has_zero

    x_plot_1, has_zero_1 = _map_zero(epochs_1)
    x_plot_2, has_zero_2 = _map_zero(epochs_2)
    has_zero_epoch = has_zero_1 or has_zero_2

    # Pearson correlations (train only), if requested
    if show_corr:
        corr_train_1, p_train_1 = pearsonr(train_cdnv_1, train_dir_cdnv_1)
        corr_train_2, p_train_2 = pearsonr(train_cdnv_2, train_dir_cdnv_2)

    # Run 1: use green/orange (e.g., 10 Superclasses)
    sns.lineplot(
        x=x_plot_1,
        y=train_cdnv_1,
        marker="o",
        color="darkgreen",
        linewidth=2.0,
        markeredgecolor="black",
    )
    sns.lineplot(
        x=x_plot_1,
        y=train_dir_cdnv_1,
        marker="o",
        color="darkorange",
        linewidth=2.0,
        markeredgecolor="black",
    )

    if val_cdnv_1 is not None and val_dir_cdnv_1 is not None and len(val_cdnv_1) > 0 and len(val_dir_cdnv_1) > 0:
        sns.lineplot(
            x=x_plot_1,
            y=val_cdnv_1,
            marker="*",
            color="darkgreen",
            linestyle="--",
            linewidth=2.0,
            markeredgecolor="black",
        )
        sns.lineplot(
            x=x_plot_1,
            y=val_dir_cdnv_1,
            marker="*",
            color="darkorange",
            linestyle="--",
            linewidth=2.0,
            markeredgecolor="black",
        )

    # Run 2: use blue/red and draw on top (e.g., 100 Classes)
    sns.lineplot(
        x=x_plot_2,
        y=train_cdnv_2,
        marker="o",
        color=cdnv_color,
        linewidth=2.0,
        markeredgecolor="black",
    )
    sns.lineplot(
        x=x_plot_2,
        y=train_dir_cdnv_2,
        marker="o",
        color=dir_cdnv_color,
        linewidth=2.0,
        markeredgecolor="black",
    )

    if val_cdnv_2 is not None and val_dir_cdnv_2 is not None and len(val_cdnv_2) > 0 and len(val_dir_cdnv_2) > 0:
        sns.lineplot(
            x=x_plot_2,
            y=val_cdnv_2,
            marker="*",
            color=cdnv_color,
            linestyle="--",
            linewidth=2.0,
            markeredgecolor="black",
        )
        sns.lineplot(
            x=x_plot_2,
            y=val_dir_cdnv_2,
            marker="*",
            color=dir_cdnv_color,
            linestyle="--",
            linewidth=2.0,
            markeredgecolor="black",
        )

    ax = plt.gca()
    set_border(ax)

    style_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", marker="o", label="Train", markersize=12),
        mlines.Line2D([], [], color="black", linestyle="--", marker="*", label="Test", markersize=15),
    ]

    # Second legend: CDNV/Dir CDNV for each run
    metric_run_handles = [
        mlines.Line2D([], [], color="darkgreen", linestyle="-", label=f"CDNV ({labels[0]})"),
        mlines.Line2D([], [], color="darkorange", linestyle="-", label=f"Dir CDNV ({labels[0]})"),
        mlines.Line2D([], [], color=cdnv_color, linestyle="-", label=f"CDNV ({labels[1]})"),
        mlines.Line2D([], [], color=dir_cdnv_color, linestyle="-", label=f"Dir CDNV ({labels[1]})"),
    ]

    # Legend 1: Train / Test (inside plot, left/top)
    legend_style = plt.legend(
        handles=style_handles,
        loc="lower left",
         bbox_to_anchor=(0.48, -0.0031),
        fontsize=30,
        framealpha=0.9,
    )
    ax.add_artist(legend_style)

    # Legend 2: CDNV / Dir CDNV for each run (inside plot, right/top)
    legend_metrics = plt.legend(
        handles=metric_run_handles,
        loc="lower left",
        #bbox_to_anchor=(0.72, 0.99),  # shifted slightly left
        fontsize=25,
        ncol=1,
        framealpha=0.9,
    )
    ax.add_artist(legend_metrics)

    if show_corr:
        corr_handles = [
            mlines.Line2D(
                [],
                [],
                linestyle="none",
                marker="",
                label=(
                    rf"{labels[0]}: $\\rho$, p-val: {corr_train_1:.2f}, {p_train_1:.2f}\n"
                    rf"{labels[1]}: $\\rho$, p-val: {corr_train_2:.2f}, {p_train_2:.2f}"
                ),
            ),
        ]
        legend2 = plt.legend(
            handles=corr_handles,
            loc="lower left",
            fontsize=25,
            handlelength=0,
            handletextpad=0.4,
        )
        legend2.get_frame().set_alpha(0.6)
        ax.add_artist(legend2)

    # X-axis: log2 scale with special handling for epoch 0 in either run
    positive_x = np.concatenate([
        x_plot_1[x_plot_1 > 0],
        x_plot_2[x_plot_2 > 0],
    ])
    if positive_x.size > 0:
        min_x_exp = 0
        max_x_exp = int(np.ceil(np.log2(positive_x.max())))
        x_exps = np.arange(min_x_exp, max_x_exp + 1)
        xticks = 2.0 ** x_exps

        if has_zero_epoch:
            xticks = np.concatenate(([0.5], xticks))

        ax.set_xscale("log", base=2)
        ax.set_xticks(xticks)

        def _x_log_formatter(x, _):
            if np.isclose(x, 0.5):
                return "0"
            if x <= 0:
                return ""
            exp = int(np.round(np.log2(x)))
            if not np.isclose(x, 2.0 ** exp):
                return ""
            if exp % 2 != 0:
                return ""
            if exp == 0:
                return "1"
            return rf"$2^{{{exp}}}$"

        ax.xaxis.set_major_formatter(FuncFormatter(_x_log_formatter))
    else:
        ax.set_xscale("log", base=2)

    # Y-axis: log2 scale based on all positive values from both runs
    all_vals = [train_cdnv_1, train_dir_cdnv_1, train_cdnv_2, train_dir_cdnv_2]
    for maybe_val in [val_cdnv_1, val_dir_cdnv_1, val_cdnv_2, val_dir_cdnv_2]:
        if maybe_val is not None and len(maybe_val) > 0:
            all_vals.append(maybe_val)

    all_vals = np.concatenate(all_vals)
    positive_vals = all_vals[all_vals > 0]

    if positive_vals.size > 0:
        min_exp = int(np.floor(np.log2(positive_vals.min())))
        max_exp = int(np.ceil(np.log2(positive_vals.max())))
        exps = np.arange(min_exp, max_exp + 1)
        yticks = 2.0 ** exps

        ax.set_yscale("log", base=2)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"$2^{{{int(np.round(np.log2(y)))}}}$")
        )
    else:
        ax.set_yscale("log", base=2)

    plt.xlabel("Epoch")
    plt.ylabel("Normalized Variance")

    # X-limits based on both runs
    all_x = np.concatenate([x_plot_1, x_plot_2]) if x_plot_1.size + x_plot_2.size > 0 else np.array([])
    if all_x.size > 0:
        positive_all_x = all_x[all_x > 0]
        xmin = positive_all_x.min() if positive_all_x.size > 0 else 0.5
        xmax = all_x.max()
        if has_zero_epoch:
            xmin = min(xmin, 0.5)
        plt.xlim(xmin * 0.9, xmax * 1.05)

    plt.ylim(1e-2, 1e2)
    plt.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8, alpha=0.9)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot CDNV and Dir CDNV vs epochs from a CSV file.",
    )
    parser.add_argument("csv_path", type=str, help="Path to cdnv CSV file")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g., .png or .pdf)",
    )

    args = parser.parse_args()
    plot_cdnv_from_csv(args.csv_path, output_path=args.output_path)

