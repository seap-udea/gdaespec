#!/usr/bin/env python3
"""Plot aggregate timing and metric summaries for the retrieval campaign.

The script reads only `campaign_5obs/times.csv` and `campaign_5obs/metrics.csv`
so figures can be regenerated without loading the full POSEIDON outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter
import matplotlib as mpl
import numpy as np
import pandas as pd

from campaign_common import CAMPAIGN_DIR, CLEAN_SPECTRUM_PATH


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

# Scientific ordering for the x-axis and legend entries.
TARGET_CASES = [(0.00, 0.00), (0.01, 0.08), (0.08, 0.54), (0.26, 0.70)]
TARGET_CASE_SET = {(round(s, 2), round(f, 2)) for s, f in TARGET_CASES}
SOURCE_ORDER = ["phoenix", "sphinx"]
SOURCE_LABELS = {"phoenix": "PHOENIX", "sphinx": "SPHINX"}
SOURCE_LINESTYLES = {"phoenix": "-", "sphinx": "--"}

STRATEGY_LABELS = {
    "gdae": "G-DAE + Chem",
    "contam": "Cont + Chem",
}
STRATEGY_COLORS = {
    "contam": "#264653",
    "gdae": "#E76F51",
}
STRATEGY_MARKERS = {"contam": "s", "gdae": "D"}
BRANCH_STRATEGY_MARKERS = {
    ("gdae", "phoenix"): "^",
    ("gdae", "sphinx"): "v",
    ("contam", "phoenix"): "s",
    ("contam", "sphinx"): "D",
}
STRATEGY_ORDER = ["contam", "gdae"]

# Shared plotting style constants.
LINE_WIDTH = 3.0
MARKER_SIZE = 7.2
ERROR_BAR_WIDTH = 1.35
PAIR_LINEWIDTH = 0.8
PAIR_ALPHA = 0.16
EDGE_COLOR = "#1f1f1f"
BASE_OFFSETS = {"contam": -0.16, "gdae": 0.16}
SOURCE_OFFSETS = {"phoenix": -0.045, "sphinx": 0.045}


def case_label(row: pd.Series) -> str:
    return f"{row['f_spot']:.2f}, {row['f_fac']:.2f}"


def load_csv(path: Path, value_col: str, exclude_test01: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if exclude_test01:
        df = df[df["id"] != "test_01"].copy()
    df["f_spot"] = df["f_spot"].astype(float)
    df["f_fac"] = df["f_fac"].astype(float)
    df[value_col] = df[value_col].astype(float)
    df["case"] = df.apply(case_label, axis=1)
    df["case_key"] = df.apply(lambda row: (round(row["f_spot"], 2), round(row["f_fac"], 2)), axis=1)
    df = df[df["case_key"].isin(TARGET_CASE_SET)].copy()
    return df


def clean_spectrum_signal(path: Path = CLEAN_SPECTRUM_PATH) -> float:
    """Return the pure-spectrum feature amplitude used to normalize MSE."""
    arr = np.genfromtxt(path, comments="#", dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in clean spectrum: {path}")
    depth = np.asarray(arr[:, 1], dtype=float)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError(f"Clean spectrum has no finite depth values: {path}")
    depth = depth[finite]
    signal = float(np.nanmax(depth) - np.nanmin(depth))
    if signal <= 0.0:
        raise ValueError(f"Clean spectrum signal must be positive, got {signal}.")
    return signal


def aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Aggregate repeated tests by branch, case, and strategy."""
    group_cols = ["branch", "f_spot", "f_fac", "case", "case_key", "strategy"]
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            mean=(value_col, "mean"),
            sigma=(value_col, lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            n_tests=(value_col, "count"),
        )
        .sort_values(["branch", "f_spot", "f_fac", "strategy"])
    )
    x_center_map = {case: i for i, case in enumerate(TARGET_CASES)}
    out["x_center"] = out["case_key"].map(x_center_map)
    out["xpos"] = out.apply(
        lambda row: float(
            row["x_center"]
            + BASE_OFFSETS[row["strategy"]]
            + SOURCE_OFFSETS[row["branch"]]
        ),
        axis=1,
    )
    return out


def marker_for(strategy: str, branch: str, marker_mode: str) -> str | None:
    if marker_mode == "none":
        return None
    if marker_mode == "branch_strategy":
        return BRANCH_STRATEGY_MARKERS[(strategy, branch)]
    return STRATEGY_MARKERS[strategy]


def plot_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    ylabel: str,
    title: str,
    use_log: bool,
    marker_mode: str,
) -> None:
    for boundary in range(len(TARGET_CASES) - 1):
        ax.axvline(boundary + 0.5, color="#d8d8d8", lw=0.8, ls=":", zorder=0)
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)

    for strategy in STRATEGY_ORDER:
        sub_pair = summary[summary["strategy"] == strategy].copy()
        for _, grp in sub_pair.groupby("case_key"):
            grp = grp.sort_values("branch")
            if grp.shape[0] < 2:
                continue
            vals = grp["mean"].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                continue
            ax.plot(
                grp["xpos"].to_numpy(dtype=float),
                vals,
                color="#8a8a8a",
                lw=PAIR_LINEWIDTH,
                alpha=PAIR_ALPHA,
                zorder=1,
            )

    for strategy in STRATEGY_ORDER:
        for branch in SOURCE_ORDER:
            sub = summary[(summary["strategy"] == strategy) & (summary["branch"] == branch)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["x_center", "xpos"])
            y = sub["mean"].to_numpy(dtype=float)
            yerr = sub["sigma"].to_numpy(dtype=float)
            x = sub["xpos"].to_numpy(dtype=float)

            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            x, y, yerr = x[mask], y[mask], yerr[mask]

            color = STRATEGY_COLORS[strategy]
            marker = marker_for(strategy, branch, marker_mode)
            markerface = color
            markeredge = EDGE_COLOR
            mew = 1.0

            x_line = x
            y_line = y
            if branch == "sphinx" and (0.0, 0.0) not in set(sub["case_key"]):
                clean = summary[
                    (summary["strategy"] == strategy)
                    & (summary["branch"] == "phoenix")
                    & (summary["case_key"] == (0.0, 0.0))
                ]
                if not clean.empty:
                    clean_x = float(
                        clean["x_center"].iloc[0]
                        + BASE_OFFSETS[strategy]
                        + SOURCE_OFFSETS["sphinx"]
                    )
                    clean_y = float(clean["mean"].iloc[0])
                    x_line = np.concatenate([[clean_x], x])
                    y_line = np.concatenate([[clean_y], y])

            ax.plot(
                x_line,
                y_line,
                linestyle=SOURCE_LINESTYLES[branch],
                linewidth=LINE_WIDTH,
                color=color,
                alpha=0.80,
                zorder=2,
            )
            errorbar_kwargs = {
                "x": x,
                "y": y,
                "yerr": yerr,
                "capsize": 4.0,
                "capthick": ERROR_BAR_WIDTH * 0.8,
                "elinewidth": ERROR_BAR_WIDTH,
                "color": color,
                "linestyle": "None",
                "alpha": 0.92,
                "zorder": 3,
            }
            if marker is None:
                ax.errorbar(fmt="none", **errorbar_kwargs)
            else:
                ax.errorbar(
                    fmt=marker,
                    ms=MARKER_SIZE,
                    markerfacecolor=markerface,
                    markeredgecolor=markeredge,
                    markeredgewidth=mew,
                    **errorbar_kwargs,
                )

    ax.text(
        0.015,
        0.93,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        zorder=10,
    )
    ax.set_ylabel(ylabel, fontsize=13)
    if use_log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=80))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.22)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.20)
    ax.tick_params(labelsize=11)
    ax.tick_params(axis="y", which="minor", length=3.0, width=0.6)


def plot_comparison(
    metrics: pd.DataFrame,
    times: pd.DataFrame,
    out_name: str,
    exclude_test01: bool,
    marker_mode: str,
    error_metric: str,
    linear_scales: bool,
) -> Path:
    if error_metric == "mse_signal_percent":
        error_col = "MSE_signal_norm_percent"
        error_ylabel = r"MSE / signal$^2$ (%)"
        error_title = r"Signal-normalized MSE"
    elif error_metric == "rmse_signal_percent":
        error_col = "RMSE_signal_percent"
        error_ylabel = r"RMSE / signal (%)"
        error_title = r"Signal-normalized RMSE"
    else:
        raise ValueError(f"Unknown error metric: {error_metric}")

    metric_specs = [
        (aggregate(times, "delta_time"), "Time (min)", "Retrieval Time", False),
        (aggregate(metrics, error_col), error_ylabel, error_title, not linear_scales),
        (aggregate(metrics, "chi2_reduced"), r"$\chi^2_r$", r"Reduced $\chi^2$", not linear_scales),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.2, 9.8), sharex=True)
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.10, right=0.97, hspace=0.24)
    fig.suptitle("PHOENIX vs SPHINX Retrieval Metrics (10 Transits)", fontsize=21, y=0.99)

    for ax, (summary, ylabel, title, use_log) in zip(axes, metric_specs):
        plot_metric_panel(ax, summary, ylabel, title, use_log, marker_mode)

    x_labels = [f"{spot:.2f}/{fac:.2f}" for spot, fac in TARGET_CASES]
    axes[-1].set_xticks(range(len(x_labels)))
    axes[-1].set_xticklabels(x_labels, rotation=35, ha="right")
    axes[-1].set_xlabel(r"$f_{spot}/f_{fac}$", fontsize=14)
    axes[-1].axhline(1.0, color="gray", lw=1.3, ls=":", alpha=0.75)

    combined_handles = [
        Line2D(
            [0, 1],
            [0, 0],
            marker=marker_for(strategy, branch, marker_mode) if marker_mode != "none" else None,
            linestyle=SOURCE_LINESTYLES[branch],
            linewidth=LINE_WIDTH,
            color=STRATEGY_COLORS[strategy],
            markerfacecolor=STRATEGY_COLORS[strategy],
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=1.0,
            markersize=MARKER_SIZE,
            label=f"{SOURCE_LABELS[branch]} - {STRATEGY_LABELS[strategy]}",
        )
        for branch in SOURCE_ORDER
        for strategy in STRATEGY_ORDER
    ]

    fig.legend(
        handles=combined_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncols=4,
        frameon=False,
        fontsize=11.2,
        columnspacing=1.25,
        handlelength=3.0,
    )

    out_dir = CAMPAIGN_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-test01",
        action="store_true",
        help="Exclude test_01 from the test-dispersion aggregates.",
    )
    parser.add_argument(
        "--marker-mode",
        choices=("strategy", "branch_strategy", "none", "all"),
        default="strategy",
        help="Marker grammar: one marker per strategy, one marker per strategy/grid pair, no markers, or all variants.",
    )
    parser.add_argument(
        "--error-metric",
        choices=("mse_signal_percent", "rmse_signal_percent", "all"),
        default="mse_signal_percent",
        help="Normalized error metric for the middle panel.",
    )
    parser.add_argument(
        "--version-tag",
        default="v3",
        help="Version tag appended to the official aggregate plot filenames.",
    )
    parser.add_argument(
        "--linear-scales",
        action="store_true",
        help="Use linear y scales for normalized MSE and reduced chi-square.",
    )
    args = parser.parse_args()

    times = load_csv(CAMPAIGN_DIR / "times.csv", "delta_time", args.exclude_test01)
    metrics = load_csv(CAMPAIGN_DIR / "metrics.csv", "MSE", args.exclude_test01)
    metrics["chi2_reduced"] = metrics["chi2_reduced"].astype(float)
    signal = clean_spectrum_signal()
    metrics["MSE_signal_norm_percent"] = 100.0 * metrics["MSE"] / signal**2
    metrics["RMSE_signal_percent"] = 100.0 * np.sqrt(metrics["MSE"]) / signal

    tag = "test02_05" if args.exclude_test01 else "test01_05"
    marker_modes = ("strategy", "branch_strategy", "none") if args.marker_mode == "all" else (args.marker_mode,)
    error_metrics = (
        ("mse_signal_percent", "mse_signal_pct"),
        ("rmse_signal_percent", "rmse_signal_pct"),
    ) if args.error_metric == "all" else ((args.error_metric, "mse_signal_pct" if args.error_metric == "mse_signal_percent" else "rmse_signal_pct"),)
    outputs = [
        plot_comparison(
            metrics,
            times,
            f"aggregate_retrieval_metrics_1sigma_{tag}_{marker_mode}_{error_tag}_{args.version_tag}.png",
            args.exclude_test01,
            marker_mode,
            error_metric,
            args.linear_scales,
        )
        for marker_mode in marker_modes
        for error_metric, error_tag in error_metrics
    ]
    for path in outputs:
        print(path)
    print(f"Clean-spectrum signal = {signal:.6e}; MSE normalized by signal^2 = {signal**2:.6e}")


if __name__ == "__main__":
    main()
