#!/usr/bin/env python3
"""Plot retrieved atmospheric parameters for the campaign comparison.

This script parses POSEIDON result text files, aggregates parameter medians and
uncertainties across tests, and writes publication-style comparison figures.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd

from campaign_common import CAMPAIGN_DIR, PLANET_NAME, TEST_IDS, iter_cases


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

# Scientific ordering for cases, branches, strategies, and parameters.
TARGET_CASES = [(0.00, 0.00), (0.01, 0.08), (0.08, 0.54), (0.26, 0.70)]
SOURCE_ORDER = ["phoenix", "sphinx"]
SOURCE_LABELS = {"phoenix": "PHOENIX", "sphinx": "SPHINX"}
SOURCE_LINESTYLES = {"phoenix": "-", "sphinx": "--"}
STRATEGY_ORDER = ["contam", "gdae"]
STRATEGY_LABELS = {"contam": "Cont + Chem", "gdae": "G-DAE + Chem"}
STRATEGY_COLORS = {"contam": "#264653", "gdae": "#E76F51"}
BRANCH_STRATEGY_MARKERS = {
    ("contam", "phoenix"): "s",
    ("contam", "sphinx"): "D",
    ("gdae", "phoenix"): "^",
    ("gdae", "sphinx"): "v",
}

PARAMS = ["log_CO2", "log_CH4", "log_O3", "log_H2O", "R_p_ref", "T"]
PARAM_TITLES = {
    "log_CO2": r"CO$_2$",
    "log_CH4": r"CH$_4$",
    "log_O3": r"O$_3$",
    "log_H2O": r"H$_2$O",
    "R_p_ref": r"$R_{p,\mathrm{ref}}$ ($R_J$)",
    "T": r"$T$ (K)",
}
EXPECTED = {
    "log_CO2": -3.0,
    "log_CH4": -8.0,
    "log_O3": -8.0,
    "log_H2O": -8.0,
    "R_p_ref": 0.0821,
    "T": 287.0,
}

# Shared plotting style constants.
LINE_WIDTH = 3.0
MARKER_SIZE = 7.2
ERROR_BAR_WIDTH = 1.35
PAIR_LINEWIDTH = 0.8
PAIR_ALPHA = 0.14
EDGE_COLOR = "#1f1f1f"
BASE_OFFSETS = {"contam": -0.16, "gdae": 0.16}
SOURCE_OFFSETS = {"phoenix": -0.045, "sphinx": 0.045}

NUM_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PARAM_LINE_RE = re.compile(
    rf"^\s*(?P<param>R_p_ref|T|log_H2O|log_CH4|log_CO2|log_O3)\s*=\s*"
    rf"(?P<val>{NUM_RE})\s*\(\+(?P<plus>{NUM_RE})\)\s*\(-(?P<minus>{NUM_RE})\)"
)


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("latin-1", errors="replace")


def one_sigma_block(text: str) -> str:
    match = re.search(
        r"1\s+\u03c3 constraints\s*\*+\s*(?P<body>.*?)\n\s*\*+\s*\n2\s+\u03c3 constraints",
        text,
        flags=re.DOTALL,
    )
    return match.group("body") if match else text


def parse_result(path: Path) -> dict[str, float]:
    """Parse median and one-sigma bounds from one POSEIDON result file."""
    row: dict[str, float] = {}
    for line in one_sigma_block(read_text(path)).splitlines():
        match = PARAM_LINE_RE.match(line)
        if not match:
            continue
        param = match.group("param")
        val = float(match.group("val"))
        plus = float(match.group("plus"))
        minus = float(match.group("minus"))
        row[param] = val
        row[f"{param}_sigma"] = 0.5 * (plus + minus)
        row[f"{param}_err_low"] = minus
        row[f"{param}_err_up"] = plus
    return row


def results_path(test_id: str, branch: str, model_name: str) -> Path:
    return (
        CAMPAIGN_DIR
        / test_id
        / branch
        / "POSEIDON_output"
        / PLANET_NAME
        / "retrievals"
        / "results"
        / f"{model_name}_results.txt"
    )


def collect_rows(exclude_test01: bool, only_test_id: str | None = None) -> pd.DataFrame:
    """Collect parsed parameter rows from available campaign result files."""
    rows = []
    if only_test_id is not None:
        test_ids = [only_test_id]
    else:
        test_ids = [test_id for test_id in TEST_IDS if not (exclude_test01 and test_id == "test_01")]
    for test_id in test_ids:
        for branch in SOURCE_ORDER:
            for case in iter_cases(branch):
                for strategy in STRATEGY_ORDER:
                    path = results_path(test_id, branch, case.model_name(strategy))
                    if not path.is_file():
                        continue
                    parsed = parse_result(path)
                    if not parsed:
                        continue
                    rows.append(
                        {
                            "id": test_id,
                            "branch": branch,
                            "strategy": strategy,
                            "f_spot": case.f_spot,
                            "f_fac": case.f_fac,
                            "case_key": (round(case.f_spot, 2), round(case.f_fac, 2)),
                            **parsed,
                        }
                    )
    return pd.DataFrame(rows)


def sample_std(values: pd.Series | np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0


def rms(values: pd.Series | np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0


def aggregate(df: pd.DataFrame, uncertainty_mode: str = "tests") -> pd.DataFrame:
    """Aggregate parameters using either test scatter or posterior intervals."""
    parts = []
    group_cols = ["branch", "strategy", "f_spot", "f_fac", "case_key"]
    for param in PARAMS:
        if param not in df.columns:
            continue
        sigma_col = f"{param}_sigma"
        low_col = f"{param}_err_low"
        up_col = f"{param}_err_up"
        if uncertainty_mode == "total" and low_col in df.columns and up_col in df.columns:
            rows = []
            for keys, grp in df.dropna(subset=[param]).groupby(group_cols):
                row = dict(zip(group_cols, keys))
                test_sigma = sample_std(grp[param])
                posterior_low = rms(grp[low_col])
                posterior_up = rms(grp[up_col])
                row.update(
                    mean=float(np.mean(grp[param])),
                    sigma_tests=test_sigma,
                    sigma_posterior_low=posterior_low,
                    sigma_posterior_up=posterior_up,
                    sigma_low=float(np.sqrt(test_sigma**2 + posterior_low**2)),
                    sigma_up=float(np.sqrt(test_sigma**2 + posterior_up**2)),
                    n_tests=int(grp[param].count()),
                )
                row["sigma"] = 0.5 * (row["sigma_low"] + row["sigma_up"])
                rows.append(row)
            sub = pd.DataFrame(rows)
        elif uncertainty_mode == "posterior" and low_col in df.columns and up_col in df.columns:
            sub = (
                df.dropna(subset=[param])
                .groupby(group_cols, as_index=False)
                .agg(
                    mean=(param, "mean"),
                    sigma_low=(low_col, "mean"),
                    sigma_up=(up_col, "mean"),
                    n_tests=(param, "count"),
                )
            )
            sub["sigma"] = 0.5 * (sub["sigma_low"] + sub["sigma_up"])
        elif uncertainty_mode == "posterior" and sigma_col in df.columns:
            sub = (
                df.dropna(subset=[param])
                .groupby(group_cols, as_index=False)
                .agg(
                    mean=(param, "mean"),
                    sigma=(sigma_col, "mean"),
                    n_tests=(param, "count"),
                )
            )
            sub["sigma_low"] = sub["sigma"]
            sub["sigma_up"] = sub["sigma"]
        else:
            sub = (
                df.dropna(subset=[param])
                .groupby(group_cols, as_index=False)
                .agg(
                    mean=(param, "mean"),
                    sigma=(param, sample_std),
                    n_tests=(param, "count"),
                )
            )
            sub["sigma_low"] = sub["sigma"]
            sub["sigma_up"] = sub["sigma"]
        sub["param"] = param
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    x_center_map = {case: i for i, case in enumerate(TARGET_CASES)}
    out["x_center"] = out["case_key"].map(x_center_map)
    out["xpos"] = out.apply(
        lambda row: float(row["x_center"] + BASE_OFFSETS[row["strategy"]] + SOURCE_OFFSETS[row["branch"]]),
        axis=1,
    )
    return out


def padded_limits(values: np.ndarray, reference: float | None = None, pad_frac: float = 0.16) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if reference is not None and np.isfinite(reference):
        finite = np.concatenate([finite, [float(reference)]])
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if np.isclose(low, high):
        span = max(abs(low) * 0.08, 0.5)
    else:
        span = high - low
    return low - pad_frac * span, high + pad_frac * span


def plot_param_panel(ax: plt.Axes, summary: pd.DataFrame, param: str, compact_y: bool) -> None:
    panel = summary[summary["param"] == param].copy()
    if panel.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    for boundary in range(len(TARGET_CASES) - 1):
        ax.axvline(boundary + 0.5, color="#d8d8d8", lw=0.8, ls=":", zorder=0)
    if param in EXPECTED:
        ax.axhline(EXPECTED[param], linestyle="--", linewidth=1.35, color="#555555", alpha=0.62, zorder=1)
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.45, alpha=0.20)
    ax.set_axisbelow(True)

    for strategy in STRATEGY_ORDER:
        sub_pair = panel[panel["strategy"] == strategy].copy()
        for _, grp in sub_pair.groupby("case_key"):
            grp = grp.sort_values("branch")
            if grp.shape[0] < 2:
                continue
            vals = grp["mean"].to_numpy(dtype=float)
            if np.isfinite(vals).all():
                ax.plot(grp["xpos"], vals, color="#8a8a8a", lw=PAIR_LINEWIDTH, alpha=PAIR_ALPHA, zorder=1.2)

    for branch in SOURCE_ORDER:
        for strategy in STRATEGY_ORDER:
            sub = panel[(panel["branch"] == branch) & (panel["strategy"] == strategy)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["x_center", "xpos"])
            x = sub["xpos"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            yerr = np.vstack(
                [
                    sub["sigma_low"].to_numpy(dtype=float),
                    sub["sigma_up"].to_numpy(dtype=float),
                ]
            )

            if branch == "sphinx" and (0.0, 0.0) not in set(sub["case_key"]):
                clean = panel[
                    (panel["branch"] == "phoenix")
                    & (panel["strategy"] == strategy)
                    & (panel["case_key"] == (0.0, 0.0))
                ]
                if not clean.empty:
                    clean_x = float(clean["x_center"].iloc[0] + BASE_OFFSETS[strategy] + SOURCE_OFFSETS["sphinx"])
                    clean_y = float(clean["mean"].iloc[0])
                    x_line = np.concatenate([[clean_x], x])
                    y_line = np.concatenate([[clean_y], y])
                else:
                    x_line, y_line = x, y
            else:
                x_line, y_line = x, y

            color = STRATEGY_COLORS[strategy]
            marker = BRANCH_STRATEGY_MARKERS[(strategy, branch)]
            ax.plot(
                x_line,
                y_line,
                linestyle=SOURCE_LINESTYLES[branch],
                linewidth=LINE_WIDTH,
                color=color,
                alpha=0.80,
                zorder=2,
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=marker,
                ms=MARKER_SIZE,
                capsize=4.0,
                capthick=ERROR_BAR_WIDTH * 0.8,
                elinewidth=ERROR_BAR_WIDTH,
                color=color,
                markerfacecolor=color,
                markeredgecolor=EDGE_COLOR,
                markeredgewidth=1.0,
                linestyle="None",
                alpha=0.92,
                zorder=3,
            )

    ax.text(
        0.015,
        0.93,
        PARAM_TITLES[param],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        zorder=10,
    )

    if param.startswith("log_"):
        if compact_y:
            low = panel["mean"] - panel["sigma_low"]
            high = panel["mean"] + panel["sigma_up"]
            ymin, ymax = padded_limits(
                np.concatenate([low.to_numpy(dtype=float), high.to_numpy(dtype=float)]),
                EXPECTED.get(param),
                pad_frac=0.18,
            )
            ax.set_ylim(max(-10.0, ymin), min(0.0, ymax))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel(r"log$_{10}$ VMR", fontsize=13)
    elif param == "R_p_ref":
        if compact_y:
            low = panel["mean"] - panel["sigma_low"]
            high = panel["mean"] + panel["sigma_up"]
            ax.set_ylim(*padded_limits(np.concatenate([low.to_numpy(dtype=float), high.to_numpy(dtype=float)]), EXPECTED.get(param)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_ylabel(r"$R_J$", fontsize=13)
    else:
        if compact_y:
            low = panel["mean"] - panel["sigma_low"]
            high = panel["mean"] + panel["sigma_up"]
            ax.set_ylim(*padded_limits(np.concatenate([low.to_numpy(dtype=float), high.to_numpy(dtype=float)]), EXPECTED.get(param)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_ylabel("K", fontsize=13)
    ax.tick_params(labelsize=11)


def plot_parameters(
    df: pd.DataFrame,
    exclude_test01: bool,
    only_test_id: str | None = None,
    compact_y: bool = False,
    uncertainty_mode: str | None = None,
) -> Path:
    uncertainty_mode = uncertainty_mode or ("posterior" if only_test_id is not None else "tests")
    summary = aggregate(df, uncertainty_mode=uncertainty_mode)
    if summary.empty:
        raise RuntimeError("No retrieved-parameter results were found.")

    fig, axes = plt.subplots(3, 2, figsize=(12.2, 9.2), sharex="col")
    fig.subplots_adjust(top=0.87, bottom=0.09, left=0.09, right=0.98, hspace=0.12, wspace=0.18)
    fig.suptitle("PHOENIX vs SPHINX Retrieved Parameters (10 Transits)", fontsize=21, y=0.99)

    for ax, param in zip(axes.ravel(), PARAMS):
        plot_param_panel(ax, summary, param, compact_y=compact_y)

    x_labels = [f"{spot:.2f}/{fac:.2f}" for spot, fac in TARGET_CASES]
    for ax in axes[-1, :]:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.set_xlabel(r"$f_{spot}/f_{fac}$", fontsize=14)
    for ax in axes[:-1, :].ravel():
        ax.tick_params(axis="x", labelbottom=False)

    handles = [
        Line2D(
            [0, 1],
            [0, 0],
            marker=BRANCH_STRATEGY_MARKERS[(strategy, branch)],
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
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncols=4,
        frameon=False,
        fontsize=11.2,
        columnspacing=1.25,
        handlelength=3.2,
        handletextpad=0.7,
    )

    if only_test_id is not None:
        tag = only_test_id
    else:
        tag = "test02_05" if exclude_test01 else "test01_05"
    out_dir = CAMPAIGN_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    compact_tag = "_compact" if compact_y else ""
    uncertainty_tag = "" if uncertainty_mode == "tests" and only_test_id is None else f"_{uncertainty_mode}"
    out_path = out_dir / f"aggregate_retrieved_parameters_1sigma_{tag}_branch_strategy{compact_tag}{uncertainty_tag}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-test01", action="store_true")
    parser.add_argument("--test-id", default=None, help="Plot a single test id, e.g. test_01.")
    parser.add_argument("--compact-y", action="store_true", help="Use adaptive y-limits per parameter.")
    parser.add_argument(
        "--uncertainty-mode",
        choices=("tests", "posterior", "total"),
        default=None,
        help="Error bars from test scatter, posterior constraints, or their quadrature sum.",
    )
    args = parser.parse_args()
    df = collect_rows(exclude_test01=args.exclude_test01, only_test_id=args.test_id)
    out = plot_parameters(
        df,
        exclude_test01=args.exclude_test01,
        only_test_id=args.test_id,
        compact_y=args.compact_y,
        uncertainty_mode=args.uncertainty_mode,
    )
    print(out)


if __name__ == "__main__":
    main()
