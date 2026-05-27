#!/usr/bin/env python3
"""Plot raw campaign observations and G-DAE reconstructions by case.

The script reads the generated `.dat` files from `campaign_5obs` and produces
quick visual checks for the five noisy realizations in each branch.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from campaign_common import CAMPAIGN_DIR, TEST_IDS, iter_cases, observations_dir

TEST_COLORS = {
    "test_01": "#1f77b4",
    "test_02": "#d62728",
    "test_03": "#2ca02c",
    "test_04": "#ff7f0e",
    "test_05": "#9467bd",
}


def load_observation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load wavelength, transit depth, and uncertainty from a campaign `.dat` file."""
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected four columns in {path}, got {arr.shape}.")
    return arr[:, 0], arr[:, 2], arr[:, 3]


def case_file(case, reconstructed: bool) -> str:
    return case.reconstruction_file if reconstructed else case.observation_file


def plot_branch(branch: str, reconstructed: bool) -> Path:
    """Plot all configured cases for one branch and product type."""
    cases = list(iter_cases(branch))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    axes_flat = axes.ravel()

    for ax, case in zip(axes_flat, cases):
        values = []
        for test_id in TEST_IDS:
            path = observations_dir(test_id, branch) / case_file(case, reconstructed)
            wl, depth, err = load_observation(path)
            depth_ppm = depth * 1.0e6
            values.append(depth_ppm)
            ax.errorbar(
                wl,
                depth_ppm,
                yerr=err * 1.0e6,
                fmt="o",
                ms=2.4,
                lw=0.7,
                alpha=0.75,
                color=TEST_COLORS[test_id],
                label=test_id,
            )

        stack = np.vstack(values)
        median_spread = np.nanmedian(np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0))
        ax.set_title(
            f"f_spot={case.f_spot:.2f}, f_fac={case.f_fac:.2f} | "
            f"median spread={median_spread:.1f} ppm",
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        ax.set_ylabel("Transit depth (ppm)")

    for ax in axes_flat[len(cases):]:
        ax.axis("off")
        if branch == "sphinx":
            ax.text(
                0.5,
                0.5,
                "No clean SPHINX case",
                ha="center",
                va="center",
                fontsize=11,
                color="#555555",
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength (um)")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(TEST_IDS), frameon=False)
    kind = "gdae_reconstructions" if reconstructed else "raw_observations"
    title = f"{branch.upper()} campaign {kind.replace('_', ' ')}"
    fig.suptitle(title, y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = CAMPAIGN_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{branch}_{kind}_5tests.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    outputs = []
    for branch in ("phoenix", "sphinx"):
        outputs.append(plot_branch(branch, reconstructed=False))
        outputs.append(plot_branch(branch, reconstructed=True))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
