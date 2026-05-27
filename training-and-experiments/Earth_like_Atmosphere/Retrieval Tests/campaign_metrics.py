#!/usr/bin/env python3
"""Compute the campaign's MSE and reduced chi-square metrics.

Metrics are evaluated against the clean input spectrum binned onto the
observation grid. For `contam` retrievals, the retrieved median spectrum is
first divided by the fitted stellar-contamination factor before comparison.
"""

from __future__ import annotations

import argparse
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from campaign_common import (
    CLEAN_SPECTRUM_PATH,
    INSTRUMENTS,
    PLANET_NAME,
    CaseConfig,
    branch_dir,
    configure_poseidon_environment,
    get_case,
    iter_cases,
    normalize_branch,
    normalize_strategy,
    normalize_test_id,
    observations_dir,
    record_metric,
    strategy_free_params,
)


@contextmanager
def pushd(path: Path):
    """Temporarily run POSEIDON file reads from a campaign run directory."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_clean_two_cols(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the clean reference spectrum sorted by wavelength."""
    arr = np.genfromtxt(path, comments="#", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] < 2:
        raise ValueError(f"Expected >= 2 columns in clean spectrum: {path}")
    wl = arr[:, 0].astype(float)
    depth = arr[:, 1].astype(float)
    order = np.argsort(wl)
    return wl[order], depth[order]


def bin_average_with_halfbins(
    wl_src: np.ndarray,
    y_src: np.ndarray,
    centers: np.ndarray,
    halfwidths: np.ndarray,
    nsamp: int = 256,
) -> np.ndarray:
    """Average the clean spectrum over each observed wavelength bin."""
    wl_src = np.asarray(wl_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    centers = np.asarray(centers, dtype=float)
    halfwidths = np.asarray(halfwidths, dtype=float)

    order = np.argsort(wl_src)
    wl_sorted = wl_src[order]
    y_sorted = y_src[order]

    out = np.empty_like(centers, dtype=float)
    for idx, (center, halfwidth) in enumerate(zip(centers, halfwidths)):
        left = center - halfwidth
        right = center + halfwidth
        sample_wl = np.linspace(left, right, nsamp)
        sample_y = np.interp(sample_wl, wl_sorted, y_sorted)
        out[idx] = np.trapz(sample_y, sample_wl) / (right - left)
    return out


def parse_retrieved_contamination_params(results_path: Path) -> dict[str, float]:
    """Parse median retrieved stellar-contamination values from POSEIDON results."""
    text = results_path.read_text(encoding="utf-8", errors="replace")
    one_sigma = re.search(
        r"1\s+\u03c3 constraints\s*\*+\s*(?P<body>.*?)\n\s*\*+\s*\n2\s+\u03c3 constraints",
        text,
        flags=re.DOTALL,
    )
    body = one_sigma.group("body") if one_sigma else text
    params: dict[str, float] = {}
    for name in ("f_spot", "f_fac", "T_spot", "T_fac", "T_phot"):
        match = re.search(rf"^{name}\s*=\s*([+-]?\d+(?:\.\d+)?)", body, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Could not parse {name} from {results_path}.")
        params[name] = float(match.group(1))
    return params


def contamination_corrected_spectrum(
    run_dir: Path,
    case: CaseConfig,
    model_name: str,
    wl_out: np.ndarray,
    spec_median: np.ndarray,
) -> np.ndarray:
    """Remove the fitted stellar-contamination factor from a retrieved spectrum."""
    configure_poseidon_environment()
    from POSEIDON.constants import R_Sun
    from POSEIDON.core import create_star
    from POSEIDON.stellar import stellar_contamination

    results_path = (
        run_dir
        / "POSEIDON_output"
        / PLANET_NAME
        / "retrievals"
        / "results"
        / f"{model_name}_results.txt"
    )
    params = parse_retrieved_contamination_params(results_path)

    star_retrieved = create_star(
        0.1192 * R_Sun,
        params["T_phot"],
        5.2396,
        0.00,
        stellar_grid="phoenix",
        stellar_contam="two_spots",
        f_spot=params["f_spot"],
        T_spot=params["T_spot"],
        f_fac=params["f_fac"],
        T_fac=params["T_fac"],
    )
    epsilon = np.asarray(stellar_contamination(star_retrieved, wl_out), dtype=float)
    return np.asarray(spec_median, dtype=float) / epsilon


def compute_metrics_for_case(
    test_id: str,
    branch: str,
    strategy: str,
    f_spot: float,
    f_fac: float,
    *,
    write: bool = True,
) -> dict[str, float]:
    """Compute MSE and reduced chi-square for one completed retrieval."""
    configure_poseidon_environment()
    from POSEIDON.core import load_data, wl_grid_constant_R
    from POSEIDON.instrument import bin_spectrum_to_data
    from POSEIDON.utility import read_data, read_retrieved_spectrum

    test_id = normalize_test_id(test_id)
    branch = normalize_branch(branch)
    strategy = normalize_strategy(strategy)
    case = get_case(branch, f_spot, f_fac)
    run_dir = branch_dir(test_id, branch)
    obs_dir = observations_dir(test_id, branch)
    observation_file = case.reconstruction_file if strategy == "gdae" else case.observation_file
    observation_path = obs_dir / observation_file
    if not observation_path.is_file():
        raise FileNotFoundError(observation_path)

    model_name = case.model_name(strategy)
    wl_model = wl_grid_constant_R(0.4, 6.0, 4000)

    with pushd(run_dir):
        wl_out, _s_low2, _s_low1, spec_median, _s_high1, _s_high2 = read_retrieved_spectrum(
            PLANET_NAME,
            model_name,
        )
        data = load_data(str(obs_dir), [observation_file], INSTRUMENTS, wl_model)

    wl_data, half_bin, _y_obs, err_obs = read_data(str(obs_dir), observation_file)
    wl_clean, y_clean = load_clean_two_cols(CLEAN_SPECTRUM_PATH)
    clean_binned = bin_average_with_halfbins(wl_clean, y_clean, wl_data, half_bin)

    if strategy == "contam":
        spec_for_metrics = contamination_corrected_spectrum(
            run_dir,
            case,
            model_name,
            np.asarray(wl_out, dtype=float),
            np.asarray(spec_median, dtype=float),
        )
    else:
        spec_for_metrics = np.asarray(spec_median, dtype=float)

    model_binned = bin_spectrum_to_data(spec_for_metrics, wl_out, data)
    sig = np.asarray(err_obs, dtype=float)
    if np.any(sig <= 0.0) or np.any(~np.isfinite(sig)):
        raise ValueError(f"Invalid observational uncertainties in {observation_path}.")

    residuals = np.asarray(model_binned, dtype=float) - np.asarray(clean_binned, dtype=float)
    if not (residuals.shape == sig.shape):
        raise ValueError(f"Shape mismatch in metric inputs for {test_id}/{branch}/{model_name}.")

    n_points = int(residuals.size)
    dof = max(n_points - strategy_free_params(strategy), 0)
    mse = float(np.mean(residuals**2))
    chi2 = float(np.sum((residuals / sig) ** 2))
    chi2_reduced = float(chi2 / dof) if dof > 0 else np.nan

    if write:
        record_metric(test_id, branch, case, strategy, mse, chi2_reduced)

    return {"MSE": mse, "chi2_reduced": chi2_reduced}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--branch", choices=("phoenix", "sphinx", "all"), default="all")
    parser.add_argument("--strategy", choices=("gdae", "contam", "all"), default="all")
    parser.add_argument("--f-spot", type=float, default=None)
    parser.add_argument("--f-fac", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_id = normalize_test_id(args.test_id)
    branches = ("phoenix", "sphinx") if args.branch == "all" else (normalize_branch(args.branch),)
    strategies = ("gdae", "contam") if args.strategy == "all" else (normalize_strategy(args.strategy),)

    for branch in branches:
        if args.f_spot is not None or args.f_fac is not None:
            if args.f_spot is None or args.f_fac is None:
                raise ValueError("--f-spot and --f-fac must be supplied together.")
            cases = [get_case(branch, args.f_spot, args.f_fac)]
        else:
            cases = list(iter_cases(branch))
        for case in cases:
            for strategy in strategies:
                metrics = compute_metrics_for_case(
                    test_id,
                    branch,
                    strategy,
                    case.f_spot,
                    case.f_fac,
                )
                print(
                    f"{test_id} {branch} {case.case_id} {strategy}: "
                    f"MSE={metrics['MSE']:.6e}, "
                    f"chi2_reduced={metrics['chi2_reduced']:.6f}"
                )


if __name__ == "__main__":
    main()
