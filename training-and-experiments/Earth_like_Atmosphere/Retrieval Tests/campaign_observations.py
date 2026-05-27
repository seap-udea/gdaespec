#!/usr/bin/env python3
"""Generate PandExo observations and G-DAE reconstructions for the campaign.

The script starts from the clean spectrum in `pandexo_spec.txt`, applies the
configured stellar-contamination curve when needed, runs PandExo to produce a
noisy JWST/NIRSpec Prism observation, and then saves the matching G-DAE
reconstruction used by the `gdae` retrieval strategy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np

from campaign_common import (
    CAMPAIGN_DIR,
    CONTAMINATION_DIR,
    GDAE_MODEL_PATH,
    N_TRANSITS,
    CLEAN_SPECTRUM_PATH,
    CaseConfig,
    branch_dir,
    get_case,
    iter_cases,
    normalize_branch,
    normalize_test_id,
    observations_dir,
    pandexo_inputs_dir,
)

CONTAMINATION_PATTERN = re.compile(r"fspot(?P<f_spot>[0-9.]+)_ffac(?P<f_fac>[0-9.]+)\.txt$")
TRIM_IDX = 18  # Drop the short-wavelength bins PandExo returns outside the useful range.


def load_two_column_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a wavelength/depth or wavelength/epsilon table sorted by wavelength."""
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}, got {arr.shape}.")
    wl = arr[:, 0].astype(np.float64)
    depth = arr[:, 1].astype(np.float64)
    order = np.argsort(wl)
    return wl[order], depth[order]


def load_epsilon_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wl, epsilon = load_two_column_spectrum(path)
    return wl, epsilon


def build_contamination_index() -> dict[tuple[str, float, float], Path]:
    """Index PHOENIX and SPHINX epsilon curves by source and coverage fractions."""
    index: dict[tuple[str, float, float], Path] = {}
    for path in sorted(CONTAMINATION_DIR.glob("*TRAPPIST-1_contam_fspot*.txt")):
        match = CONTAMINATION_PATTERN.search(path.name)
        if not match:
            continue
        source = "sphinx" if path.name.startswith("sphinx_") else "original"
        f_spot = float(match.group("f_spot"))
        f_fac = float(match.group("f_fac"))
        index[(source, f_spot, f_fac)] = path
    return index


def find_contamination_curve(case: CaseConfig) -> Path:
    if case.contam_source == "clean":
        raise ValueError("Clean cases do not use a contamination curve.")
    index = build_contamination_index()
    key = (case.contam_source, case.f_spot, case.f_fac)
    if key not in index:
        raise KeyError(
            "No contamination curve found for "
            f"source={case.contam_source}, f_spot={case.f_spot}, f_fac={case.f_fac}."
        )
    return index[key]


def save_contaminated_input_spectrum(case: CaseConfig, test_id: str) -> Path:
    """Create the two-column spectrum PandExo will use for one campaign case."""
    output_dir = pandexo_inputs_dir(test_id, case.branch)
    output_dir.mkdir(parents=True, exist_ok=True)
    wl_clean, depth_clean = load_two_column_spectrum(CLEAN_SPECTRUM_PATH)

    if case.contam_source == "clean":
        depth_out = depth_clean
    else:
        wl_eps, epsilon = load_epsilon_curve(find_contamination_curve(case))
        epsilon_interp = np.interp(
            wl_clean,
            wl_eps,
            epsilon,
            left=epsilon[0],
            right=epsilon[-1],
        )
        depth_out = depth_clean * epsilon_interp

    out_path = output_dir / f"{case.observation_stem}_input.txt"
    np.savetxt(out_path, np.column_stack((wl_clean, depth_out)), fmt="%.10e")
    return out_path


def save_pandexo_spectrum_to_dat(input_spec_path: Path, output_dat_path: Path) -> None:
    """Run PandExo once and save a POSEIDON-compatible four-column observation.

    Output columns are wavelength, half-bin width, noisy transit depth, and
    transit-depth uncertainty.
    """
    import pandexo.engine.justdoit as jdi

    exo_dict = jdi.load_exo_dict()
    exo_dict["observation"].update(
        {
            "sat_level": 80,
            "sat_unit": "%",
            "baseline_unit": "total",
            "baseline": 0.9535 * 3 * 60 * 60,
            "noise_floor": 0,
            "noccultations": N_TRANSITS,
        }
    )
    exo_dict["star"].update(
        {
            "type": "phoenix",
            "mag": 11.354,
            "ref_wave": 1.25,
            "temp": 2566,
            "metal": 0.0,
            "logg": 5.2396,
        }
    )
    exo_dict["planet"].update(
        {
            "type": "user",
            "w_unit": "um",
            "f_unit": "rp^2/r*^2",
            "transit_duration": 0.9535 * 60 * 60,
            "td_unit": "s",
            "exopath": str(input_spec_path),
        }
    )

    inst_dict = jdi.load_mode_dict("NIRSpec Prism")
    inst_dict["configuration"]["detector"].update({"subarray": "sub512", "ngroup": 6})

    results = jdi.run_pandexo(exo_dict, inst_dict)
    final_spec = results["FinalSpectrum"]

    waves_trim = final_spec["wave"][TRIM_IDX:]
    spec_rand_trim = final_spec["spectrum_w_rand"][TRIM_IDX:]
    err_trim = final_spec["error_w_floor"][TRIM_IDX:]
    wave_err = np.gradient(waves_trim) / 2.0

    output_dat_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_dat_path,
        np.column_stack((waves_trim, wave_err, spec_rand_trim, err_trim)),
        fmt="%.6e",
    )


def bin_average_with_halfbins(
    wl_src: np.ndarray,
    y_src: np.ndarray,
    centers: np.ndarray,
    halfwidths: np.ndarray,
    nsamp: int = 256,
) -> np.ndarray:
    """Average a high-resolution spectrum over observation bins."""
    wl_src = np.asarray(wl_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    halfwidths = np.asarray(halfwidths, dtype=np.float64)

    order = np.argsort(wl_src)
    wl_sorted = wl_src[order]
    y_sorted = y_src[order]

    out = np.empty_like(centers, dtype=np.float64)
    for idx, (center, halfwidth) in enumerate(zip(centers, halfwidths)):
        left = center - halfwidth
        right = center + halfwidth
        sample_wl = np.linspace(left, right, nsamp)
        sample_flux = np.interp(sample_wl, wl_sorted, y_sorted)
        out[idx] = np.trapz(sample_flux, sample_wl) / (right - left)
    return out.astype(np.float32)


def normalize_min_max_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    value_min = float(values.min())
    value_max = float(values.max())
    value_range = value_max - value_min
    if value_range == 0.0:
        return np.zeros_like(values)
    return (values - value_min) / value_range


def inverse_min_max_1d(values_norm: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values_norm = np.asarray(values_norm, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    return values_norm * (float(reference.max()) - float(reference.min())) + float(reference.min())


def maybe_flip_observation_vectors(
    wl: np.ndarray,
    d_wl: np.ndarray,
    y_obs: np.ndarray,
    y_err: np.ndarray,
    y_clean_binned: np.ndarray,
    *,
    force_flip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not force_flip:
        return wl, d_wl, y_obs, y_err, y_clean_binned
    return (
        wl[::-1].copy(),
        d_wl[::-1].copy(),
        y_obs[::-1].copy(),
        y_err[::-1].copy(),
        y_clean_binned[::-1].copy(),
    )


def load_autoencoder(model_path: Path = GDAE_MODEL_PATH) -> Any:
    from tensorflow import keras

    return keras.models.load_model(model_path)


def reconstruct_observation_file(
    observation_path: Path,
    output_dir: Path,
    model_path: Path = GDAE_MODEL_PATH,
    *,
    force_flip: bool = True,
) -> Path:
    """Use the trained G-DAE to reconstruct one noisy observation file."""
    observation = np.loadtxt(observation_path)
    if observation.ndim != 2 or observation.shape[1] < 4:
        raise ValueError(f"Expected four columns in {observation_path}, got {observation.shape}.")

    wl = observation[:, 0].astype(np.float32)
    d_wl = observation[:, 1].astype(np.float32)
    y_obs = observation[:, 2].astype(np.float32)
    y_err = observation[:, 3].astype(np.float32)

    wl_clean, y_clean = load_two_column_spectrum(CLEAN_SPECTRUM_PATH)
    y_clean_binned = bin_average_with_halfbins(wl_clean, y_clean, wl, d_wl)
    wl, d_wl, y_obs, y_err, y_clean_binned = maybe_flip_observation_vectors(
        wl,
        d_wl,
        y_obs,
        y_err,
        y_clean_binned,
        force_flip=force_flip,
    )

    model = load_autoencoder(model_path)
    x_norm = normalize_min_max_1d(y_obs).reshape(1, -1).astype(np.float32)
    y_recon_norm = model.predict(x_norm, verbose=0)[0].astype(np.float32)
    y_recon = inverse_min_max_1d(y_recon_norm, y_clean_binned)

    recon_path = output_dir / f"{observation_path.stem}_recon.dat"
    np.savetxt(recon_path, np.column_stack((wl, d_wl, y_recon, y_err)), fmt="%.10e")
    return recon_path


def export_case_observation(
    case: CaseConfig,
    test_id: str,
    *,
    overwrite: bool = False,
    reconstruct_only: bool = False,
) -> tuple[Path, Path]:
    """Generate the noisy observation and reconstruction for one configured case."""
    obs_dir = observations_dir(test_id, case.branch)
    obs_dir.mkdir(parents=True, exist_ok=True)
    observation_path = obs_dir / case.observation_file
    recon_path = obs_dir / case.reconstruction_file

    if observation_path.exists() and recon_path.exists() and not overwrite:
        print(f"Skipping existing observation and reconstruction: {test_id}/{case.branch}/{case.case_id}")
        return observation_path, recon_path

    if not reconstruct_only:
        input_path = save_contaminated_input_spectrum(case, test_id)
        save_pandexo_spectrum_to_dat(input_path, observation_path)
        print(f"Saved observation     : {observation_path}")
    elif not observation_path.exists():
        raise FileNotFoundError(f"Cannot reconstruct missing observation: {observation_path}")

    recon_path = reconstruct_observation_file(observation_path, obs_dir)
    print(f"Saved reconstruction : {recon_path}")
    return observation_path, recon_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-id", required=True, help="Campaign test id, e.g. test_02.")
    parser.add_argument("--branch", choices=("phoenix", "sphinx", "all"), default="all")
    parser.add_argument("--f-spot", type=float, default=None)
    parser.add_argument("--f-fac", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reconstruct-only",
        action="store_true",
        help="Do not run PandExo; rebuild *_recon.dat from existing observations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_id = normalize_test_id(args.test_id)
    (CAMPAIGN_DIR / "plots").mkdir(parents=True, exist_ok=True)

    if args.branch == "all":
        branches = ("phoenix", "sphinx")
    else:
        branches = (normalize_branch(args.branch),)

    selected_cases: list[CaseConfig] = []
    for branch in branches:
        if args.f_spot is not None or args.f_fac is not None:
            if args.f_spot is None or args.f_fac is None:
                raise ValueError("--f-spot and --f-fac must be supplied together.")
            selected_cases.append(get_case(branch, args.f_spot, args.f_fac))
        else:
            selected_cases.extend(iter_cases(branch))

    for case in selected_cases:
        branch_dir(test_id, case.branch).mkdir(parents=True, exist_ok=True)
        export_case_observation(
            case,
            test_id,
            overwrite=args.overwrite,
            reconstruct_only=args.reconstruct_only,
        )


if __name__ == "__main__":
    main()
