#!/usr/bin/env python3
"""Run one MPI POSEIDON retrieval for the 5-observation campaign.

This script is the execution entry point for both active strategies. `gdae`
uses the reconstructed observation and retrieves atmospheric parameters only;
`contam` uses the raw noisy observation and retrieves atmospheric plus
stellar-contamination parameters.

Example:
    mpirun -n 12 python -u campaign_retrieval_mpi.py --test-id test_02 --branch phoenix --strategy gdae --f-spot 0.26 --f-fac 0.70
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
from mpi4py import MPI

from campaign_common import (
    INSTRUMENTS,
    N_TRANSITS,
    PLANET_NAME,
    CaseConfig,
    branch_dir,
    configure_poseidon_environment,
    figures_dir,
    get_case,
    normalize_branch,
    normalize_strategy,
    normalize_test_id,
    observations_dir,
    record_time,
)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

warnings.filterwarnings("ignore", category=UserWarning, module=r"pysynphot")
configure_poseidon_environment()

comm = MPI.COMM_WORLD
_rank = comm.Get_rank()


def install_float64_shared_memory_override() -> None:
    """Force POSEIDON shared-memory arrays to float64 for retrieval stability."""
    import POSEIDON.utility as utility

    def shared_memory_array_force64(node_rank, node_comm, shape):
        dtype = np.float64
        itemsize = np.dtype(dtype).itemsize
        nbytes = int(np.prod(shape)) * itemsize if node_rank == 0 else 0
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
        buf, _ = win.Shared_query(MPI.PROC_NULL)
        arr = np.ndarray(buffer=buf, dtype=dtype, shape=shape)
        return arr, win

    utility.shared_memory_array = shared_memory_array_force64


install_float64_shared_memory_override()

from POSEIDON.constants import M_E, R_E, R_Sun
from POSEIDON.core import (
    create_planet,
    create_star,
    define_model,
    load_data,
    read_opacities,
    set_priors,
    wl_grid_constant_R,
)
from POSEIDON.corner import generate_cornerplot
from POSEIDON.retrieval import run_retrieval
from POSEIDON.utility import plot_collection, read_retrieved_spectrum
from POSEIDON.visuals import plot_spectra_retrieved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--branch", choices=("phoenix", "sphinx"), required=True)
    parser.add_argument("--strategy", choices=("gdae", "contam"), required=True)
    parser.add_argument("--f-spot", type=float, required=True)
    parser.add_argument("--f-fac", type=float, required=True)
    return parser.parse_args()


def build_star(strategy: str, case: CaseConfig):
    """Create the stellar model used by the selected retrieval strategy."""
    r_star = 0.1192 * R_Sun
    t_star = 2566.0
    metallicity_star = 0.00
    log_g_star = 5.2396

    if strategy == "contam":
        return create_star(
            r_star,
            t_star,
            log_g_star,
            metallicity_star,
            stellar_grid="phoenix",
            stellar_contam="two_spots",
            f_spot=case.f_spot,
            T_spot=0.86 * t_star,
            f_fac=case.f_fac,
            T_fac=t_star + 100.0,
        )

    return create_star(r_star, t_star, log_g_star, metallicity_star, stellar_grid="phoenix")


def define_retrieval_model(strategy: str, model_name: str):
    """Define the POSEIDON atmospheric model for a campaign retrieval."""
    bulk_species = ["N2"]
    param_species = ["H2O", "CH4", "CO2", "O3"]
    kwargs = {
        "PT_profile": "isotherm",
        "cloud_model": "cloud-free",
    }
    if strategy == "contam":
        kwargs["stellar_contam"] = "two_spots"
    return define_model(model_name, bulk_species, param_species, **kwargs)


def build_priors(strategy: str, planet, star, model, data, r_planet: float):
    """Build atmospheric priors and, for `contam`, stellar-contamination priors."""
    t_star = 2566.0
    prior_types = {
        "T": "uniform",
        "R_p_ref": "uniform",
        "log_H2O": "uniform",
        "log_CH4": "uniform",
        "log_CO2": "uniform",
        "log_O3": "uniform",
    }
    prior_ranges = {
        "T": [200.0, 400.0],
        "R_p_ref": [0.85 * r_planet, 1.15 * r_planet],
        "log_H2O": [-8.0, -1.0],
        "log_CH4": [-8.0, -1.0],
        "log_CO2": [-5.0, -1.0],
        "log_O3": [-8.0, -1.0],
    }
    if strategy == "contam":
        prior_types.update(
            {
                "f_spot": "uniform",
                "f_fac": "uniform",
                "T_phot": "uniform",
                "T_fac": "uniform",
                "T_spot": "uniform",
            }
        )
        prior_ranges.update(
            {
                "f_spot": [0.0, 0.26],
                "f_fac": [0.0, 0.70],
                "T_phot": [0.9 * t_star, 1.1 * t_star],
                "T_fac": [t_star, t_star + 150.0],
                "T_spot": [0.8 * t_star, 0.9 * t_star],
            }
        )
    return set_priors(planet, star, model, data, prior_types, prior_ranges)


def run_case(test_id: str, branch: str, strategy: str, case: CaseConfig) -> None:
    """Run POSEIDON, save figures, and record timing and metrics for one case."""
    from campaign_metrics import compute_metrics_for_case

    run_dir = branch_dir(test_id, branch)
    data_dir = observations_dir(test_id, branch)
    figure_dir = figures_dir(test_id, branch)
    figure_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    obs_file = case.reconstruction_file if strategy == "gdae" else case.observation_file
    obs_path = data_dir / obs_file
    if not obs_path.is_file():
        raise FileNotFoundError(
            f"Observation required by {strategy} is missing: {obs_path}. "
            "Run campaign_observations.py first so G-DAE has its *_recon.dat input."
        )

    model_name = case.model_name(strategy)
    if _rank == 0:
        print(
            ">> Campaign retrieval: "
            f"id={test_id}, branch={branch}, strategy={strategy}, "
            f"N_TRANSITS={N_TRANSITS}, f_spot={case.f_spot:.2f}, f_fac={case.f_fac:.2f}",
            flush=True,
        )
        print(f">> Observation: {obs_path}", flush=True)
        print(f">> Model: {model_name}", flush=True)

    r_planet = 0.917985 * R_E
    m_planet = 0.6356 * M_E
    planet = create_planet(PLANET_NAME, r_planet, mass=m_planet, T_eq=255.0)
    star = build_star(strategy, case)

    wl_min, wl_max, res = 0.4, 6.0, 4000
    wl = wl_grid_constant_R(wl_min, wl_max, res)
    data = load_data(str(data_dir), [obs_file], INSTRUMENTS, wl)
    model = define_retrieval_model(strategy, model_name)

    if _rank == 0:
        print(f"Free parameters: {model['param_names']}", flush=True)

    priors = build_priors(strategy, planet, star, model, data, r_planet)
    t_fine = np.arange(200.0, 400.0 + 10.0, 10.0)
    log_p_fine = np.arange(-2.0, 2.0 + 0.2, 0.2)
    opac = read_opacities(model, wl, "opacity_sampling", t_fine, log_p_fine)

    p_grid = np.logspace(np.log10(2.0), np.log10(1.0e-2), 100)
    p_ref = 1.0

    if _rank == 0:
        print(">> Initializing retrieval with MultiNest...", flush=True)

    t_ret_start = time.time()
    run_retrieval(
        planet,
        star,
        model,
        opac,
        data,
        priors,
        wl,
        p_grid,
        p_ref,
        R=res,
        spectrum_type="transmission",
        sampling_algorithm="MultiNest",
        N_live=500,
        verbose=(_rank == 0),
    )
    delta_minutes = (time.time() - t_ret_start) / 60.0

    if _rank == 0:
        print(f">> Retrieval completed. Delta time: {delta_minutes:.2f} min", flush=True)

    comm.Barrier()

    if _rank == 0:
        wl_ret, s_low2, s_low1, s_med, s_high1, s_high2 = read_retrieved_spectrum(
            PLANET_NAME,
            model_name,
        )
        spectra_med = plot_collection(s_med, wl_ret, collection=[])
        spectra_low1 = plot_collection(s_low1, wl_ret, collection=[])
        spectra_low2 = plot_collection(s_low2, wl_ret, collection=[])
        spectra_high1 = plot_collection(s_high1, wl_ret, collection=[])
        spectra_high2 = plot_collection(s_high2, wl_ret, collection=[])

        fig_spec = plot_spectra_retrieved(
            spectra_med,
            spectra_low2,
            spectra_low1,
            spectra_high1,
            spectra_high2,
            PLANET_NAME,
            data,
            R_to_bin=100,
            data_labels=INSTRUMENTS,
            data_colour_list=["lime"],
        )
        fig_spec.savefig(
            figure_dir / f"{PLANET_NAME}_{model_name}_retrieved_spectrum.png",
            dpi=180,
            bbox_inches="tight",
        )

        corner_out = generate_cornerplot(planet, model)
        fig_corner = None
        if hasattr(corner_out, "savefig"):
            fig_corner = corner_out
        elif isinstance(corner_out, tuple) and corner_out and hasattr(corner_out[0], "savefig"):
            fig_corner = corner_out[0]
        if fig_corner is not None:
            fig_corner.savefig(
                figure_dir / f"{PLANET_NAME}_{model_name}_corner.png",
                dpi=180,
                bbox_inches="tight",
            )

        record_time(test_id, branch, case, strategy, delta_minutes)
        metrics = compute_metrics_for_case(
            test_id,
            branch,
            strategy,
            case.f_spot,
            case.f_fac,
            write=True,
        )
        print(f">> Figures saved in: {figure_dir.resolve()}", flush=True)
        print(f">> Timing row written: delta_time={delta_minutes:.2f} min", flush=True)
        print(
            f">> Metrics row written: MSE={metrics['MSE']:.6e}, "
            f"chi2_reduced={metrics['chi2_reduced']:.6f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    test_id = normalize_test_id(args.test_id)
    branch = normalize_branch(args.branch)
    strategy = normalize_strategy(args.strategy)
    case = get_case(branch, args.f_spot, args.f_fac)
    run_case(test_id, branch, strategy, case)


if __name__ == "__main__":
    main()
