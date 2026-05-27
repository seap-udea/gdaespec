#!/usr/bin/env python3
"""Shared helpers for the 5-observation, 10-transit retrieval campaign.

This module is the contract for the campaign workflow: it defines the physical
cases, output folder layout, CSV schemas, and small utilities shared by the
observation, retrieval, metric, and plotting scripts.
"""

from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Scientific constants and file locations used by every campaign script.
N_TRANSITS = 10
PLANET_NAME = "Trappist-1e"
INSTRUMENTS = ["JWST_NIRSpec_PRISM"]

RETRIEVAL_TESTS_DIR = Path(__file__).resolve().parent
EARTH_DIR = RETRIEVAL_TESTS_DIR.parent
CAMPAIGN_DIR = RETRIEVAL_TESTS_DIR / "campaign_5obs"
CLEAN_SPECTRUM_PATH = RETRIEVAL_TESTS_DIR / "pandexo_spec.txt"
GDAE_MODEL_PATH = EARTH_DIR / "Models" / "G-DAE.keras"
CONTAMINATION_DIR = EARTH_DIR / "stellar_contamination"

TEST_IDS = tuple(f"test_{idx:02d}" for idx in range(1, 6))
BRANCHES = ("phoenix", "sphinx")
STRATEGIES = ("gdae", "contam")

TIMES_COLUMNS = ["id", "branch", "f_spot", "f_fac", "strategy", "delta_time"]
METRICS_COLUMNS = ["id", "branch", "f_spot", "f_fac", "strategy", "MSE", "chi2_reduced"]
CSV_KEY_COLUMNS = ["id", "branch", "f_spot", "f_fac", "strategy"]


@dataclass(frozen=True)
class CaseConfig:
    """One physical contamination case in one branch."""

    branch: str
    f_spot: float
    f_fac: float
    contam_source: str

    @property
    def case_id(self) -> str:
        return f"{self.f_spot:.2f}spot-{self.f_fac:.2f}fac"

    @property
    def observation_stem(self) -> str:
        source_prefix = "sphinx_" if self.branch == "sphinx" else ""
        return (
            f"pandexo_output_{N_TRANSITS}transits_"
            f"{source_prefix}fspot{self.f_spot:.2f}_ffac{self.f_fac:.2f}"
        )

    @property
    def observation_file(self) -> str:
        return f"{self.observation_stem}.dat"

    @property
    def reconstruction_file(self) -> str:
        return f"{self.observation_stem}_recon.dat"

    def model_name(self, strategy: str) -> str:
        mode_label = strategy_to_mode_label(strategy)
        prefix = "sphinx_" if self.branch == "sphinx" else ""
        return (
            f"{prefix}{mode_label}_{N_TRANSITS}T_"
            f"{self.f_spot:.2f}spot-{self.f_fac:.2f}fac"
        )


# Retrieval cases. PHOENIX includes the clean reference case; SPHINX is used
# only for the model-mismatch contamination injections.
PHOENIX_CASES = (
    CaseConfig("phoenix", 0.00, 0.00, "clean"),
    CaseConfig("phoenix", 0.01, 0.08, "original"),
    CaseConfig("phoenix", 0.08, 0.54, "original"),
    CaseConfig("phoenix", 0.26, 0.70, "original"),
)

SPHINX_CASES = (
    CaseConfig("sphinx", 0.01, 0.08, "sphinx"),
    CaseConfig("sphinx", 0.08, 0.54, "sphinx"),
    CaseConfig("sphinx", 0.26, 0.70, "sphinx"),
)


def normalize_test_id(test_id: str) -> str:
    """Accept '1' or 'test_01' and return the canonical test id."""
    value = str(test_id).strip().lower()
    if value.startswith("test_"):
        number = int(value.split("_", 1)[1])
    else:
        number = int(value)
    normalized = f"test_{number:02d}"
    if normalized not in TEST_IDS:
        raise ValueError(f"Unknown test id: {test_id!r}. Expected one of {TEST_IDS}.")
    return normalized


def normalize_branch(branch: str) -> str:
    value = str(branch).strip().lower()
    if value not in BRANCHES:
        raise ValueError(f"Unknown branch: {branch!r}. Expected one of {BRANCHES}.")
    return value


def normalize_strategy(strategy: str) -> str:
    value = str(strategy).strip().lower().replace("recon", "gdae")
    if value not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. Expected one of {STRATEGIES}.")
    return value


def strategy_to_mode_label(strategy: str) -> str:
    strategy = normalize_strategy(strategy)
    return "recon" if strategy == "gdae" else "contam"


def strategy_free_params(strategy: str) -> int:
    strategy = normalize_strategy(strategy)
    return 6 if strategy == "gdae" else 11


def iter_cases(branch: str | None = None) -> Iterable[CaseConfig]:
    """Yield configured cases for one branch or for the full campaign."""
    if branch is None:
        yield from PHOENIX_CASES
        yield from SPHINX_CASES
        return

    branch = normalize_branch(branch)
    yield from (PHOENIX_CASES if branch == "phoenix" else SPHINX_CASES)


def get_case(branch: str, f_spot: float, f_fac: float) -> CaseConfig:
    """Return the configured case matching a branch and coverage fractions."""
    branch = normalize_branch(branch)
    for case in iter_cases(branch):
        if abs(case.f_spot - float(f_spot)) < 1.0e-9 and abs(case.f_fac - float(f_fac)) < 1.0e-9:
            return case
    raise ValueError(f"No configured {branch} case for f_spot={f_spot}, f_fac={f_fac}.")


def branch_dir(test_id: str, branch: str) -> Path:
    return CAMPAIGN_DIR / normalize_test_id(test_id) / normalize_branch(branch)


def observations_dir(test_id: str, branch: str) -> Path:
    return branch_dir(test_id, branch) / "observations"


def figures_dir(test_id: str, branch: str) -> Path:
    return branch_dir(test_id, branch) / "figures"


def pandexo_inputs_dir(test_id: str, branch: str) -> Path:
    return branch_dir(test_id, branch) / "pandexo_inputs"


def times_csv_path(test_id: str | None = None, branch: str | None = None) -> Path:
    if test_id is None:
        return CAMPAIGN_DIR / "times.csv"
    if branch is None:
        raise ValueError("branch is required for per-branch times.csv")
    return branch_dir(test_id, branch) / "run_times.csv"


def metrics_csv_path(test_id: str | None = None, branch: str | None = None) -> Path:
    if test_id is None:
        return CAMPAIGN_DIR / "metrics.csv"
    if branch is None:
        raise ValueError("branch is required for per-branch metrics.csv")
    return branch_dir(test_id, branch) / "run_metrics.csv"


def ensure_campaign_layout() -> None:
    """Create the campaign folder tree without launching any retrieval."""
    (CAMPAIGN_DIR / "plots").mkdir(parents=True, exist_ok=True)
    for test_id in TEST_IDS:
        for branch in BRANCHES:
            root = branch_dir(test_id, branch)
            for name in ("observations", "pandexo_inputs", "POSEIDON_output", "figures"):
                (root / name).mkdir(parents=True, exist_ok=True)
            ensure_csv(metrics_csv_path(test_id, branch), METRICS_COLUMNS)
            ensure_csv(times_csv_path(test_id, branch), TIMES_COLUMNS)
    ensure_csv(CAMPAIGN_DIR / "metrics.csv", METRICS_COLUMNS)
    ensure_csv(CAMPAIGN_DIR / "times.csv", TIMES_COLUMNS)


def ensure_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=columns).writeheader()


def csv_float(value: float) -> str:
    return f"{float(value):.2f}"


def csv_metric(value: float) -> str:
    return f"{float(value):.10g}"


def upsert_csv_row(path: Path, columns: list[str], row: dict[str, object], key_columns: list[str]) -> None:
    """Insert or replace one row while keeping a deliberately small CSV schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    key = {col: str(row[col]) for col in key_columns}
    rows: list[dict[str, str]] = []

    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for existing in reader:
                if all(str(existing.get(col, "")) == key[col] for col in key_columns):
                    continue
                rows.append({col: existing.get(col, "") for col in columns})

    rows.append({col: str(row.get(col, "")) for col in columns})

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def record_time(test_id: str, branch: str, case: CaseConfig, strategy: str, delta_time: float) -> None:
    """Write a retrieval duration to both aggregate and per-branch CSV files."""
    row = {
        "id": normalize_test_id(test_id),
        "branch": normalize_branch(branch),
        "f_spot": csv_float(case.f_spot),
        "f_fac": csv_float(case.f_fac),
        "strategy": normalize_strategy(strategy),
        "delta_time": csv_metric(delta_time),
    }
    upsert_csv_row(times_csv_path(), TIMES_COLUMNS, row, CSV_KEY_COLUMNS)
    upsert_csv_row(times_csv_path(test_id, branch), TIMES_COLUMNS, row, CSV_KEY_COLUMNS)


def record_metric(
    test_id: str,
    branch: str,
    case: CaseConfig,
    strategy: str,
    mse: float,
    chi2_reduced: float,
) -> None:
    row = {
        "id": normalize_test_id(test_id),
        "branch": normalize_branch(branch),
        "f_spot": csv_float(case.f_spot),
        "f_fac": csv_float(case.f_fac),
        "strategy": normalize_strategy(strategy),
        "MSE": csv_metric(mse),
        "chi2_reduced": csv_metric(chi2_reduced),
    }
    upsert_csv_row(metrics_csv_path(), METRICS_COLUMNS, row, CSV_KEY_COLUMNS)
    upsert_csv_row(metrics_csv_path(test_id, branch), METRICS_COLUMNS, row, CSV_KEY_COLUMNS)


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """Copy one file and return True when the destination changed."""
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return False
    shutil.copy2(src, dst)
    return True


def configure_pysynphot_data() -> None:
    """Set PYSYN_CDBS to the local PHOENIX grid when the shell did not."""
    if os.getenv("PYSYN_CDBS"):
        return

    candidates = (
        Path("C:/Proyectos/Astro/Pandexo/pysynphot_data"),
        Path("/mnt/c/Proyectos/Astro/Pandexo/pysynphot_data"),
    )
    for candidate in candidates:
        if (candidate / "grid" / "phoenix" / "catalog.fits").is_file():
            os.environ["PYSYN_CDBS"] = str(candidate)
            return


def configure_poseidon_input_data() -> None:
    """Set POSEIDON_input_data when non-interactive shells skip .bashrc."""
    if os.getenv("POSEIDON_input_data"):
        return

    candidates = (
        Path("/mnt/d/Proyectos/IA_SpecAtm_Bio/Data/POSEIDON/inputs"),
        Path("D:/Proyectos/IA_SpecAtm_Bio/Data/POSEIDON/inputs"),
    )
    for candidate in candidates:
        if (candidate / "opacity").is_dir():
            os.environ["POSEIDON_input_data"] = str(candidate)
            return


def configure_poseidon_environment() -> None:
    """Configure local data paths needed by POSEIDON retrieval utilities."""
    configure_pysynphot_data()
    configure_poseidon_input_data()
