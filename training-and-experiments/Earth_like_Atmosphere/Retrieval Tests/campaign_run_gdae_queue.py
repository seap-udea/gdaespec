#!/usr/bin/env python3
"""Run missing G-DAE campaign retrievals sequentially.

This script is meant to be launched from the POSEIDON conda environment, ideally
inside WSL, from the `Retrieval Tests` directory. It checks for existing
POSEIDON result files before launching each MPI job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from campaign_common import CAMPAIGN_DIR, TEST_IDS, iter_cases, normalize_test_id


def result_exists(test_id: str, branch: str, model_name: str) -> bool:
    """Return True when the expected POSEIDON result file already exists."""
    return (
        CAMPAIGN_DIR
        / test_id
        / branch
        / "POSEIDON_output"
        / "Trappist-1e"
        / "retrievals"
        / "results"
        / f"{model_name}_results.txt"
    ).is_file()


def build_jobs(
    test_ids: list[str],
    include_test01: bool,
    branch_filter: str | None,
    f_spot_filter: float | None,
    f_fac_filter: float | None,
) -> list[tuple[str, str, float, float]]:
    """Build the list of missing campaign cases to run."""
    jobs: list[tuple[str, str, float, float]] = []
    for test_id in test_ids:
        if test_id == "test_01" and not include_test01:
            continue
        for branch in ("phoenix", "sphinx"):
            if branch_filter is not None and branch != branch_filter:
                continue
            for case in iter_cases(branch):
                if f_spot_filter is not None and abs(case.f_spot - f_spot_filter) > 1.0e-9:
                    continue
                if f_fac_filter is not None and abs(case.f_fac - f_fac_filter) > 1.0e-9:
                    continue
                if result_exists(test_id, branch, case.model_name("gdae")):
                    continue
                jobs.append((test_id, branch, case.f_spot, case.f_fac))
    return jobs


def run_job(job: tuple[str, str, float, float], nproc: int, dry_run: bool) -> int:
    """Launch one G-DAE retrieval job with mpirun."""
    test_id, branch, f_spot, f_fac = job
    log_dir = CAMPAIGN_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stamp}_{test_id}_{branch}_gdae_{f_spot:.2f}spot-{f_fac:.2f}fac.log"
    cmd = [
        "mpirun",
        "-n",
        str(nproc),
        sys.executable,
        "-u",
        "campaign_retrieval_mpi.py",
        "--test-id",
        test_id,
        "--branch",
        branch,
        "--strategy",
        "gdae",
        "--f-spot",
        f"{f_spot:.2f}",
        "--f-fac",
        f"{f_fac:.2f}",
    ]

    print(" ".join(cmd))
    print(f"  log: {log_path}")
    if dry_run:
        return 0

    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        stream.write(f"# Started: {datetime.now().isoformat(timespec='seconds')}\n")
        stream.write(f"# Command: {' '.join(cmd)}\n\n")
        stream.flush()
        completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"\n# Finished: {datetime.now().isoformat(timespec='seconds')}\n")
        stream.write(f"# Return code: {completed.returncode}\n")
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc", type=int, default=12)
    parser.add_argument("--test-id", action="append", default=None)
    parser.add_argument("--include-test01", action="store_true")
    parser.add_argument("--branch", choices=("phoenix", "sphinx"), default=None)
    parser.add_argument("--f-spot", type=float, default=None)
    parser.add_argument("--f-fac", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.f_spot is None) != (args.f_fac is None):
        raise ValueError("--f-spot and --f-fac must be supplied together.")
    test_ids = [normalize_test_id(value) for value in args.test_id] if args.test_id else list(TEST_IDS)
    jobs = build_jobs(
        test_ids,
        include_test01=args.include_test01,
        branch_filter=args.branch,
        f_spot_filter=args.f_spot,
        f_fac_filter=args.f_fac,
    )
    print(f"Queued G-DAE jobs: {len(jobs)}")
    for job in jobs:
        code = run_job(job, nproc=args.nproc, dry_run=args.dry_run)
        if code != 0:
            print(f"Job failed with return code {code}: {job}", file=sys.stderr)
            if not args.keep_going:
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
