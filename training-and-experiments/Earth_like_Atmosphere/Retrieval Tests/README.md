# Retrieval Tests

This directory contains the POSEIDON retrieval workflow used to compare two
ways of handling stellar contamination in synthetic TRAPPIST-1e transmission
observations.

The active workflow is the five-observation campaign in `campaign_5obs/`.

## Scientific Comparison

Each campaign test uses a synthetic 10-transit observation generated from the
same clean atmospheric spectrum and one contamination case.

The noisy spectra are generated with PandExo in `campaign_observations.py`.
That script starts from `pandexo_spec.txt`, applies the selected
`epsilon(lambda)` contamination curve from `../stellar_contamination/`, and
writes the noisy observation files under `campaign_5obs/test_XX/<branch>/observations/`.
It also writes the matching G-DAE reconstructions as `*_recon.dat`.

The two retrieval strategies are:

1. `gdae`
   The observed spectrum is preprocessed with the trained G-DAE. POSEIDON then
   retrieves the atmospheric parameters from the reconstructed `*_recon.dat`
   spectrum.
2. `contam`
   POSEIDON retrieves atmospheric parameters and stellar-contamination
   parameters jointly from the raw contaminated observation.

The campaign includes two contamination branches:

- `phoenix`: contamination curves from the PHOENIX-based stellar model grid.
- `sphinx`: observations injected with SPHINX contamination curves while the
  retrieval still uses the PHOENIX stellar model prescription.

## Files

| Path | Purpose |
| --- | --- |
| `campaign_common.py` | Shared campaign configuration, paths, case definitions, and CSV helpers. |
| `campaign_setup.py` | Creates the campaign directory tree and empty CSV headers. |
| `campaign_observations.py` | Generates PandExo observations and G-DAE reconstructions. |
| `campaign_retrieval_mpi.py` | Runs one POSEIDON retrieval case with MPI. |
| `campaign_metrics.py` | Computes MSE and reduced chi-square for completed retrievals. |
| `campaign_plot_aggregates.py` | Plots aggregate timing and metric summaries from campaign CSV files. |
| `campaign_plot_parameters.py` | Plots retrieved atmospheric parameters from POSEIDON result files. |
| `campaign_plot_observations.py` | Plots raw observations and G-DAE reconstructions across tests. |
| `campaign_run_gdae_queue.py` | Runs the missing `gdae` campaign jobs. |
| `campaign_run_contam_queue.py` | Runs the missing `contam` campaign jobs. |
| `run_gdae_queue.sh`, `run_contam_queue.sh` | Local WSL queue helpers. Edit paths before using elsewhere. |
| `pandexo_spec.txt` | Clean spectrum used as the PandExo input baseline. |
| `campaign_5obs/` | Campaign CSV summaries and generated per-test products. |

## Campaign Layout

`campaign_5obs/` contains:

- `metrics.csv`: aggregate metric table with
  `id,branch,f_spot,f_fac,strategy,MSE,chi2_reduced`.
- `times.csv`: aggregate run-time table with
  `id,branch,f_spot,f_fac,strategy,delta_time`.
- `test_01/` to `test_05/`: generated observations, retrieval outputs, and
  figures for each synthetic observation.
- `plots/`: aggregate figures generated from the campaign outputs.

Large generated products inside each test folder are ignored by Git. The CSV
summaries are the compact campaign record.

## Inputs

The campaign expects these project files:

- `pandexo_spec.txt`
- `../Models/G-DAE.keras`
- `../stellar_contamination/`

The retrieval step also requires a working POSEIDON, PandExo/Pandeia, and MPI
environment with the corresponding opacity and stellar-model data available
locally.

## Typical Workflow

Run commands from this directory.

```bash
python campaign_setup.py
python campaign_observations.py --test-id test_02 --branch all
mpirun -n 12 python -u campaign_retrieval_mpi.py --test-id test_02 --branch phoenix --strategy gdae --f-spot 0.26 --f-fac 0.70
python campaign_metrics.py --test-id test_02 --branch phoenix --strategy gdae --f-spot 0.26 --f-fac 0.70
python campaign_plot_aggregates.py
```

For batch execution, use:

```bash
python campaign_run_gdae_queue.py --nproc 12 --keep-going
python campaign_run_contam_queue.py --nproc 12 --keep-going
```

The shell wrappers are optional. Prefer the Python queue commands when running
on another machine.
