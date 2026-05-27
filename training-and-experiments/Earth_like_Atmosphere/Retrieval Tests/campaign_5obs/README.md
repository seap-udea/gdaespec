# campaign_5obs

This folder stores the five-observation retrieval campaign for the Earth-like
TRAPPIST-1e analogue.

Each `test_XX` folder is one independent 10-transit synthetic observation. For
each test, the campaign compares:

- `gdae`: retrieval on the G-DAE reconstructed spectrum.
- `contam`: retrieval on the raw contaminated spectrum with stellar
  contamination fitted inside POSEIDON.

The compact campaign outputs are:

- `times.csv`: `id,branch,f_spot,f_fac,strategy,delta_time`
- `metrics.csv`: `id,branch,f_spot,f_fac,strategy,MSE,chi2_reduced`

Generated observations, POSEIDON products, figures, logs, and plots are local
run products and are ignored by Git.

Typical commands are documented in the parent
[`README.md`](../README.md).
