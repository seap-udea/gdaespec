# Stellar Contamination Assets

This directory stores the stellar-contamination curves used in the Earth-like
TRAPPIST-1 workflow.

## Contents

- `TRAPPIST-1_contam_fspot*_ffac*.txt`
  PHOENIX-based contamination curves.
- `sphinx_TRAPPIST-1_contam_fspot*_ffac*.txt`
  SPHINX-based contamination curves for the same spot and facula coverage
  fractions.
- `generate_sphinx_trappist1_contamination.ipynb`
  Notebook used to generate the SPHINX curves.
- `sphinx_data/`
  Local SPHINX stellar grids used as input to generate `epsilon(lambda)`.

## Role in the Workflow

The retrieval campaign reads this folder through
[`../Retrieval Tests/campaign_observations.py`](../Retrieval%20Tests/campaign_observations.py).

The campaign uses:

- PHOENIX curves for the `phoenix` branch.
- SPHINX curves for the `sphinx` injection branch.

Both branches write their generated observations under
[`../Retrieval Tests/campaign_5obs`](../Retrieval%20Tests/campaign_5obs/).

The SPHINX curves were generated with
`generate_sphinx_trappist1_contamination.ipynb` from the local grids in
`sphinx_data/`, using the same spot and facula filling factors as the PHOENIX
curves.

## References

- Base Earth-like spectral dataset: [MultiREx-public - DZF-MLBiosignatureClassification](https://github.com/D4san/MultiREx-public/tree/main/examples/papers/DZF-MLBiosignatureClassification)
- Associated paper: [Monthly Notices of the Royal Astronomical Society (MNRAS), 2025](https://academic.oup.com/mnras/article/539/2/1528/8109635)
- SPHINX grid archive used as user-provided input: [Zenodo record 7416042](https://zenodo.org/records/7416042)
