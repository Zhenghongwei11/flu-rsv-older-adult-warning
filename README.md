# Public Surveillance Forecasting of Older-Adult Influenza/RSV Hospitalizations

This repository contains the public data, code, and frozen analysis outputs needed to reproduce the figures and tables for a study of short-term forecasts of U.S. older-adult influenza and RSV hospitalization rates. The analysis evaluates whether public surveillance signals add predictive value beyond autoregressive and seasonal structure.

## Repository contents
- Frozen public data snapshots: `data/snapshots/`
- Data-source manifest: `data/manifest.tsv`
- Retrieval log with checksums: `results/dataset_retrieval_log.tsv`
- Harmonized analysis tables: `results/analysis/`
- Forecast benchmarks and alert summaries: `results/benchmarks/`
- Figure source tables: `results/figures/`
- Publication figures: `plots/publication/`
- Secondary age-strata tables: `results_strata/`

## Not included
Manuscript files, cover letters, journal upload artifacts, and local planning notes are not part of this public reproducibility repository.

## Quick start
Create a fresh environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Reproduce the current figures and tables from the frozen snapshots included in this repository:

```bash
SKIP_FETCH=1 ./scripts/run_all.sh
```

By default, this command reproduces the primary national older-adult analysis (`SITES=Overall`, `AGE_GROUPS="65+ yr"`) using standardized ridge models with an explicit intercept, nonnegative forecast truncation, and forecast-origin-specific tuning.

Run secondary age-strata analyses:

```bash
SKIP_FETCH=1 ./scripts/run_secondary_strata.sh
```

To refresh public data snapshots before rerunning the analysis:

```bash
./scripts/run_all.sh
```

## Data sources
All inputs are public and programmatically retrieved. Dataset IDs / endpoints and extraction rules are documented in `data/manifest.tsv`.

## Reproducibility (Google Colab)
- Notebook: `notebooks/colab_reproducibility.ipynb`

## License
MIT (see `LICENSE`).

## How to cite
See `CITATION.cff`.

## Archive
- Concept DOI: 10.5281/zenodo.19562183
- Latest version DOI: see the current GitHub release and Zenodo record.

## Reproducibility bundle
To build the public reproducibility bundle used as the release asset, run:

```bash
python3 tools/build_review_bundle.py
```
