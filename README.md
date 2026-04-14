# Older-Adult Influenza/RSV Hospital Burden Early Warning (Public Data)

Reproducible, audit-ready pipeline that forecasts **older-adult (65+) hospitalization burden** for influenza (FluSurv-NET) and RSV (RSV-NET) using **publicly available U.S. surveillance data**, and translates forecasts into **decision-relevant alerts** (utility, expected cost, lead time).

## What this repository contains
- Data snapshots (frozen inputs): `data/snapshots/…`
- Provenance log (queries + checksums): `results/dataset_retrieval_log.tsv`
- Harmonized analysis table: `results/analysis/analysis_table.tsv`
- Benchmark results (source of truth for claims): `results/benchmarks/…`
- Publication figures: `plots/publication/…`
- Secondary age-strata outputs (65–74 / 75–84 / 85+): `results_strata/…`

## What this repository intentionally does *not* contain
- Manuscript / cover letter / journal upload artifacts (kept local-only)
- Internal planning notes and other non-reproducibility scaffolding

## Quick start
Create a fresh environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Run end-to-end (fetches public data, then builds tables + benchmarks + figures):

```bash
./scripts/run_all.sh
```

Run secondary age-strata analysis (no ED proxying; uses the same public signals, stratifies outcomes):

```bash
SKIP_FETCH=1 ./scripts/run_secondary_strata.sh
```

Re-run using the latest existing snapshots (recommended for “paper freeze” reruns):

```bash
SKIP_FETCH=1 ./scripts/run_all.sh
```

## Data sources
All inputs are public and programmatically retrieved. Dataset IDs / endpoints and extraction rules are documented in `data/manifest.tsv`.

## Reproducibility (Google Colab)
- Notebook: `notebooks/colab_reproducibility.ipynb`

## License
MIT (see `LICENSE`).

## How to cite
See `CITATION.cff`.

## Archive (Zenodo)
- DOI: (pending; will be updated after the re-release)

## Public review bundle (GitHub release / Zenodo)
To build the sanitized public review bundle zip (suitable as a GitHub release asset and journal supplement), run:

```bash
python3 tools/build_review_bundle.py
```
