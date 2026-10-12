# Older-adult respiratory hospitalization forecasts

Code and frozen public surveillance inputs for comparing the predictive contributions of emergency-department visits, laboratory positivity and outpatient influenza-like illness to influenza and RSV hospitalization rates in adults aged 65 years or older.

The last observed hospitalization week is July 4, 2026. The evaluation is retrospective: the inputs are retrieved surveillance snapshots, not reconstructed historical releases. Laboratory positivity and weighted influenza-like illness include all ages; hospitalization outcomes and emergency-department indicators concern adults aged 65 years or older.

## Reproduce figures and tables

Use Python 3.11 or newer in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_all.py
```

This regenerates Figures 1–3 and S1–S2, Tables 1–2 and S1–S12, the source-publication table, and the 14-sheet supplementary workbook from the supplied numerical outputs.

To recompute all forecasts and paired uncertainty estimates from the frozen weekly inputs before generating the figures and tables:

```sh
.venv/bin/python run_all.py --refit
```

The full computation fits models separately at one- to four-week leads, compares errors on common forecast weeks, controls historical observations in additional comparisons, and evaluates fixed alternative seasonal curves. It uses chronological penalty selection and calendar-segmented block resampling. The two terminal-observation settings are both retained. No network access or credentials are required.

## Files

- `data/weekly_analysis.tsv`: aligned weekly outcomes and predictors used for estimation.
- `data/surveillance_sources.tsv`: observed source histories used to display coverage.
- `data/sources.json`: source identifiers, retrieval period and input definitions.
- `data/table_definitions.json`: source/model definitions, table notes and abbreviations.
- `scripts/`: numerical estimation, comparisons, plotting, table export and workbook generation.
- `results/`: forecasts, paired errors, intervals and descriptive calibration estimates.
- `figures/`: five figures in PDF/PNG and their underlying plotted values.
- `tables/`: all study tables in TSV and the supplementary workbook.

Table 1 describes the sources and Table S2 defines the models. Other numbered tables are generated from the numerical outputs; source-publication information is metadata. Positive MAE improvement favors the model with added indicators. Training-history comparisons and alternative seasonal curves answer distinct questions and are identified in the tables.

The frozen inputs and numerical outputs define the study version. Minor floating-point differences can occur across numerical libraries; figure typography can vary if Arial or Helvetica is unavailable. The previous public version remains available in the repository history and its archived release.

## Cite this version

Version v0.5.0 is archived at Zenodo: [10.5281/zenodo.23319750](https://doi.org/10.5281/zenodo.23319750). The corresponding [GitHub release](https://github.com/Zhenghongwei11/flu-rsv-older-adult-warning/releases/tag/v0.5.0) preserves the frozen inputs, analyses and figure/table generators.
