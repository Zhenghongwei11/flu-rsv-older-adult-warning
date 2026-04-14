#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COHORT_PROFILE="${COHORT_PROFILE:-older_adult_strata}"
OUT_ROOT="${OUT_ROOT:-results_strata}"
PLOTS_OUT="${PLOTS_OUT:-plots/publication_strata}"

echo "[run_secondary_strata] cohort_profile=${COHORT_PROFILE}"
echo "[run_secondary_strata] out_root=${OUT_ROOT}"
echo "[run_secondary_strata] plots_out=${PLOTS_OUT}"

echo "[run_secondary_strata] step=build_tables"
python3 scripts/build_analysis_tables.py --cohort-profile "${COHORT_PROFILE}" --outdir "${OUT_ROOT}"

echo "[run_secondary_strata] step=data_quality_audit"
python3 scripts/report_data_quality.py --input "${OUT_ROOT}/analysis/analysis_table.tsv" --outdir "${OUT_ROOT}/analysis"

echo "[run_secondary_strata] step=evaluate"
python3 scripts/evaluate_forecasts.py \
  --cohort-profile "${COHORT_PROFILE}" \
  --input "${OUT_ROOT}/analysis/analysis_table.tsv" \
  --outdir "${OUT_ROOT}/benchmarks"

echo "[run_secondary_strata] step=era_stratified_supplement"
python3 scripts/era_stratified_analysis.py \
  --predictions-long "${OUT_ROOT}/benchmarks/predictions_long.tsv" \
  --out "${OUT_ROOT}/benchmarks/era_paired_benchmark.tsv"

echo "[run_secondary_strata] step=site_trend_validation"
python3 scripts/site_trend_validation.py \
  --cohort-profile "${COHORT_PROFILE}" \
  --analysis-table "${OUT_ROOT}/analysis/analysis_table.tsv" \
  --out "${OUT_ROOT}/benchmarks/site_trend_correlation.tsv"

echo "[run_secondary_strata] step=pseudo_prospective_report"
python3 scripts/pseudo_prospective_report.py \
  --cohort-profile "${COHORT_PROFILE}" \
  --predictions-long "${OUT_ROOT}/benchmarks/predictions_long.tsv" \
  --alert-utility "${OUT_ROOT}/benchmarks/alert_utility.tsv" \
  --out-summary "${OUT_ROOT}/benchmarks/pseudo_prospective_episode_summary.tsv" \
  --out-episodes "${OUT_ROOT}/benchmarks/pseudo_prospective_episode_log.tsv"

echo "[run_secondary_strata] step=lag_analysis"
python3 scripts/lag_analysis.py \
  --input "${OUT_ROOT}/analysis/analysis_table.tsv" \
  --out "${OUT_ROOT}/figures/lag_analysis.tsv"

echo "[run_secondary_strata] step=publication_plots"
python3 scripts/plot_publication_figures.py \
  --cohort-profile "${COHORT_PROFILE}" \
  --outdir "${PLOTS_OUT}" \
  --time-series-overview "${OUT_ROOT}/figures/time_series_overview.tsv" \
  --signal-coverage "${OUT_ROOT}/analysis/signal_coverage_report.tsv" \
  --lag-analysis "${OUT_ROOT}/figures/lag_analysis.tsv" \
  --paired-benchmark "${OUT_ROOT}/benchmarks/paired_benchmark.tsv" \
  --expected-cost "${OUT_ROOT}/benchmarks/expected_cost.tsv" \
  --alert-lead-time "${OUT_ROOT}/benchmarks/alert_lead_time.tsv"

echo "[run_secondary_strata] step=refresh_key_results"
python3 scripts/refresh_key_results.py \
  --cohort-profile "${COHORT_PROFILE}" \
  --paired-benchmark "${OUT_ROOT}/benchmarks/paired_benchmark.tsv" \
  --expected-cost "${OUT_ROOT}/benchmarks/expected_cost.tsv" \
  --alert-utility "${OUT_ROOT}/benchmarks/alert_utility.tsv" \
  --pseudo-prospective "${OUT_ROOT}/benchmarks/pseudo_prospective_episode_summary.tsv" \
  --missingness "${OUT_ROOT}/analysis/missingness_report.tsv" \
  --site-trend "${OUT_ROOT}/benchmarks/site_trend_correlation.tsv" \
  --out "results_strata/KEY_RESULTS_LOCK_STRATA.md"

echo "[run_secondary_strata] done"
