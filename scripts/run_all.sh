#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
RUN_ID="$(date -u +"%Y-%m-%dT%H%M%SZ")"
LOG="logs/run_${RUN_ID}.log"
COHORT_PROFILE="${COHORT_PROFILE:-older_adult_65plus}"
SITES="${SITES:-Overall}"
AGE_GROUPS="${AGE_GROUPS:-65+ yr}"
TUNE_PER_ORIGIN="${TUNE_PER_ORIGIN:-1}"

{
  echo "[run_all] run_id=${RUN_ID}"
  echo "[run_all] cohort_profile=${COHORT_PROFILE}"
  echo "[run_all] sites=${SITES}"
  echo "[run_all] age_groups=${AGE_GROUPS}"
  echo "[run_all] tune_per_origin=${TUNE_PER_ORIGIN}"
  if [[ "${SKIP_FETCH:-0}" == "1" ]]; then
    echo "[run_all] step=fetch_* (skipped; using latest existing snapshots)"
  else
    echo "[run_all] step=fetch_respnet"
    python3 scripts/fetch_socrata.py \
      --dataset-id kvib-3txy \
      --where "rate_type='Observed' AND surveillance_network in ('FluSurv-NET','RSV-NET')" \
      --order "week_ending_date ASC"

    echo "[run_all] step=fetch_nrevss_rsv"
    python3 scripts/fetch_socrata.py \
      --dataset-id 52kb-ccu2 \
      --order "repweekdate ASC"

    echo "[run_all] step=fetch_viral_positivity"
    python3 scripts/fetch_socrata.py \
      --dataset-id seuz-s2cv \
      --order "week_end ASC"

    echo "[run_all] step=fetch_nssp_ed_visits"
    python3 scripts/fetch_socrata.py \
      --dataset-id 7xva-uux8 \
      --where "geography='United States' AND demographics_type='Age Group' AND pathogen in ('Influenza','RSV')" \
      --order "week_end ASC"

    echo "[run_all] step=fetch_wili"
    python3 scripts/fetch_delphi_fluview_wili.py --regions nat --epiweeks 201001-202652
  fi

  echo "[run_all] step=build_tables"
  python3 scripts/build_analysis_tables.py --cohort-profile "${COHORT_PROFILE}"

  echo "[run_all] step=data_quality_audit"
  python3 scripts/report_data_quality.py

  echo "[run_all] step=evaluate"
  python3 scripts/evaluate_forecasts.py \
    --outdir results/benchmarks \
    --cohort-profile "${COHORT_PROFILE}" \
    --sites "${SITES}" \
    --age-groups "${AGE_GROUPS}" \
    --ridge-preprocess standardized_intercept \
    --nonnegative-handling truncate \
    --tune-per-origin "${TUNE_PER_ORIGIN}"

  echo "[run_all] step=q1_reporting_metadata"
  python3 scripts/write_reporting_metadata.py \
    --ridge-preprocess standardized_intercept \
    --nonnegative-handling truncate \
    --tune-per-origin "${TUNE_PER_ORIGIN}"

  echo "[run_all] step=nested_signal_increment_audit"
  python3 scripts/audit_nested_signal_increment.py

  echo "[run_all] step=mdes"
  python3 scripts/compute_paired_mdes.py --cohort-profile "${COHORT_PROFILE}"

  echo "[run_all] step=era_stratified_supplement"
  python3 scripts/era_stratified_analysis.py

  echo "[run_all] step=site_trend_validation"
  python3 scripts/site_trend_validation.py --cohort-profile "${COHORT_PROFILE}"

  echo "[run_all] step=pseudo_prospective_report"
  python3 scripts/pseudo_prospective_report.py --cohort-profile "${COHORT_PROFILE}"

  if [[ "${RUN_SHIFT_SUPP:-0}" == "1" ]]; then
    echo "[run_all] step=shift_supplement_evaluate"
    python3 scripts/evaluate_forecasts.py --outdir results/benchmarks_shift --min-train 52 --eval-last-n 9999

    echo "[run_all] step=shift_supplement_era_stratified"
    python3 scripts/era_stratified_analysis.py \
      --predictions-long results/benchmarks_shift/predictions_long.tsv \
      --out results/benchmarks_shift/era_paired_benchmark.tsv
  fi

  echo "[run_all] step=lag_analysis"
  python3 scripts/lag_analysis.py

  echo "[run_all] step=publication_plots"
  python3 scripts/plot_publication_figures.py --cohort-profile "${COHORT_PROFILE}"

  echo "[run_all] done"
} | tee "$LOG"

echo "Log: $LOG"
