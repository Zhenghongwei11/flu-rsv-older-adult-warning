#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


FEATURE_SETS = {
    "seasonal_naive": "same_mmw_r_week_previous_year",
    "ridge_univariate": "sin52,cos52,y_lag1,y_lag2",
    "ridge_with_ed": "sin52,cos52,y_lag1,y_lag2,ed_pathogen_pct_65p_lag1,ed_pathogen_pct_65p_lag2",
    "ridge_with_signals": "sin52,cos52,y_lag1,y_lag2,pathogen_or_syndromic_signal_lag1,pathogen_or_syndromic_signal_lag2",
    "ridge_with_signals_plus_ed": "ar_seasonal,public_virologic_or_syndromic_signals,ed_pathogen_pct_65p_lags",
    "ridge_with_signals_plus_flu_pos": "ar_seasonal,wili_lags,flu_positivity_lags",
    "ridge_with_signals_plus_flu_pos_plus_ed": "ar_seasonal,wili_lags,flu_positivity_lags,ed_influenza_pct_65p_lags",
}


def write_model_specification(
    out: Path,
    *,
    ridge_preprocess: str,
    nonnegative_handling: str,
    tune_per_origin: bool,
) -> None:
    if ridge_preprocess == "standardized_intercept":
        intercept_handling = "explicit_unpenalized_intercept"
        predictor_scaling = "predictors_standardized_within_each_training_window"
        penalty_handling = "slope_coefficients_penalized_by_single_lambda_intercept_unpenalized"
    elif ridge_preprocess == "legacy":
        intercept_handling = "none_legacy_ridge_origin_through_zero"
        predictor_scaling = "none_legacy_raw_feature_scale"
        penalty_handling = "all_coefficients_penalized_equally_by_single_lambda"
    else:
        raise ValueError(f"Unknown ridge_preprocess: {ridge_preprocess}")

    if nonnegative_handling == "truncate":
        nonnegative_desc = "negative_predictions_truncated_to_zero"
    elif nonnegative_handling == "none":
        nonnegative_desc = "none_unconstrained_real_valued_predictions"
    else:
        raise ValueError(f"Unknown nonnegative_handling: {nonnegative_handling}")

    tuning_scheme = (
        "lambda_retuned_within_each_forecast_origin_training_window"
        if tune_per_origin
        else "nested_prequential_lambda_tuning_before_outer_evaluation_window_when_enabled"
    )

    rows: list[dict[str, object]] = []
    for model, feature_set in FEATURE_SETS.items():
        if model == "seasonal_naive":
            rows.append(
                {
                    "model": model,
                    "intercept_handling": "not_applicable",
                    "predictor_scaling": "not_applicable",
                    "penalty_handling": "not_applicable",
                    "tuning_scheme": "none",
                    "tuning_grid_id": "",
                    "missing_signal_handling": "requires same MMWR week previous year; week 53 falls back to week 52 when needed",
                    "max_forward_fill_weeks": "",
                    "nonnegative_handling": "inherits nonnegative observed prior-year rate",
                    "feature_set_id": feature_set,
                }
            )
            continue
        rows.append(
                {
                    "model": model,
                    "intercept_handling": intercept_handling,
                    "predictor_scaling": predictor_scaling,
                    "penalty_handling": penalty_handling,
                    "tuning_scheme": tuning_scheme,
                    "tuning_grid_id": "0.01,0.1,1,10,100_default",
                    "missing_signal_handling": "forward_fill_within_contiguous_weekly_blocks_for_public_signals",
                    "max_forward_fill_weeks": 8,
                    "nonnegative_handling": nonnegative_desc,
                    "feature_set_id": feature_set,
                }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


def write_data_availability(out: Path) -> None:
    rows = [
        {
            "dataset_id": "cdc_kvib-3txy",
            "source_name": "CDC RESP-NET Rates and Clinical Data",
            "epidemiologic_week_end_rule": "Week Ending Date field in CDC public dataset",
            "public_release_timing": "CDC dashboard/dataset updated weekly",
            "revision_cycle": "Preliminary data; recent rates subject to reporting delays and previous rates updated as new data are received",
            "available_at_forecast_origin_rule": "Current project uses retrieved public snapshot, not historical as-of values for each origin",
            "notes": "Supports hospitalization outcomes by age group and state/site; latest weeks may include nowcast on CDC dashboard",
        },
        {
            "dataset_id": "cdc_7xva-uux8",
            "source_name": "NSSP Emergency Department Visits by Demographic Category",
            "epidemiologic_week_end_rule": "week_end is MMWR week ending date",
            "public_release_timing": "Updated once per week on Fridays according to dataset metadata",
            "revision_cycle": "Current public API exposes current snapshot; historical value vintages are not reconstructed by this project",
            "available_at_forecast_origin_rule": "Use only lags aligned to the forecast origin within the retrieved snapshot",
            "notes": "Public data are national only and expose older adults as 65+ years, not 65-74/75-84/85+ ED strata",
        },
        {
            "dataset_id": "cdc_seuz-s2cv",
            "source_name": "CDC viral respiratory pathogen test positivity",
            "epidemiologic_week_end_rule": "week_end field",
            "public_release_timing": "Public CDC open-data release timing; exact forecast-origin availability not reconstructed",
            "revision_cycle": "Current project uses retrieved public snapshot",
            "available_at_forecast_origin_rule": "Use lagged values only; no same-week future outcome information",
            "notes": "Used as public virologic signal layer where coverage is adequate",
        },
        {
            "dataset_id": "delphi_fluview_wili",
            "source_name": "Delphi Epidata FluView wILI national endpoint",
            "epidemiologic_week_end_rule": "MMWR epiweek from Delphi API",
            "public_release_timing": "API retrieval timestamp logged when snapshot is fetched",
            "revision_cycle": "Current project uses retrieved public snapshot",
            "available_at_forecast_origin_rule": "Use lagged national wILI values only",
            "notes": "Optional influenza syndromic signal; not a replacement for ED incremental-value analysis",
        },
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Write reviewer-facing reporting metadata tables.")
    ap.add_argument(
        "--ridge-preprocess",
        choices=["legacy", "standardized_intercept"],
        default="standardized_intercept",
        help="Model specification used for ridge outputs.",
    )
    ap.add_argument(
        "--nonnegative-handling",
        choices=["none", "truncate"],
        default="truncate",
        help="Nonnegative prediction handling used for ridge outputs.",
    )
    ap.add_argument(
        "--tune-per-origin",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether ridge lambda was retuned inside each forecast origin.",
    )
    args = ap.parse_args()
    write_model_specification(
        RESULTS / "benchmarks/model_specification.tsv",
        ridge_preprocess=str(args.ridge_preprocess),
        nonnegative_handling=str(args.nonnegative_handling),
        tune_per_origin=bool(int(args.tune_per_origin)),
    )
    write_data_availability(RESULTS / "analysis/data_availability_by_source.tsv")
    print("Wrote results/benchmarks/model_specification.tsv")
    print("Wrote results/analysis/data_availability_by_source.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
