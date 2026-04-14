from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cohort_profiles import cohort_from_env, get_cohort

ROOT = Path(__file__).resolve().parents[1]


def _pick_one(df: pd.DataFrame, **equals) -> pd.Series:
    out = df
    for k, v in equals.items():
        out = out[out[k] == v]
    if len(out) != 1:
        keys = ", ".join(f"{k}={v!r}" for k, v in equals.items())
        raise ValueError(f"Expected 1 row for {keys}, got {len(out)}")
    return out.iloc[0]


def _pick_first_available(df: pd.DataFrame, method_candidates: list[str], **equals) -> pd.Series:
    last_err: Exception | None = None
    for m in method_candidates:
        try:
            return _pick_one(df, **{**equals, "method": m})
        except Exception as e:
            last_err = e
            continue
    keys = ", ".join(f"{k}={v!r}" for k, v in equals.items())
    raise ValueError(f"No available method match for {keys}; tried {method_candidates!r}. Last error: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh a key-results markdown summary from pipeline TSV outputs.")
    ap.add_argument(
        "--cohort-profile",
        default=cohort_from_env("older_adult_65plus"),
        help="Cohort profile id (default: env COHORT_PROFILE or 'older_adult_65plus').",
    )
    ap.add_argument(
        "--paired-benchmark",
        default=str(ROOT / "results/benchmarks/paired_benchmark.tsv"),
        help="Path to paired_benchmark.tsv",
    )
    ap.add_argument(
        "--expected-cost",
        default=str(ROOT / "results/benchmarks/expected_cost.tsv"),
        help="Path to expected_cost.tsv",
    )
    ap.add_argument(
        "--alert-utility",
        default=str(ROOT / "results/benchmarks/alert_utility.tsv"),
        help="Path to alert_utility.tsv",
    )
    ap.add_argument(
        "--pseudo-prospective",
        default=str(ROOT / "results/benchmarks/pseudo_prospective_episode_summary.tsv"),
        help="Path to pseudo_prospective_episode_summary.tsv",
    )
    ap.add_argument(
        "--missingness",
        default=str(ROOT / "results/analysis/missingness_report.tsv"),
        help="Path to missingness_report.tsv",
    )
    ap.add_argument(
        "--site-trend",
        default=str(ROOT / "results/benchmarks/site_trend_correlation.tsv"),
        help="Path to site_trend_correlation.tsv",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "results/KEY_RESULTS_LOCK.md"),
        help="Output markdown path.",
    )
    args = ap.parse_args()
    cohort = get_cohort(args.cohort_profile)
    age_exemplar = str(cohort.key_exemplar_age)

    def _rel_to_root(p: str) -> str:
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except Exception:
            return str(p)

    def _infer_results_root(paths: list[str]) -> str:
        roots: list[str] = []
        for p in paths:
            rel = _rel_to_root(p)
            parts = Path(rel).parts
            if "benchmarks" in parts:
                i = parts.index("benchmarks")
                roots.append(str(Path(*parts[:i])))
            elif "analysis" in parts:
                i = parts.index("analysis")
                roots.append(str(Path(*parts[:i])))
            else:
                roots.append(str(Path(rel).parent))
        roots = [r for r in roots if r and r != "."]
        if not roots:
            return "outputs"
        if all(r == roots[0] for r in roots):
            return roots[0] + "/" if not roots[0].endswith("/") else roots[0]
        return "outputs"

    results_root = _infer_results_root(
        [
            args.paired_benchmark,
            args.expected_cost,
            args.alert_utility,
            args.pseudo_prospective,
            args.missingness,
            args.site_trend,
        ]
    )

    src_paired = _rel_to_root(args.paired_benchmark)
    src_expected_cost = _rel_to_root(args.expected_cost)
    src_alert_utility = _rel_to_root(args.alert_utility)
    src_pseudo = _rel_to_root(args.pseudo_prospective)
    src_missingness = _rel_to_root(args.missingness)
    src_site_trend = _rel_to_root(args.site_trend)

    paired = pd.read_csv(args.paired_benchmark, sep="\t")
    expected_cost = pd.read_csv(args.expected_cost, sep="\t")
    alert_utility = pd.read_csv(args.alert_utility, sep="\t")
    pseudo = pd.read_csv(args.pseudo_prospective, sep="\t")
    missingness = pd.read_csv(args.missingness, sep="\t")
    site_trend = pd.read_csv(args.site_trend, sep="\t")

    # Primary reviewer-facing context: reporting-delay sensitivity exclude_last_weeks=4
    flu_pb = _pick_one(
        paired,
        exclude_last_weeks=4,
        surveillance_network="FluSurv-NET",
        age_group=age_exemplar,
        site="Overall",
        train_scope="within_site",
        method="ridge_with_signals",
        horizon_weeks=1,
        metric="MAE",
    )
    rsv_pb_univ = _pick_one(
        paired,
        exclude_last_weeks=4,
        surveillance_network="RSV-NET",
        age_group=age_exemplar,
        site="Overall",
        train_scope="within_site",
        method="ridge_univariate",
        horizon_weeks=1,
        metric="MAE",
    )
    # RSV exemplar method depends on cohort/signal availability.
    rsv_pb_exemplar = _pick_first_available(
        paired,
        method_candidates=["ridge_with_ed", "ridge_univariate"],
        exclude_last_weeks=4,
        surveillance_network="RSV-NET",
        age_group=age_exemplar,
        site="Overall",
        train_scope="within_site",
        horizon_weeks=1,
        metric="MAE",
    )

    # Alert utility (explicit operational trade-off; RSV exemplar)
    rsv_au = _pick_first_available(
        alert_utility,
        method_candidates=["ridge_with_ed", "ridge_univariate"],
        exclude_last_weeks=4,
        surveillance_network="RSV-NET",
        age_group=age_exemplar,
        site="Overall",
        horizon_weeks=1,
        threshold_rule="train_quantile_0.85",
    )

    def _expected_cost_sub(network: str, age_group: str) -> pd.DataFrame:
        sub = expected_cost[
            (expected_cost["exclude_last_weeks"] == 4)
            & (expected_cost["surveillance_network"] == network)
            & (expected_cost["age_group"] == age_group)
            & (expected_cost["site"] == "Overall")
            & (expected_cost["horizon_weeks"] == 1)
            & (expected_cost["cost_ratio_fn_to_fp"] == 2.0)
            & (expected_cost["method"] != "seasonal_naive")
        ].copy()
        if len(sub) == 0:
            raise ValueError(f"No expected_cost rows for {network} {age_group}")
        return sub

    def _expected_cost_summary(network: str, age_group: str, primary_method: str) -> dict[str, object]:
        sub = _expected_cost_sub(network, age_group)
        best = sub.loc[sub["expected_cost_ratio_vs_baseline"].idxmin()]

        fixed = _pick_one(
            expected_cost,
            exclude_last_weeks=4,
            surveillance_network=network,
            age_group=age_group,
            site="Overall",
            horizon_weeks=1,
            cost_ratio_fn_to_fp=2.0,
            method=primary_method,
            threshold_rule="train_quantile_0.85",
        )
        method_sub = sub[sub["method"] == primary_method]
        return {
            "best_row": best,
            "fixed_row": fixed,
            "method_min": float(method_sub["expected_cost_ratio_vs_baseline"].min()),
            "method_max": float(method_sub["expected_cost_ratio_vs_baseline"].max()),
            "grid_rows_h1_cost2": int(len(sub)),
        }

    flu_ec = _expected_cost_summary("FluSurv-NET", age_exemplar, primary_method="ridge_with_signals")
    rsv_ec = _expected_cost_summary("RSV-NET", age_exemplar, primary_method=str(rsv_au.method))

    # Total number of retrospective expected-cost configurations in the primary subset
    # (exclude_last_weeks=4, age=exemplar, site=Overall), across all horizons/cost ratios/thresholds/methods.
    ec_primary = expected_cost[
        (expected_cost["exclude_last_weeks"] == 4)
        & (expected_cost["age_group"] == age_exemplar)
        & (expected_cost["site"] == "Overall")
        & (expected_cost["method"] != "seasonal_naive")
    ][
        [
            "surveillance_network",
            "horizon_weeks",
            "method",
            "threshold_rule",
            "cost_ratio_fn_to_fp",
        ]
    ].drop_duplicates()
    ec_primary_n = int(len(ec_primary))

    # Pseudo-prospective episode-level PPV (RSV exemplar)
    rsv_pp_ridge = _pick_first_available(
        pseudo,
        method_candidates=["ridge_with_ed", "ridge_univariate"],
        exclude_last_weeks=4,
        surveillance_network="RSV-NET",
        age_group=age_exemplar,
        site="Overall",
        threshold_rule="train_quantile_0.85",
    )
    rsv_pp_base = _pick_one(
        pseudo,
        exclude_last_weeks=4,
        surveillance_network="RSV-NET",
        age_group=age_exemplar,
        site="Overall",
        method="seasonal_naive",
        threshold_rule="train_quantile_0.85",
    )

    def _ppv_note(row: pd.Series) -> str:
        val = row.get("episode_ppv_note", "")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return ""

    # Missingness (RSV lab positivity; disclosure exemplar)
    rsv_missing = _pick_one(
        missingness,
        surveillance_network="RSV-NET",
        age_group=age_exemplar,
        site="Overall",
        variable="rsv_positivity",
    )

    # Site trend concordance (median Spearman r with overlap>=100)
    def _median_site_r(network: str, age: str) -> float:
        sub = site_trend[
            (site_trend["surveillance_network"] == network)
            & (site_trend["reference_age_group"] == age)
            & (site_trend["n_overlap"] >= 100)
        ]
        if len(sub) == 0:
            raise ValueError(f"No site_trend rows for {network} {age}")
        return float(sub["spearman_r"].median())

    med_r_flu = _median_site_r("FluSurv-NET", age_exemplar)
    med_r_rsv = _median_site_r("RSV-NET", age_exemplar)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def f2(x: float) -> str:
        return f"{x:.2f}"

    def f3(x: float) -> str:
        return f"{x:.3f}"

    def f1(x: float) -> str:
        return f"{x:.1f}"

    rsv_lines: list[str] = []
    rsv_lines.append(
        f"- RSV-NET `{age_exemplar}` (`{str(rsv_pb_univ.method)}`): paired MAE improvement {f2(float(rsv_pb_univ.improvement_over_baseline))} (95% CI {f2(float(rsv_pb_univ.improvement_ci_lo))} to {f2(float(rsv_pb_univ.improvement_ci_hi))}); paired_n={int(rsv_pb_univ.paired_n)}."
    )
    if str(rsv_pb_exemplar.method) != str(rsv_pb_univ.method):
        rsv_lines.append(
            f"- RSV-NET `{age_exemplar}` (`{str(rsv_pb_exemplar.method)}`): paired MAE improvement {f2(float(rsv_pb_exemplar.improvement_over_baseline))} (95% CI {f2(float(rsv_pb_exemplar.improvement_ci_lo))} to {f2(float(rsv_pb_exemplar.improvement_ci_hi))}); paired_n={int(rsv_pb_exemplar.paired_n)}."
        )

    md = f"""# Key Results (locked; numbers must match `{results_root}`)

This file is an auto-refreshed writing aid: it summarizes a small set of **manuscript-facing** effect sizes supporting the main claims (C1–C3) using the current default pipeline outputs.

Do not edit numbers manually. Refresh via:

```bash
python3 scripts/refresh_key_results.py
```

## Primary reviewer-facing context
- We emphasize `exclude_last_weeks=4` (reporting-delay/backfill sensitivity) as a conservative, audit-friendly setting.
- National primary analyses: `site=Overall`, cohort profile `{cohort.profile_id}`, exemplar age `{age_exemplar}`.

## Data feasibility disclosure (RSV lab positivity)
Source: `{src_missingness}`.

- RSV-NET `{age_exemplar}`, `site=Overall`: RSV lab positivity missingness = {f1(float(rsv_missing.frac_missing)*100)}% ({int(rsv_missing.n_missing)}/{int(rsv_missing.n_rows)} weeks).

## C1 — Paired short-horizon improvements vs seasonal naïve (horizon=1; exclude_last_weeks=4)
Source: `{src_paired}` (`metric=MAE`, matched weeks).

- FluSurv-NET `{age_exemplar}` (`ridge_with_signals`): paired MAE improvement {f2(float(flu_pb.improvement_over_baseline))} (95% CI {f2(float(flu_pb.improvement_ci_lo))} to {f2(float(flu_pb.improvement_ci_hi))}); paired_n={int(flu_pb.paired_n)}.
{chr(10).join(rsv_lines)}

## C2 — Site proxy representativeness (feasibility-driven external validity)
Source: `{src_site_trend}` (median Spearman r across sites with overlap≥100 weeks; lag allowed).

- FluSurv-NET `{age_exemplar}`: median Spearman r ≈ {f2(med_r_flu)}.
- RSV-NET `{age_exemplar}`: median Spearman r ≈ {f2(med_r_rsv)}.

## C3 — Decision-analytic evaluation exemplars (exclude_last_weeks=4)
### Expected cost ratios vs baseline (horizon=1; miss:false-alarm cost ratio=2.0)
Source: `{src_expected_cost}` (`expected_cost_ratio_vs_baseline`).

- Selection note: in the primary subset (`exclude_last_weeks=4`, `site=Overall`, age `{age_exemplar}`), there are {ec_primary_n} retrospective method × threshold × cost-ratio × horizon configurations across both viruses. “Best” rows are therefore optimistic retrospective minima and should not be read as prospective guarantees.
- FluSurv-NET `{age_exemplar}` (fixed `train_quantile_0.85`, `ridge_with_signals`): expected-cost ratio = {f3(float(flu_ec["fixed_row"].expected_cost_ratio_vs_baseline))}. Range over the predeclared threshold family for this method: [{f3(float(flu_ec["method_min"]))}, {f3(float(flu_ec["method_max"]))}].
- FluSurv-NET `{age_exemplar}` (optimistic minimum across methods and thresholds): {f3(float(flu_ec["best_row"].expected_cost_ratio_vs_baseline))} ({flu_ec["best_row"].method}; {flu_ec["best_row"].threshold_rule}; threshold={f2(float(flu_ec["best_row"].threshold_value))}).
- RSV-NET `{age_exemplar}` (fixed `train_quantile_0.85`, `{str(rsv_ec["fixed_row"].method)}`): expected-cost ratio = {f3(float(rsv_ec["fixed_row"].expected_cost_ratio_vs_baseline))}. Range over the predeclared threshold family for this method: [{f3(float(rsv_ec["method_min"]))}, {f3(float(rsv_ec["method_max"]))}].
- RSV-NET `{age_exemplar}` (optimistic minimum across methods and thresholds): {f3(float(rsv_ec["best_row"].expected_cost_ratio_vs_baseline))} ({rsv_ec["best_row"].method}; {rsv_ec["best_row"].threshold_rule}; threshold={f2(float(rsv_ec["best_row"].threshold_value))}).

### Alert-utility trade-off (RSV exemplar; horizon=1; threshold=train_quantile_0.85)
Source: `{src_alert_utility}` (paired evaluation set).

- RSV-NET `{age_exemplar}`, `{str(rsv_au.method)}`: sensitivity/specificity = {f3(float(rsv_au.sensitivity))}/{f3(float(rsv_au.specificity))} vs baseline {f3(float(rsv_au.baseline_sensitivity))}/{f3(float(rsv_au.baseline_specificity))}; alert frequency {f3(float(rsv_au.alert_frequency))} vs baseline {f3(float(rsv_au.baseline_alert_frequency))}.

### Pseudo-prospective episode PPV (RSV exemplar; threshold=train_quantile_0.85)
Source: `{src_pseudo}`.

- RSV-NET `{age_exemplar}`, `{str(rsv_pp_ridge.method)}`: episode PPV {f2(float(rsv_pp_ridge.episode_ppv))} (episodes_true={int(rsv_pp_ridge.episodes_true)}, episodes_false={int(rsv_pp_ridge.episodes_false)}; alert-week fraction={f3(float(rsv_pp_ridge.alert_week_fraction))}). {_ppv_note(rsv_pp_ridge)}
- RSV-NET `{age_exemplar}`, seasonal naïve: episode PPV {f2(float(rsv_pp_base.episode_ppv))} (episodes_true={int(rsv_pp_base.episodes_true)}, episodes_false={int(rsv_pp_base.episodes_false)}; alert-week fraction={f3(float(rsv_pp_base.alert_week_fraction))}). {_ppv_note(rsv_pp_base)}
"""

    out_path.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
