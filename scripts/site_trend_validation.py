#!/usr/bin/env python3
"""
Spatial/site validation:
Quantify how well site-level all-ages RESP-NET trends align with national cohort trends.

Rationale: the public export often has richer site coverage for age_group=Overall than for
age-stratified series. To use site-level series as an external validity check for a national
main analysis, we provide an evidence table showing concordance between:
  (a) a national reference series (site=Overall, cohort primary age group(s)), and
  (b) site-level all-ages series (site!=Overall, age_group=Overall),
using Spearman correlation with moving-block bootstrap CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_profiles import cohort_from_env, get_cohort

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=int)
    block_len = int(max(1, block_len))
    if block_len <= 1 or n < 2:
        return rng.integers(0, n, size=n, dtype=int)
    block_len = int(min(block_len, n))
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks, dtype=int)
    idx = np.concatenate([np.arange(int(s), int(s) + block_len, dtype=int) for s in starts.tolist()])[:n]
    return np.asarray(idx, dtype=int)


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman = Pearson on ranks.
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if len(xr) < 3:
        return float("nan")
    # Avoid numpy warnings when ranks are constant (std==0).
    if not (np.isfinite(xr).all() and np.isfinite(yr).all()):
        return float("nan")
    if float(np.std(xr)) <= 0.0 or float(np.std(yr)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def bootstrap_ci_spearman(
    x: np.ndarray,
    y: np.ndarray,
    reps: int,
    block_len: int,
    seed: int,
) -> tuple[float, float]:
    n = int(len(x))
    if n < 30 or reps <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(int(seed))
    vals: list[float] = []
    for _ in range(int(reps)):
        idx = moving_block_bootstrap_indices(n, block_len=block_len, rng=rng)
        r = spearman_r(x[idx], y[idx])
        if np.isfinite(r):
            vals.append(float(r))
    if len(vals) < max(30, reps // 3):
        return (float("nan"), float("nan"))
    arr = np.asarray(vals, dtype=float)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Site validation via concordance of national cohort vs site all-ages trends.")
    ap.add_argument(
        "--cohort-profile",
        default=cohort_from_env("older_adult_65plus"),
        help="Cohort profile id (default: env COHORT_PROFILE or 'older_adult_65plus').",
    )
    ap.add_argument(
        "--analysis-table",
        default=str(RESULTS_DIR / "analysis/analysis_table.tsv"),
        help="Harmonized analysis table (default: results/analysis/analysis_table.tsv)",
    )
    ap.add_argument(
        "--out",
        default=str(RESULTS_DIR / "benchmarks/site_trend_correlation.tsv"),
        help="Output TSV path (default: results/benchmarks/site_trend_correlation.tsv)",
    )
    ap.add_argument("--lags", default="-4,-3,-2,-1,0,1,2,3,4", help="Comma-separated lag weeks to test (default: -4..4)")
    ap.add_argument("--bootstrap-reps", type=int, default=600, help="Block-bootstrap reps for CI (default: 600)")
    ap.add_argument("--bootstrap-block-len", type=int, default=4, help="Block length in weeks (default: 4)")
    ap.add_argument("--bootstrap-seed", type=int, default=20260326, help="Seed (default: 20260326)")
    args = ap.parse_args()
    cohort = get_cohort(args.cohort_profile)

    df = pd.read_csv(args.analysis_table, sep="\t")
    if df.empty:
        raise SystemExit("Empty analysis table.")

    df["week_ending_date"] = pd.to_datetime(df["week_ending_date"], errors="coerce").dt.date
    df["rate_per_100k"] = pd.to_numeric(df["rate_per_100k"], errors="coerce")
    df = df.dropna(subset=["week_ending_date", "rate_per_100k", "surveillance_network", "site", "age_group"])

    lags = [int(x.strip()) for x in str(args.lags).split(",") if x.strip()]

    # National cohort reference series (site=Overall, age_group in cohort primary groups).
    ref = df[(df["site"].astype(str) == "Overall") & (df["age_group"].isin(list(cohort.primary_respnet_ages)))].copy()
    if ref.empty:
        raise SystemExit("No national cohort reference series found (site=Overall, age_group in cohort primary groups).")

    # Site all-ages series (site != Overall, age_group=Overall).
    sites = df[(df["site"].astype(str) != "Overall") & (df["age_group"].astype(str) == "Overall")].copy()
    if sites.empty:
        raise SystemExit("No site all-ages series found (site!=Overall, age_group=Overall).")

    rows: list[dict[str, object]] = []
    for (net, age), g_ref in ref.groupby(["surveillance_network", "age_group"]):
        g_ref = g_ref.sort_values("week_ending_date")
        y_ref = g_ref[["week_ending_date", "rate_per_100k"]].rename(columns={"rate_per_100k": "y_ref"})

        for site, g_site in sites[sites["surveillance_network"] == net].groupby("site"):
            g_site = g_site.sort_values("week_ending_date")
            y_site = g_site[["week_ending_date", "rate_per_100k"]].rename(columns={"rate_per_100k": "y_site"})

            merged0 = y_ref.merge(y_site, on="week_ending_date", how="inner").sort_values("week_ending_date")
            if len(merged0) < 30:
                continue

            best = None
            for lag in lags:
                m = merged0.copy()
                # lag > 0 means: compare y_ref(t) with y_site(t+lag) => shift site backward.
                m["y_site_lagged"] = m["y_site"].shift(-lag)
                m2 = m.dropna(subset=["y_ref", "y_site_lagged"])
                if len(m2) < 30:
                    continue
                x = m2["y_ref"].to_numpy(dtype=float)
                y = m2["y_site_lagged"].to_numpy(dtype=float)
                r = spearman_r(x, y)
                if not np.isfinite(r):
                    continue
                lo, hi = bootstrap_ci_spearman(x, y, reps=int(args.bootstrap_reps), block_len=int(args.bootstrap_block_len), seed=int(args.bootstrap_seed) + lag)
                cand = {
                    "surveillance_network": str(net),
                    "reference_age_group": str(age),
                    "site": str(site),
                    "lag_weeks_site_minus_national": int(lag),
                    "n_overlap": int(len(m2)),
                    "spearman_r": float(r),
                    "spearman_ci_lo": float(lo),
                    "spearman_ci_hi": float(hi),
                    "week_min": str(m2["week_ending_date"].min()),
                    "week_max": str(m2["week_ending_date"].max()),
                }
                if best is None or abs(cand["spearman_r"]) > abs(best["spearman_r"]):
                    best = cand

            if best is not None:
                rows.append(best)

    out = pd.DataFrame(rows)
    ensure_dir(Path(args.out).parent)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
