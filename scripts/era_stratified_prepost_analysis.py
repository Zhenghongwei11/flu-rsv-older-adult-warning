#!/usr/bin/env python3
"""
Era-stratified supplementary analysis for reviewer-facing regime-shift concerns.

Goal: quantify paired-to-baseline MAE improvements separately in two time eras that bracket
the post-2021 seasonality disruption. Because the current RESP-NET public export is seasonal
and includes multi-month gaps, the earliest evaluable target weeks in our rolling-origin
prediction tables begin in 2020-03; we therefore operationalize the “pre” era as the earliest
available era in the export rather than strictly pre-2020.

This uses the already-generated rolling-origin predictions (long format) and computes paired
MAE improvements vs the seasonal-naïve baseline on matched target weeks within each era.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


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


def bootstrap_ci_improvement(
    abs_err_baseline: np.ndarray,
    abs_err_method: np.ndarray,
    reps: int,
    block_len: int,
    seed: int,
) -> tuple[float, float]:
    n = int(len(abs_err_baseline))
    if n <= 0:
        return (float("nan"), float("nan"))
    if reps <= 0 or n < 20:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(int(seed))
    imps: list[float] = []
    for _ in range(int(reps)):
        idx = moving_block_bootstrap_indices(n, block_len=block_len, rng=rng)
        b = abs_err_baseline[idx]
        m = abs_err_method[idx]
        if not (np.isfinite(b).all() and np.isfinite(m).all()):
            continue
        imps.append(float(np.mean(b) - np.mean(m)))
    if len(imps) < max(30, int(0.5 * reps)):
        return (float("nan"), float("nan"))
    arr = np.asarray(imps, dtype=float)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre/post era-stratified paired MAE improvements (supplement).")
    ap.add_argument(
        "--predictions-long",
        default=str(RESULTS_DIR / "benchmarks/predictions_long.tsv"),
        help="Predictions long table (default: results/benchmarks/predictions_long.tsv)",
    )
    ap.add_argument(
        "--out",
        default=str(RESULTS_DIR / "benchmarks/era_paired_benchmark_prepost2021.tsv"),
        help="Output TSV path (default: results/benchmarks/era_paired_benchmark_prepost2021.tsv)",
    )
    ap.add_argument("--baseline-method", default="seasonal_naive", help="Baseline method name (default: seasonal_naive)")
    ap.add_argument("--bootstrap-reps", type=int, default=600, help="Moving-block bootstrap reps for CI (default: 600)")
    ap.add_argument("--bootstrap-block-len", type=int, default=4, help="Moving-block length in weeks (default: 4)")
    ap.add_argument("--bootstrap-seed", type=int, default=20260412, help="RNG seed (default: 20260412)")
    ap.add_argument("--era1-name", default="early_export_era", help="Era 1 label (default: early_export_era)")
    ap.add_argument("--era2-name", default="post_2021", help="Era 2 label (default: post_2021)")
    ap.add_argument(
        "--era1-end",
        default="2021-09-30",
        help="Era 1 end date inclusive (default: 2021-09-30)",
    )
    ap.add_argument(
        "--era2-start",
        default="2021-10-01",
        help="Era 2 start date inclusive (default: 2021-10-01)",
    )
    ap.add_argument(
        "--min-paired-n",
        type=int,
        default=20,
        help="Minimum paired forecast count within an era slice (default: 20)",
    )
    args = ap.parse_args()

    pred = pd.read_csv(args.predictions_long, sep="\t")
    if pred.empty:
        raise SystemExit("No predictions found.")

    pred["target_week_ending_date"] = pd.to_datetime(pred["target_week_ending_date"], errors="coerce").dt.date
    pred = pred.dropna(subset=["target_week_ending_date", "y_true", "y_pred"])
    pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
    pred = pred[np.isfinite(pred["y_true"]) & np.isfinite(pred["y_pred"])]
    if pred.empty:
        raise SystemExit("No finite predictions after cleaning.")

    era1_name = str(args.era1_name)
    era2_name = str(args.era2_name)
    era1_end = parse_date(str(args.era1_end))
    era2_start = parse_date(str(args.era2_start))
    eras = [
        {"era": era1_name, "start": pred["target_week_ending_date"].min(), "end": era1_end},
        {"era": era2_name, "start": era2_start, "end": pred["target_week_ending_date"].max()},
    ]

    key_cols = [
        "exclude_last_weeks",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "horizon_weeks",
        "origin_idx",
        "target_epiweek",
        "target_week_ending_date",
        "y_true",
    ]

    baseline_name = str(args.baseline_method)
    base = pred[pred["method"].astype(str) == baseline_name].copy()
    if base.empty:
        raise SystemExit(f"Baseline method not found: {baseline_name}")
    base = base[key_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_baseline"})

    methods = sorted([m for m in pred["method"].astype(str).unique().tolist() if m != baseline_name])
    rows: list[dict[str, object]] = []

    for method in methods:
        pm = pred[pred["method"].astype(str) == method].copy()
        if pm.empty:
            continue
        pm = pm[key_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_method"})

        merged = pm.merge(base, on=key_cols, how="inner")
        if merged.empty:
            continue
        merged = merged.sort_values(["target_week_ending_date", "origin_idx"]).reset_index(drop=True)

        merged["abs_err_baseline"] = (merged["y_true"] - merged["y_pred_baseline"]).abs()
        merged["abs_err_method"] = (merged["y_true"] - merged["y_pred_method"]).abs()

        grp_cols = [
            "exclude_last_weeks",
            "surveillance_network",
            "age_group",
            "site",
            "train_scope",
            "horizon_weeks",
        ]

        for era in eras:
            era_name = str(era["era"])
            start = era["start"]
            end = era["end"]
            sub = merged[(merged["target_week_ending_date"] >= start) & (merged["target_week_ending_date"] <= end)].copy()
            if sub.empty:
                continue

            for keys, g in sub.groupby(grp_cols):
                if len(g) < int(args.min_paired_n):
                    continue
                b = g["abs_err_baseline"].to_numpy(dtype=float)
                m = g["abs_err_method"].to_numpy(dtype=float)
                if not (np.isfinite(b).all() and np.isfinite(m).all()):
                    continue

                imp = float(np.mean(b) - np.mean(m))
                lo, hi = bootstrap_ci_improvement(
                    b,
                    m,
                    reps=int(args.bootstrap_reps),
                    block_len=int(args.bootstrap_block_len),
                    seed=int(args.bootstrap_seed) + int(hash((era_name, method, keys)) % 100000),
                )

                row = dict(zip(grp_cols, keys))
                row.update(
                    {
                        "era": era_name,
                        "era_start": start.isoformat(),
                        "era_end": end.isoformat(),
                        "method": method,
                        "paired_n": int(len(g)),
                        "metric": "MAE",
                        "baseline_value": float(np.mean(b)),
                        "method_value": float(np.mean(m)),
                        "improvement_over_baseline": imp,
                        "improvement_ci_lo": lo,
                        "improvement_ci_hi": hi,
                    }
                )
                rows.append(row)

    out = pd.DataFrame(rows)
    ensure_dir(Path(args.out).parent)
    out.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
