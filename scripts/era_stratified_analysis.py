#!/usr/bin/env python3
"""
Era-stratified supplementary analysis to support the narrative that strong seasonal baselines
can be brittle under non-stationary dynamics (e.g., post-2021 RSV timing/amplitude changes).

This script reads the already-generated rolling-origin predictions (long format) and computes
paired-to-baseline MAE improvements within predeclared time eras (by target week-ending date).

Output is designed to be a supplementary, reviewer-facing table for "why not seasonal naive?".
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


def build_eras(min_date: dt.date, max_date: dt.date, era_cut1: dt.date, era_cut2: dt.date) -> list[dict[str, object]]:
    eras: list[dict[str, object]] = []
    # Era 1: from min_date to day before cut1
    eras.append({"era": "early_post_shift", "start": min_date, "end": min(max_date, era_cut1 - dt.timedelta(days=1))})
    # Era 2: [cut1, day before cut2]
    eras.append({"era": "ed_signal_era", "start": max(min_date, era_cut1), "end": min(max_date, era_cut2 - dt.timedelta(days=1))})
    # Era 3: [cut2, max_date]
    eras.append({"era": "recent", "start": max(min_date, era_cut2), "end": max_date})
    # Drop empty eras
    out: list[dict[str, object]] = []
    for e in eras:
        if e["start"] <= e["end"]:
            out.append(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Era-stratified paired-to-baseline MAE improvements (supplement).")
    ap.add_argument(
        "--predictions-long",
        default=str(RESULTS_DIR / "benchmarks/predictions_long.tsv"),
        help="Predictions long table (default: results/benchmarks/predictions_long.tsv)",
    )
    ap.add_argument(
        "--out",
        default=str(RESULTS_DIR / "benchmarks/era_paired_benchmark.tsv"),
        help="Output TSV path (default: results/benchmarks/era_paired_benchmark.tsv)",
    )
    ap.add_argument("--baseline-method", default="seasonal_naive", help="Baseline method name (default: seasonal_naive)")
    ap.add_argument("--bootstrap-reps", type=int, default=600, help="Moving-block bootstrap reps for CI (default: 600)")
    ap.add_argument("--bootstrap-block-len", type=int, default=4, help="Moving-block length in weeks (default: 4)")
    ap.add_argument("--bootstrap-seed", type=int, default=20260326, help="RNG seed (default: 20260326)")
    ap.add_argument(
        "--era-cut1",
        default="2022-10-01",
        help="Cut date 1 (inclusive) separating early era → ED era (default: 2022-10-01)",
    )
    ap.add_argument(
        "--era-cut2",
        default="2023-10-01",
        help="Cut date 2 (inclusive) separating ED era → recent era (default: 2023-10-01)",
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

    min_date = pred["target_week_ending_date"].min()
    max_date = pred["target_week_ending_date"].max()
    eras = build_eras(min_date, max_date, parse_date(str(args.era_cut1)), parse_date(str(args.era_cut2)))

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

        # Paired absolute errors
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
                if len(g) < 20:
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
                    seed=int(args.bootstrap_seed) + int(hash((era_name, method)) % 100000),
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
