#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


MATCH_KEYS = [
    "origin_epiweek",
    "origin_week_ending_date",
    "target_epiweek",
    "target_week_ending_date",
]

GROUP_KEYS = [
    "exclude_last_weeks",
    "surveillance_network",
    "age_group",
    "site",
    "train_scope",
    "horizon_weeks",
]

MODEL_PAIRS = [
    ("seasonal_naive", "ridge_univariate", "ar_core_vs_seasonal_naive"),
    ("seasonal_naive", "ridge_with_ed", "ar_plus_ed_vs_seasonal_naive"),
    ("ridge_univariate", "ridge_with_ed", "ed_signal"),
    ("ridge_univariate", "ridge_with_signals", "syndromic_or_virologic_signals"),
    ("ridge_univariate", "ridge_with_signals_plus_ed", "signals_plus_ed"),
    ("ridge_univariate", "ridge_with_signals_plus_flu_pos", "signals_plus_flu_positivity"),
    ("ridge_univariate", "ridge_with_signals_plus_flu_pos_plus_ed", "all_public_signals"),
]


def stable_seed(base_seed: int, parts: Iterable[object]) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return (int(base_seed) + zlib.adler32(s)) % (2**32 - 1)


def moving_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if block_len <= 1 or n < 2:
        return rng.integers(0, n, size=n, dtype=int)
    block_len = min(int(block_len), n)
    n_blocks = int(math.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    return np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]


def block_bootstrap_ci(
    diffs: np.ndarray,
    *,
    block_len: int,
    reps: int,
    seed: int,
) -> tuple[float, float]:
    diffs = diffs[np.isfinite(diffs)].astype(float)
    n = int(diffs.size)
    if n < 20 or reps <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    reps = int(reps)
    stats = np.empty(reps, dtype=float)
    for start in range(0, reps, 500):
        stop = min(start + 500, reps)
        idx = np.vstack([moving_block_indices(n, block_len=block_len, rng=rng) for _ in range(stop - start)])
        stats[start:stop] = diffs[idx].mean(axis=1)
    stats = stats[np.isfinite(stats)]
    if stats.size < max(30, reps // 3):
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return (float(lo), float(hi))


def coerce_predictions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "exclude_last_weeks",
        "horizon_weeks",
        "origin_epiweek",
        "target_epiweek",
        "y_true",
        "y_pred",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["origin_week_ending_date", "target_week_ending_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def prediction_validity(preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = GROUP_KEYS + ["method"]
    for group, sub in preds.groupby(keys, dropna=False):
        d = dict(zip(keys, group, strict=True))
        y_pred = pd.to_numeric(sub["y_pred"], errors="coerce")
        neg = y_pred < 0
        d.update(
            {
                "outcome": d["surveillance_network"],
                "recency_exclusion_weeks": int(d["exclude_last_weeks"]),
                "n_predictions": int(y_pred.notna().sum()),
                "n_negative_predictions": int(neg.sum()),
                "min_prediction": float(y_pred.min()) if y_pred.notna().any() else float("nan"),
                "max_prediction": float(y_pred.max()) if y_pred.notna().any() else float("nan"),
            }
        )
        rows.append(d)
    cols = [
        "outcome",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "horizon_weeks",
        "exclude_last_weeks",
        "recency_exclusion_weeks",
        "method",
        "n_predictions",
        "n_negative_predictions",
        "min_prediction",
        "max_prediction",
    ]
    return pd.DataFrame(rows)[cols].sort_values(cols[:8] + ["method"]).reset_index(drop=True)


def matched_pair(
    sub: pd.DataFrame,
    *,
    comparator: str,
    candidate: str,
) -> pd.DataFrame:
    keep = MATCH_KEYS + ["y_true", "y_pred"]
    comp = (
        sub[sub["method"] == comparator][keep]
        .rename(columns={"y_pred": "comparator_pred"})
        .dropna(subset=MATCH_KEYS + ["y_true", "comparator_pred"])
    )
    cand = (
        sub[sub["method"] == candidate][MATCH_KEYS + ["y_pred"]]
        .rename(columns={"y_pred": "candidate_pred"})
        .dropna(subset=MATCH_KEYS + ["candidate_pred"])
    )
    merged = comp.merge(cand, on=MATCH_KEYS, how="inner")
    if merged.empty:
        return merged
    merged = merged.sort_values(["origin_week_ending_date", "target_week_ending_date", "target_epiweek"])
    return merged.reset_index(drop=True)


def nested_signal_increment(preds: pd.DataFrame, *, block_len: int, reps: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, sub in preds.groupby(GROUP_KEYS, dropna=False):
        group_d = dict(zip(GROUP_KEYS, group, strict=True))
        present = set(str(x) for x in sub["method"].dropna().unique())
        for comparator, candidate, signal_layer in MODEL_PAIRS:
            if comparator not in present or candidate not in present:
                continue
            merged = matched_pair(sub, comparator=comparator, candidate=candidate)
            if merged.empty:
                continue
            y = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=float)
            comp = pd.to_numeric(merged["comparator_pred"], errors="coerce").to_numpy(dtype=float)
            cand = pd.to_numeric(merged["candidate_pred"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y) & np.isfinite(comp) & np.isfinite(cand)
            y, comp, cand = y[ok], comp[ok], cand[ok]
            if y.size == 0:
                continue
            comp_loss_i = np.abs(y - comp)
            cand_loss_i = np.abs(y - cand)
            diffs = comp_loss_i - cand_loss_i
            ci_seed = stable_seed(
                seed,
                [
                    group_d["exclude_last_weeks"],
                    group_d["surveillance_network"],
                    group_d["age_group"],
                    group_d["site"],
                    group_d["train_scope"],
                    group_d["horizon_weeks"],
                    comparator,
                    candidate,
                ],
            )
            lo, hi = block_bootstrap_ci(diffs, block_len=block_len, reps=reps, seed=ci_seed)
            rows.append(
                {
                    "outcome": group_d["surveillance_network"],
                    "surveillance_network": group_d["surveillance_network"],
                    "age_group": group_d["age_group"],
                    "site": group_d["site"],
                    "train_scope": group_d["train_scope"],
                    "horizon_weeks": int(group_d["horizon_weeks"]),
                    "exclude_last_weeks": int(group_d["exclude_last_weeks"]),
                    "recency_exclusion_weeks": int(group_d["exclude_last_weeks"]),
                    "comparator_model": comparator,
                    "candidate_model": candidate,
                    "signal_layer_added": signal_layer,
                    "matched_n": int(y.size),
                    "loss_metric": "MAE",
                    "comparator_loss": float(np.mean(comp_loss_i)),
                    "candidate_loss": float(np.mean(cand_loss_i)),
                    "loss_difference": float(np.mean(diffs)),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "inference_method": f"moving_block_bootstrap_mean_loss_difference:block_len={int(block_len)};reps={int(reps)}",
                }
            )
    if not rows:
        return pd.DataFrame()
    cols = [
        "outcome",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "horizon_weeks",
        "exclude_last_weeks",
        "recency_exclusion_weeks",
        "comparator_model",
        "candidate_model",
        "signal_layer_added",
        "matched_n",
        "loss_metric",
        "comparator_loss",
        "candidate_loss",
        "loss_difference",
        "ci_lower",
        "ci_upper",
        "inference_method",
    ]
    return pd.DataFrame(rows)[cols].sort_values(cols[:11]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit nested public-signal incremental value and negative prediction validity."
    )
    ap.add_argument(
        "--predictions",
        default=str(ROOT / "results/benchmarks/predictions_long.tsv"),
        help="Input predictions_long.tsv.",
    )
    ap.add_argument(
        "--out-nested",
        default=str(ROOT / "results/benchmarks/nested_signal_increment.tsv"),
        help="Output nested signal-increment table.",
    )
    ap.add_argument(
        "--out-validity",
        default=str(ROOT / "results/benchmarks/prediction_validity.tsv"),
        help="Output prediction-validity audit table.",
    )
    ap.add_argument("--block-len", type=int, default=4, help="Moving-block bootstrap block length.")
    ap.add_argument("--bootstrap-reps", type=int, default=1000, help="Moving-block bootstrap repetitions.")
    ap.add_argument("--seed", type=int, default=20260715, help="Base random seed.")
    args = ap.parse_args()

    preds = coerce_predictions(pd.read_csv(args.predictions, sep="\t"))

    nested = nested_signal_increment(
        preds,
        block_len=int(args.block_len),
        reps=int(args.bootstrap_reps),
        seed=int(args.seed),
    )
    validity = prediction_validity(preds)

    Path(args.out_nested).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_validity).parent.mkdir(parents=True, exist_ok=True)
    nested.to_csv(args.out_nested, sep="\t", index=False)
    validity.to_csv(args.out_validity, sep="\t", index=False)

    focus = nested[
        (nested["exclude_last_weeks"] == 4)
        & (nested["surveillance_network"] == "RSV-NET")
        & (nested["age_group"] == "65+ yr")
        & (nested["site"] == "Overall")
        & (nested["train_scope"] == "within_site")
        & (nested["horizon_weeks"] == 1)
        & (nested["comparator_model"] == "ridge_univariate")
        & (nested["candidate_model"] == "ridge_with_ed")
    ]
    if not focus.empty:
        print(focus.to_string(index=False))
    print(f"Wrote {args.out_nested} ({len(nested)} rows)")
    print(f"Wrote {args.out_validity} ({len(validity)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
