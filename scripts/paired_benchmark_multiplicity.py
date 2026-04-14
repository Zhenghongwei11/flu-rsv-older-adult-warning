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


def _stable_seed(base_seed: int, parts: Iterable[object]) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return (int(base_seed) + zlib.adler32(s)) % (2**32 - 1)


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _paired_improvements(
    preds: pd.DataFrame,
    *,
    exclude_last_weeks: int,
    surveillance_network: str,
    age_group: str,
    site: str,
    train_scope: str,
    horizon_weeks: int,
    method: str,
    baseline_method: str = "seasonal_naive",
) -> np.ndarray:
    sub = preds[
        (preds["exclude_last_weeks"] == exclude_last_weeks)
        & (preds["surveillance_network"] == surveillance_network)
        & (preds["age_group"] == age_group)
        & (preds["site"] == site)
        & (preds["train_scope"] == train_scope)
        & (preds["horizon_weeks"] == horizon_weeks)
        & (preds["method"].isin([baseline_method, method]))
    ].copy()
    if sub.empty:
        return np.array([], dtype=float)

    base = sub[sub["method"] == baseline_method][["target_epiweek", "y_true", "y_pred"]].rename(
        columns={"y_pred": "y_pred_base"}
    )
    met = sub[sub["method"] == method][["target_epiweek", "y_pred"]].rename(columns={"y_pred": "y_pred_method"})
    merged = base.merge(met, on="target_epiweek", how="inner")
    if merged.empty:
        return np.array([], dtype=float)

    y = pd.to_numeric(merged["y_true"], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(merged["y_pred_base"], errors="coerce").to_numpy(dtype=float)
    m = pd.to_numeric(merged["y_pred_method"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(b) & np.isfinite(m)
    y = y[ok]
    b = b[ok]
    m = m[ok]
    imp = np.abs(y - b) - np.abs(y - m)
    return imp.astype(float)


def _sign_flip_pvalue(
    diffs: np.ndarray,
    *,
    rng: np.random.Generator,
    reps: int = 20_000,
) -> float:
    """
    Two-sided sign-flip permutation test for mean(diffs)=0.

    Notes:
    - Assumes symmetry of diffs around 0 under the null.
    - Ignores serial correlation; treat as an approximate reviewer-facing diagnostic.
    """
    diffs = diffs[np.isfinite(diffs)].astype(float)
    n = int(diffs.shape[0])
    if n < 5:
        return float("nan")
    t_obs = float(np.mean(diffs))
    # Exact enumeration for tiny n.
    if n <= 20:
        # All 2^n sign patterns.
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * n)).T.reshape(-1, n)
        stats = (signs * diffs[None, :]).mean(axis=1)
    else:
        r = int(max(1_000, reps))
        signs = rng.choice([-1.0, 1.0], size=(r, n), replace=True)
        stats = (signs * diffs[None, :]).mean(axis=1)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return float("nan")
    p = float(np.mean(np.abs(stats) >= abs(t_obs)))
    # Conservative: ensure non-zero p for finite sample.
    p = max(p, 1.0 / float(stats.size))
    return min(1.0, p)


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = pvals.astype(float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = int(order.size)
    adj = np.empty(m, dtype=float)
    for k, i in enumerate(order):
        adj[k] = (m - k) * p[i]
    # enforce monotonicity
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)
    out[order] = adj
    return out


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = pvals.astype(float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = int(order.size)
    q = np.empty(m, dtype=float)
    for k, i in enumerate(order, start=1):
        q[k - 1] = p[i] * m / k
    # enforce monotone non-increasing when walking back
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out[order] = q
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Multiplicity-aware p-value table for paired MAE improvements.")
    ap.add_argument(
        "--predictions",
        default=str(ROOT / "results/benchmarks/predictions_long.tsv"),
        help="Path to predictions_long.tsv",
    )
    ap.add_argument(
        "--paired-benchmark",
        default=str(ROOT / "results/benchmarks/paired_benchmark.tsv"),
        help="Path to paired_benchmark.tsv",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "results/benchmarks/paired_benchmark_multiplicity.tsv"),
        help="Output TSV path.",
    )
    ap.add_argument("--exclude-last-weeks", type=int, default=4, help="Primary exclude_last_weeks family (default: 4)")
    ap.add_argument("--site", default="Overall", help="Site filter for multiplicity family (default: Overall)")
    ap.add_argument("--age-group", default="65+ yr", help="Age-group filter for multiplicity family (default: 65+ yr)")
    ap.add_argument("--metric", default="MAE", help="Metric filter for multiplicity family (default: MAE)")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed (default: 123)")
    ap.add_argument("--signflip-reps", type=int, default=20_000, help="Monte Carlo reps for sign-flip test (default: 20000)")
    args = ap.parse_args()

    preds = _read_tsv(args.predictions)
    paired = _read_tsv(args.paired_benchmark)

    fam = paired[
        (paired["exclude_last_weeks"] == int(args.exclude_last_weeks))
        & (paired["site"] == str(args.site))
        & (paired["age_group"] == str(args.age_group))
        & (paired["metric"] == str(args.metric))
    ].copy()
    if fam.empty:
        raise SystemExit("No rows matched the multiplicity family filters.")

    fam = fam.sort_values(["surveillance_network", "horizon_weeks", "method"]).reset_index(drop=True)

    pvals: list[float] = []
    observed_check: list[float] = []
    paired_n: list[int] = []
    for r in fam.itertuples(index=False):
        diffs = _paired_improvements(
            preds,
            exclude_last_weeks=int(r.exclude_last_weeks),
            surveillance_network=str(r.surveillance_network),
            age_group=str(r.age_group),
            site=str(r.site),
            train_scope=str(r.train_scope),
            horizon_weeks=int(r.horizon_weeks),
            method=str(r.method),
            baseline_method="seasonal_naive",
        )
        paired_n.append(int(diffs.shape[0]))
        observed_check.append(float(np.mean(diffs)) if diffs.size else float("nan"))
        rng = np.random.default_rng(_stable_seed(int(args.seed), [r.surveillance_network, r.horizon_weeks, r.method]))
        pvals.append(_sign_flip_pvalue(diffs, rng=rng, reps=int(args.signflip_reps)))

    fam["paired_n_recomputed"] = paired_n
    fam["observed_improvement_recomputed"] = observed_check
    fam["p_value_signflip_two_sided"] = pvals

    p_arr = np.asarray(pvals, dtype=float)
    fam["p_holm_family"] = _holm_adjust(p_arr)
    fam["q_bh_fdr_family"] = _bh_fdr(p_arr)
    fam["multiplicity_family"] = (
        f"exclude_last_weeks={int(args.exclude_last_weeks)};site={args.site};age_group={args.age_group};metric={args.metric}"
    )
    # Predeclared claim set: keep inference scoped to the 4 main-text comparisons
    # (flu ridge_with_signals and RSV ridge_with_ed at horizons 1–2).
    claim_mask = (
        ((fam["surveillance_network"] == "FluSurv-NET") & (fam["method"] == "ridge_with_signals") & (fam["horizon_weeks"].isin([1, 2])))
        | ((fam["surveillance_network"] == "RSV-NET") & (fam["method"] == "ridge_with_ed") & (fam["horizon_weeks"].isin([1, 2])))
    )
    fam["p_holm_claimset"] = float("nan")
    if claim_mask.sum() > 0:
        claim_p = fam.loc[claim_mask, "p_value_signflip_two_sided"].to_numpy(dtype=float)
        fam.loc[claim_mask, "p_holm_claimset"] = _holm_adjust(claim_p)
    fam["claimset_definition"] = (
        "Holm adjustment within the 4 predeclared main-text paired-MAE comparisons: "
        "FluSurv-NET ridge_with_signals (h=1,2) and RSV-NET ridge_with_ed (h=1,2), "
        "all at exclude_last_weeks=4, site=Overall, age_group=65+ yr."
    )
    fam["p_value_notes"] = (
        "Two-sided sign-flip permutation p-value for mean paired MAE improvement; approximate (ignores serial correlation). "
        "Holm and BH-FDR are computed across the family defined in multiplicity_family; p_holm_claimset is Holm within the predeclared claim set."
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fam.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
