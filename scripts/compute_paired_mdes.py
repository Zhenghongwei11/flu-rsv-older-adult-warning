#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from cohort_profiles import cohort_from_env, get_cohort
from utils import relpath

ROOT = Path(__file__).resolve().parents[1]


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def _mdes_normal(sd: float, n: int, alpha_two_sided: float, power: float) -> float:
    if n <= 0 or not np.isfinite(sd) or sd <= 0:
        return float("nan")
    nd = NormalDist()
    z = float(nd.inv_cdf(1 - alpha_two_sided / 2) + nd.inv_cdf(power))
    return float(z * sd / sqrt(n))


def _paired_improvement_series(
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
) -> pd.Series:
    keys = [
        "exclude_last_weeks",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "horizon_weeks",
    ]
    sub = preds.copy()
    sub = sub[
        (sub["exclude_last_weeks"] == exclude_last_weeks)
        & (sub["surveillance_network"] == surveillance_network)
        & (sub["age_group"] == age_group)
        & (sub["site"] == site)
        & (sub["train_scope"] == train_scope)
        & (sub["horizon_weeks"] == horizon_weeks)
        & (sub["method"].isin([baseline_method, method]))
    ]
    if sub.empty:
        return pd.Series(dtype=float)

    base = sub[sub["method"] == baseline_method][["target_epiweek", "y_true", "y_pred"]].rename(
        columns={"y_pred": "y_pred_base"}
    )
    met = sub[sub["method"] == method][["target_epiweek", "y_pred"]].rename(columns={"y_pred": "y_pred_method"})
    merged = base.merge(met, on="target_epiweek", how="inner")
    if merged.empty:
        return pd.Series(dtype=float)

    ae_base = (merged["y_true"] - merged["y_pred_base"]).abs()
    ae_method = (merged["y_true"] - merged["y_pred_method"]).abs()
    imp = ae_base - ae_method
    return imp.astype(float)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute MDES for paired MAE improvements (normal approximation).")
    ap.add_argument(
        "--cohort-profile",
        default=cohort_from_env("older_adult_65plus"),
        help="Cohort profile id (default: env COHORT_PROFILE or 'older_adult_65plus').",
    )
    ap.add_argument(
        "--predictions",
        default=str(ROOT / "results/benchmarks/predictions_long.tsv"),
        help="Path to predictions_long.tsv",
    )
    ap.add_argument(
        "--paired-benchmark",
        default=str(ROOT / "results/benchmarks/paired_benchmark.tsv"),
        help="Path to paired_benchmark.tsv (used to enumerate comparisons).",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "results/benchmarks/paired_mdes.tsv"),
        help="Output TSV path.",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha (default: 0.05)")
    ap.add_argument("--power", type=float, default=0.8, help="Target power (default: 0.8)")
    args = ap.parse_args()

    cohort = get_cohort(args.cohort_profile)
    preds = _read_tsv(args.predictions)
    paired = _read_tsv(args.paired_benchmark)

    # Enumerate MAE comparisons and compute MDES for the cohort's primary endpoints.
    keep_age = set(cohort.primary_respnet_ages)
    paired = paired[(paired["metric"] == "MAE") & (paired["age_group"].isin(sorted(keep_age)))]
    if paired.empty:
        raise SystemExit(f"No MAE paired rows found for cohort primary ages {sorted(keep_age)!r}")

    rows: list[dict[str, object]] = []
    for r in paired.itertuples(index=False):
        imp = _paired_improvement_series(
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
        n = int(imp.shape[0])
        sd = float(imp.std(ddof=1)) if n >= 2 else float("nan")
        mdes = _mdes_normal(sd, n=n, alpha_two_sided=float(args.alpha), power=float(args.power))
        rows.append(
            {
                "exclude_last_weeks": int(r.exclude_last_weeks),
                "surveillance_network": str(r.surveillance_network),
                "age_group": str(r.age_group),
                "site": str(r.site),
                "train_scope": str(r.train_scope),
                "horizon_weeks": int(r.horizon_weeks),
                "baseline_method": "seasonal_naive",
                "method": str(r.method),
                "paired_n": n,
                "observed_improvement": float(imp.mean()) if n > 0 else float("nan"),
                "sd_weekly_improvement": sd,
                "mdes_improvement": mdes,
                "mdes_alpha_two_sided": float(args.alpha),
                "mdes_power": float(args.power),
                "notes": "MDES uses normal approximation for paired MAE improvement; interpret as an effect-size scale, not a hypothesis test.",
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["exclude_last_weeks", "surveillance_network", "age_group", "site", "horizon_weeks", "method"]
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {relpath(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
