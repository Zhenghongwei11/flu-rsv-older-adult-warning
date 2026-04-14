#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import RESULTS_DIR, ensure_dir, relpath


ID_COLS = {"surveillance_network", "site", "week_ending_date", "epiweeks", "age_group"}


def missingness_long(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ID_COLS]
    rows: list[dict[str, object]] = []
    for key, g in df.groupby(group_cols):
        g = g.copy()
        n = int(len(g))
        if n == 0:
            continue
        tmin = str(g["week_ending_date"].min()) if "week_ending_date" in g.columns else ""
        tmax = str(g["week_ending_date"].max()) if "week_ending_date" in g.columns else ""
        base = {k: v for k, v in zip(group_cols, key)} if isinstance(key, tuple) else {group_cols[0]: key}
        for c in cols:
            nm = int(pd.isna(g[c]).sum())
            rows.append(
                {
                    **base,
                    "n_rows": n,
                    "time_min": tmin,
                    "time_max": tmax,
                    "variable": c,
                    "n_missing": nm,
                    "frac_missing": float(nm / n),
                }
            )
    return pd.DataFrame(rows)


def coverage_long(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ID_COLS]
    rows: list[dict[str, object]] = []
    for key, g in df.groupby(group_cols):
        g = g.sort_values("epiweeks").copy()
        base = {k: v for k, v in zip(group_cols, key)} if isinstance(key, tuple) else {group_cols[0]: key}
        for c in cols:
            ok = g[c].notna()
            n_ok = int(ok.sum())
            if n_ok == 0:
                rows.append(
                    {
                        **base,
                        "variable": c,
                        "n_nonmissing": 0,
                        "first_epiweeks": pd.NA,
                        "last_epiweeks": pd.NA,
                        "first_week_ending_date": pd.NA,
                        "last_week_ending_date": pd.NA,
                    }
                )
                continue
            gg = g.loc[ok, ["epiweeks", "week_ending_date"]]
            rows.append(
                {
                    **base,
                    "variable": c,
                    "n_nonmissing": n_ok,
                    "first_epiweeks": int(gg["epiweeks"].iloc[0]) if pd.notna(gg["epiweeks"].iloc[0]) else pd.NA,
                    "last_epiweeks": int(gg["epiweeks"].iloc[-1]) if pd.notna(gg["epiweeks"].iloc[-1]) else pd.NA,
                    "first_week_ending_date": str(gg["week_ending_date"].iloc[0]),
                    "last_week_ending_date": str(gg["week_ending_date"].iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit data quality: missingness + signal coverage in analysis table.")
    ap.add_argument("--input", default=str(RESULTS_DIR / "analysis/analysis_table.tsv"))
    ap.add_argument("--outdir", default=str(RESULTS_DIR / "analysis"))
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    df = pd.read_csv(inp, sep="\t")
    ensure_dir(Path(args.outdir))

    group_cols = ["surveillance_network", "age_group", "site"]
    miss = missingness_long(df, group_cols=group_cols)
    cov = coverage_long(df, group_cols=group_cols)

    miss_path = Path(args.outdir) / "missingness_report.tsv"
    cov_path = Path(args.outdir) / "signal_coverage_report.tsv"
    miss.to_csv(miss_path, sep="\t", index=False)
    cov.to_csv(cov_path, sep="\t", index=False)

    print(f"Wrote {relpath(miss_path)}")
    print(f"Wrote {relpath(cov_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
