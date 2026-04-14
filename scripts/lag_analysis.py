#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pathlib import Path

from utils import RESULTS_DIR, ensure_dir


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 20:
        return None
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute simple lag-correlation tables for plausibility / lag selection.")
    ap.add_argument("--input", default=str(RESULTS_DIR / "analysis/analysis_table.tsv"))
    ap.add_argument("--max-lag", type=int, default=8, help="Max lag in weeks (default: 8)")
    ap.add_argument("--out", default=str(RESULTS_DIR / "figures/lag_analysis.tsv"), help="Output TSV path.")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    if "site" in df.columns:
        df = df[df["site"].astype(str) == "Overall"].copy()
    df = df.sort_values(["surveillance_network", "age_group", "epiweeks"]).reset_index(drop=True)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    rows: list[dict[str, object]] = []

    for (net, age), g in df.groupby(["surveillance_network", "age_group"]):
        g = g.drop_duplicates("epiweeks").sort_values("epiweeks")
        y = pd.to_numeric(g["rate_per_100k"], errors="coerce").to_numpy(dtype=float)

        signals = {
            "rsv_positivity": pd.to_numeric(g.get("rsv_positivity"), errors="coerce").to_numpy(dtype=float),
            "wili": pd.to_numeric(g.get("wili"), errors="coerce").to_numpy(dtype=float),
        }

        for s_name, s in signals.items():
            for lag in range(0, args.max_lag + 1):
                # correlate y_t with s_{t-lag}
                a = y[lag:]
                b = s[:-lag] if lag > 0 else s
                c = corr(a, b)
                if c is None:
                    continue
                rows.append(
                    {
                        "surveillance_network": net,
                        "age_group": age,
                        "signal": s_name,
                        "lag_weeks": lag,
                        "n": int(np.isfinite(a).sum()),
                        "corr": c,
                    }
                )

    out = pd.DataFrame(rows).sort_values(["surveillance_network", "age_group", "signal", "lag_weeks"])
    out.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
