#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity-check that key pipeline outputs exist.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument("--cohort-profile", default=None, help="Optional cohort profile label for display only.")
    args = ap.parse_args()

    root = args.root
    # Figures live in `plots/publication/` (manuscript-ready) and `results/figures/` stores figure source tables.
    fig_dir = root / "plots/publication"
    if not (fig_dir / "Fig1_time_series_overview.png").exists():
        alt = root / "results/figures"
        if (alt / "Fig1_time_series_overview.png").exists():
            fig_dir = alt

    required = [
        root / "results/analysis/analysis_table.tsv",
        root / "results/benchmarks/paired_benchmark.tsv",
        root / "results/benchmarks/expected_cost.tsv",
        fig_dir / "Fig1_time_series_overview.png",
        fig_dir / "Fig3_paired_benchmark.png",
        fig_dir / "FigS4_incremental_value_beyond_seasonality.png",
    ]

    missing: list[Path] = [p for p in required if not p.exists()]
    label = f" ({args.cohort_profile})" if args.cohort_profile else ""
    print(f"[smoke_check_outputs] root={root.resolve()}{label}")

    if missing:
        print("[smoke_check_outputs] MISSING:")
        for p in missing:
            print(f"  - {p.as_posix()}")
        return 2

    print("[smoke_check_outputs] OK. Key outputs present:")
    for p in required:
        size = p.stat().st_size
        print(f"  - {p.as_posix()} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
