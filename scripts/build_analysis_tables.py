#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from cohort_profiles import cohort_from_env, get_cohort
from utils import RESULTS_DIR, SNAPSHOT_DIR, ensure_dir, relpath


def latest_snapshot(glob_pattern: str) -> Path:
    paths = sorted(SNAPSHOT_DIR.glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No snapshots matched: {glob_pattern}")
    return paths[-1]


def load_latest_respnet() -> pd.DataFrame:
    path = latest_snapshot("cdc_kvib-3txy/*.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    return df


def load_latest_nrevss_rsv() -> pd.DataFrame:
    path = latest_snapshot("cdc_52kb-ccu2/*.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    return df


def load_latest_viral_positivity() -> pd.DataFrame:
    path = latest_snapshot("cdc_seuz-s2cv/*.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    return df


def load_latest_nssp_ed_visits() -> pd.DataFrame:
    path = latest_snapshot("cdc_7xva-uux8/*.csv.gz")
    df = pd.read_csv(path, compression="gzip")
    return df


def load_latest_wili() -> pd.DataFrame:
    path = latest_snapshot("delphi_fluview_wili/*.json.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    epi = payload.get("epidata", []) or []
    df = pd.DataFrame(epi)
    return df


def build_outcomes(resp: pd.DataFrame, age_groups: list[str]) -> pd.DataFrame:
    # Filter to observed weekly rates and surveillance networks of interest.
    keep = resp.copy()
    keep = keep[keep["rate_type"].astype(str).str.lower() == "observed"]
    keep = keep[keep["surveillance_network"].isin(["FluSurv-NET", "RSV-NET"])]
    keep = keep[keep["age_group"].isin(age_groups)]

    keep["week_ending_date"] = pd.to_datetime(keep["week_ending_date"]).dt.date
    keep["weekly_rate"] = pd.to_numeric(keep["weekly_rate"], errors="coerce")
    keep["mmwr_year"] = pd.to_numeric(keep["mmwr_year"], errors="coerce")
    keep["mmwr_week"] = pd.to_numeric(keep["mmwr_week"], errors="coerce")
    keep["epiweeks"] = (keep["mmwr_year"] * 100 + keep["mmwr_week"]).astype("Int64")

    out = (
        keep[["surveillance_network", "site", "week_ending_date", "epiweeks", "age_group", "weekly_rate"]]
        .rename(columns={"weekly_rate": "rate_per_100k"})
        .sort_values(["surveillance_network", "site", "age_group", "week_ending_date"])
    )
    return out


def build_rsv_signal(nrevss: pd.DataFrame, testtype: str = "PCR") -> pd.DataFrame:
    keep = nrevss.copy()
    keep = keep[keep["testtype"].astype(str) == testtype]
    # NREVSS uses a compact week code (typically YYWW, e.g., 1713 for 2017 week 13).
    # We normalize to the same 6-digit epiweek code used by RESP-NET (YYYYWW).
    keep["repweekcode"] = pd.to_numeric(keep["repweekcode"], errors="coerce").astype("Int64")
    keep["rsvpos"] = pd.to_numeric(keep["rsvpos"], errors="coerce").fillna(0.0)
    keep["rsvtest"] = pd.to_numeric(keep["rsvtest"], errors="coerce").fillna(0.0)

    def normalize_repweekcode(v: object) -> pd._libs.missing.NAType | int:
        if pd.isna(v):
            return pd.NA
        try:
            code = int(v)
        except Exception:
            return pd.NA
        if code >= 100000:
            # Already YYYYWW.
            return code
        if code >= 1000:
            yy = code // 100
            ww = code % 100
            year = 2000 + yy if yy < 100 else yy
            return year * 100 + ww
        return pd.NA

    keep["epiweeks"] = keep["repweekcode"].map(normalize_repweekcode).astype("Int64")
    keep = keep.dropna(subset=["epiweeks"])

    # IMPORTANT: NREVSS has its own date convention; we key positivity by epiweek to align with RESP-NET.
    grp = keep.groupby(["epiweeks"], as_index=False)[["rsvpos", "rsvtest"]].sum()
    grp["rsv_positivity"] = grp["rsvpos"] / grp["rsvtest"].where(grp["rsvtest"] > 0, pd.NA)
    return grp.sort_values("epiweeks")


def build_flu_positivity_signal(vpos: pd.DataFrame) -> pd.DataFrame:
    keep = vpos.copy()
    if keep.empty:
        return pd.DataFrame(columns=["week_ending_date", "flu_positivity"])
    if "pathogen" in keep.columns:
        keep = keep[keep["pathogen"].astype(str) == "Influenza"]
    keep["week_end"] = pd.to_datetime(keep["week_end"], errors="coerce").dt.date
    keep["percent_test_positivity"] = pd.to_numeric(keep.get("percent_test_positivity"), errors="coerce")
    keep = keep.dropna(subset=["week_end", "percent_test_positivity"])
    # Convert percent to proportion.
    keep["flu_positivity"] = keep["percent_test_positivity"] / 100.0
    out = keep.groupby("week_end", as_index=False)["flu_positivity"].mean()
    return out.rename(columns={"week_end": "week_ending_date"}).sort_values("week_ending_date")


def build_wili_signal(wili: pd.DataFrame) -> pd.DataFrame:
    keep = wili.copy()
    if keep.empty:
        return pd.DataFrame(columns=["epiweeks", "wili"])
    keep = keep[keep.get("region").astype(str) == "nat"] if "region" in keep.columns else keep
    keep["epiweeks"] = pd.to_numeric(keep["epiweek"], errors="coerce").astype("Int64")
    keep["wili"] = pd.to_numeric(keep.get("wili"), errors="coerce")
    keep = keep[["epiweeks", "wili"]].dropna(subset=["epiweeks"]).sort_values("epiweeks")
    return keep


def build_nssp_ed_signals(ed: pd.DataFrame, age_map: dict[str, str]) -> pd.DataFrame:
    """
    NSSP ED percent visits, filtered to geography=United States and demographics_type=Age Group in the fetch step.
    Output is weekly, keyed by week_ending_date, with columns for influenza/RSV by age bin.
    """
    keep = ed.copy()
    if keep.empty:
        return pd.DataFrame(columns=["week_ending_date"])

    keep["week_end"] = pd.to_datetime(keep["week_end"], errors="coerce").dt.date
    keep = keep.dropna(subset=["week_end"])
    keep["percent_visits"] = pd.to_numeric(keep.get("percent_visits"), errors="coerce")

    if not age_map:
        return pd.DataFrame(columns=["week_ending_date"])

    keep = keep[keep.get("demographics_type").astype(str) == "Age Group"]
    keep = keep[keep.get("demographics_values").astype(str).isin(age_map.keys())]
    keep["age_bin"] = keep["demographics_values"].map(age_map)
    keep = keep[keep.get("pathogen").astype(str).isin(["Influenza", "RSV"])]

    if keep.empty:
        return pd.DataFrame(columns=["week_ending_date"])

    piv = keep.pivot_table(
        index=["week_end"],
        columns=["pathogen", "age_bin"],
        values="percent_visits",
        aggfunc="mean",
    )
    piv.columns = [f"ed_{p.lower()}_pct_{a}" for p, a in piv.columns.to_list()]
    out = piv.reset_index().rename(columns={"week_end": "week_ending_date"}).sort_values("week_ending_date")

    # Stabilize output columns (so downstream feature code can rely on presence even if an export is sparse).
    expected_cols: list[str] = []
    for suffix in sorted(set(age_map.values())):
        expected_cols.append(f"ed_influenza_pct_{suffix}")
        expected_cols.append(f"ed_rsv_pct_{suffix}")
    for c in expected_cols:
        if c not in out.columns:
            out[c] = pd.NA

    # Convert percent to proportion.
    for c in [x for x in out.columns if x.startswith("ed_") and "_pct_" in x]:
        out[c] = out[c] / 100.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build harmonized outcome/signal tables and figure anchor tables.")
    ap.add_argument(
        "--cohort-profile",
        default=cohort_from_env("older_adult_65plus"),
        help="Cohort profile id (default: env COHORT_PROFILE or 'older_adult_65plus').",
    )
    ap.add_argument(
        "--age-groups",
        default=None,
        help="Comma-separated RESP-NET age_group values to include (overrides cohort defaults).",
    )
    ap.add_argument("--outdir", default=str(RESULTS_DIR), help="Output root directory (default: results/)")
    args = ap.parse_args()

    cohort = get_cohort(args.cohort_profile)
    out_root = Path(args.outdir)
    out_analysis = out_root / "analysis"
    out_figures = out_root / "figures"
    if args.age_groups:
        age_groups = [x.strip() for x in str(args.age_groups).split(",") if x.strip()]
    else:
        age_groups = list(cohort.respnet_age_groups_for_tables)

    resp = load_latest_respnet()
    nrevss = load_latest_nrevss_rsv()
    vpos = load_latest_viral_positivity()
    ed = load_latest_nssp_ed_visits()
    wili = load_latest_wili()

    outcomes = build_outcomes(resp, age_groups=age_groups)
    rsv_signal = build_rsv_signal(nrevss, testtype="PCR")
    flu_pos_signal = build_flu_positivity_signal(vpos)
    ed_signals = build_nssp_ed_signals(ed, age_map=dict(cohort.nssp_ed_age_map))

    # wILI is keyed by epiweek, not date. We keep it as-is for now; later we align via epiweek↔date mapping.
    wili_signal = build_wili_signal(wili)

    ensure_dir(out_figures)
    ensure_dir(out_analysis)

    outcomes.to_csv(out_analysis / "outcomes_weekly.tsv", sep="\t", index=False)
    rsv_signal.to_csv(out_analysis / "signal_rsv_positivity_weekly.tsv", sep="\t", index=False)
    flu_pos_signal.to_csv(out_analysis / "signal_flu_positivity_weekly.tsv", sep="\t", index=False)
    ed_signals.to_csv(out_analysis / "signal_nssp_ed_visits_weekly.tsv", sep="\t", index=False)
    wili_signal.to_csv(out_analysis / "signal_wili.tsv", sep="\t", index=False)

    merged = (
        outcomes.merge(rsv_signal, on=["epiweeks"], how="left")
        .merge(wili_signal, on="epiweeks", how="left")
        .merge(flu_pos_signal, on="week_ending_date", how="left")
        .merge(ed_signals, on="week_ending_date", how="left")
    )

    # Mainline figure anchors restricted to site=Overall for clean narrative figures.
    overview = merged[merged["site"].astype(str) == "Overall"].copy()
    overview.to_csv(out_figures / "time_series_overview.tsv", sep="\t", index=False)

    # External-validity figure anchor: all sites.
    merged.to_csv(out_figures / "time_series_sites.tsv", sep="\t", index=False)

    merged.to_csv(out_analysis / "analysis_table.tsv", sep="\t", index=False)

    print("Wrote:")
    print(f"- {relpath(out_analysis / 'outcomes_weekly.tsv')}")
    print(f"- {relpath(out_analysis / 'signal_rsv_positivity_weekly.tsv')}")
    print(f"- {relpath(out_analysis / 'signal_flu_positivity_weekly.tsv')}")
    print(f"- {relpath(out_analysis / 'signal_nssp_ed_visits_weekly.tsv')}")
    print(f"- {relpath(out_analysis / 'signal_wili.tsv')}")
    print(f"- {relpath(out_analysis / 'analysis_table.tsv')}")
    print(f"- {relpath(out_figures / 'time_series_overview.tsv')}")
    print(f"- {relpath(out_figures / 'time_series_sites.tsv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
