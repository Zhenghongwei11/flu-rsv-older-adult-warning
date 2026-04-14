#!/usr/bin/env python3
"""
Pseudo-prospective validation:
Simulate a "Monday morning report" that issues/clears an alert based on short-horizon forecasts.

We implement a simple, auditable alert policy:
  - At each origin week, compute max predicted burden over horizons 1–4.
  - Raise an alert if max_pred >= threshold_value.
  - Alert episodes are consecutive alert weeks (state machine).
  - A true episode is one where an event onset occurs within the next 4 weeks.

Outputs:
  - episode-level summary table (PPV, false episodes, lead time) suitable for the main text/supplement.
  - episode log (one row per episode) for audit and reviewer traceability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_profiles import cohort_from_env, get_cohort

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def onset_weeks(event: np.ndarray) -> np.ndarray:
    if len(event) == 0:
        return np.array([], dtype=bool)
    prev = np.concatenate([[False], event[:-1]])
    return event & (~prev)


@dataclass(frozen=True)
class Episode:
    start_epiweek: int
    end_epiweek: int
    start_date: str
    end_date: str
    true_episode: bool
    onset_epiweek: int | None
    lead_weeks: int | None


def build_episode_log(origin_grid: pd.DataFrame, alert: np.ndarray, onset_targets: set[int], lookahead_weeks: int) -> list[Episode]:
    epiweeks = origin_grid["origin_epiweek"].to_numpy(dtype=int)
    dates = origin_grid["origin_week_ending_date"].astype(str).to_list()
    n = len(epiweeks)
    episodes: list[Episode] = []
    i = 0
    while i < n:
        if not bool(alert[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(alert[j + 1]):
            j += 1

        start_w = int(epiweeks[i])
        end_w = int(epiweeks[j])
        # Look ahead from episode start using actual future target epiweeks derived from predictions.
        # IMPORTANT: epiweek codes are not simple integers across year boundaries, so we do NOT use
        # arithmetic like `start_w + k` for matching.
        true_episode = False
        lead = None
        onset_week = None
        # origin_grid contains origin_idx; we use actual future target_epiweek values stored in origin_grid (precomputed).
        future = origin_grid.iloc[i]["future_target_epiweeks"]
        if isinstance(future, list):
            for step, tw in enumerate(future[: int(lookahead_weeks)], start=1):
                if int(tw) in onset_targets:
                    true_episode = True
                    onset_week = int(tw)
                    lead = int(step)
                    break

        episodes.append(
            Episode(
                start_epiweek=start_w,
                end_epiweek=end_w,
                start_date=str(dates[i]),
                end_date=str(dates[j]),
                true_episode=bool(true_episode),
                onset_epiweek=onset_week,
                lead_weeks=lead,
            )
        )
        i = j + 1
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser(description="Pseudo-prospective weekly alert simulation from rolling-origin predictions.")
    ap.add_argument(
        "--cohort-profile",
        default=cohort_from_env("older_adult_65plus"),
        help="Cohort profile id (default: env COHORT_PROFILE or 'older_adult_65plus').",
    )
    ap.add_argument(
        "--predictions-long",
        default=str(RESULTS_DIR / "benchmarks/predictions_long.tsv"),
        help="Predictions long table (default: results/benchmarks/predictions_long.tsv)",
    )
    ap.add_argument(
        "--alert-utility",
        default=str(RESULTS_DIR / "benchmarks/alert_utility.tsv"),
        help="Alert utility table (default: results/benchmarks/alert_utility.tsv) used to source threshold values",
    )
    ap.add_argument(
        "--out-summary",
        default=str(RESULTS_DIR / "benchmarks/pseudo_prospective_episode_summary.tsv"),
        help="Output summary TSV (default: results/benchmarks/pseudo_prospective_episode_summary.tsv)",
    )
    ap.add_argument(
        "--out-episodes",
        default=str(RESULTS_DIR / "benchmarks/pseudo_prospective_episode_log.tsv"),
        help="Output episode log TSV (default: results/benchmarks/pseudo_prospective_episode_log.tsv)",
    )
    ap.add_argument("--lookahead-weeks", type=int, default=4, help="Lookahead window in weeks (default: 4)")
    ap.add_argument(
        "--threshold-rules",
        default="train_quantile_0.85,train_quantile_0.90,train_quantile_0.95",
        help="Comma-separated threshold rules to simulate (default: train_quantile_0.85,train_quantile_0.90,train_quantile_0.95)",
    )
    ap.add_argument("--exclude-last-weeks", default="0,4", help="Comma-separated exclude_last_weeks values to include (default: 0,4)")
    args = ap.parse_args()
    cohort = get_cohort(args.cohort_profile)

    pred = pd.read_csv(args.predictions_long, sep="\t")
    if pred.empty:
        raise SystemExit("Empty predictions_long.")
    util = pd.read_csv(args.alert_utility, sep="\t")
    if util.empty:
        raise SystemExit("Empty alert_utility.")

    pred = pred[pred["train_scope"].astype(str) == "within_site"].copy()
    pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
    pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
    pred = pred.dropna(subset=["y_true", "y_pred", "origin_epiweek", "target_epiweek", "origin_week_ending_date"])

    excl_list = [int(x.strip()) for x in str(args.exclude_last_weeks).split(",") if x.strip()]
    pred = pred[pred["exclude_last_weeks"].isin(excl_list)].copy()

    # Restrict to mainline national outcomes for narrative clarity (site=Overall + cohort primary ages).
    pred = pred[(pred["site"].astype(str) == "Overall") & (pred["age_group"].isin(list(cohort.primary_respnet_ages)))].copy()
    if pred.empty:
        raise SystemExit("No predictions for site=Overall and cohort primary age group(s) after filtering.")

    thr_rules = [x.strip() for x in str(args.threshold_rules).split(",") if x.strip()]
    util = util[(util["threshold_rule"].astype(str).isin(thr_rules)) & (util["site"].astype(str) == "Overall")].copy()

    # We simulate per method and per outcome series.
    group_cols = ["exclude_last_weeks", "surveillance_network", "age_group", "site", "method"]
    rows_summary: list[dict[str, object]] = []
    rows_episode: list[dict[str, object]] = []

    # Precompute threshold values per outcome series (exclude_last_weeks, net, age, site).
    thr_map: dict[tuple[int, str, str, str, str], float] = {}
    for _, r in util.groupby(["exclude_last_weeks", "surveillance_network", "age_group", "site", "threshold_rule"]).head(1).iterrows():
        thr_map[
            (int(r["exclude_last_weeks"]), str(r["surveillance_network"]), str(r["age_group"]), str(r["site"]), str(r["threshold_rule"]))
        ] = float(r["threshold_value"])

    for key, g in pred.groupby(group_cols):
        excl, net, age, site, method = key

        # Build origin-week grid: max prediction over horizons 1..4 (policy lookahead window).
        gg = g[g["horizon_weeks"].isin([1, 2, 3, 4])].copy()
        if gg.empty:
            continue

        # For each origin, we also keep the list of future target_epiweeks (for onset matching).
        future_map: dict[int, list[int]] = {}
        for origin, hsub in gg.groupby("origin_epiweek"):
            future_map[int(origin)] = [int(x) for x in hsub.sort_values("horizon_weeks")["target_epiweek"].tolist()]

        origin_grid = (
            gg.groupby(["origin_epiweek", "origin_week_ending_date"], as_index=False)["y_pred"]
            .max()
            .rename(columns={"y_pred": "max_pred_1to4"})
            .sort_values("origin_epiweek")
            .reset_index(drop=True)
        )
        origin_grid["future_target_epiweeks"] = origin_grid["origin_epiweek"].map(lambda w: future_map.get(int(w), []))

        if len(origin_grid) < 30:
            continue

        # Define events and onset on the target-week grid (unique target epiweeks).
        tgt = gg.drop_duplicates(subset=["target_epiweek"])[["target_epiweek", "y_true"]].copy().sort_values("target_epiweek")
        y_tgt = pd.to_numeric(tgt["y_true"], errors="coerce").to_numpy(dtype=float)
        ok_tgt = np.isfinite(y_tgt)

        for thr_rule in thr_rules:
            thr_key = (int(excl), str(net), str(age), str(site), str(thr_rule))
            if thr_key not in thr_map:
                continue
            thr = float(thr_map[thr_key])

            alert = origin_grid["max_pred_1to4"].to_numpy(dtype=float) >= float(thr)

            event = (y_tgt >= float(thr)) & ok_tgt
            onset = onset_weeks(event)
            onset_weeks_set = set(tgt.loc[onset, "target_epiweek"].to_numpy(dtype=int).tolist())

            episodes = build_episode_log(origin_grid, alert, onset_weeks_set, lookahead_weeks=int(args.lookahead_weeks))
            if len(episodes) == 0:
                continue

            n_ep = int(len(episodes))
            n_true = int(sum(1 for e in episodes if e.true_episode))
            ppv_ep = float(n_true / n_ep) if n_ep > 0 else float("nan")
            # Interpretation note to prevent over-claiming when the event is rare.
            if n_ep <= 2:
                ppv_note = "Few alert episodes; PPV is unstable under this episode definition."
            elif n_true == 0:
                ppv_note = "PPV=0 with zero true episodes under this episode definition; interpret as no detected onsets during alerted periods (rare-event setting)."
            else:
                ppv_note = ""
            lead_list = [e.lead_weeks for e in episodes if e.true_episode and e.lead_weeks is not None]
            lead_med = float(np.median(lead_list)) if lead_list else float("nan")
            lead_mean = float(np.mean(lead_list)) if lead_list else float("nan")

            rows_summary.append(
                {
                    "exclude_last_weeks": int(excl),
                    "surveillance_network": str(net),
                    "age_group": str(age),
                    "site": str(site),
                    "method": str(method),
                    "threshold_rule": str(thr_rule),
                    "threshold_value": float(thr),
                    "n_origin_weeks": int(len(origin_grid)),
                    "alert_week_fraction": float(np.mean(alert.astype(float))),
                    "episodes_total": int(n_ep),
                    "episodes_true": int(n_true),
                    "episodes_false": int(n_ep - n_true),
                    "episode_ppv": float(ppv_ep),
                    "episode_ppv_note": str(ppv_note),
                    "lead_weeks_median_true": float(lead_med),
                    "lead_weeks_mean_true": float(lead_mean),
                }
            )

            for e in episodes:
                rows_episode.append(
                    {
                        "exclude_last_weeks": int(excl),
                        "surveillance_network": str(net),
                        "age_group": str(age),
                        "site": str(site),
                        "method": str(method),
                        "threshold_rule": str(thr_rule),
                        "threshold_value": float(thr),
                        "episode_start_epiweek": int(e.start_epiweek),
                        "episode_end_epiweek": int(e.end_epiweek),
                        "episode_start_week_ending_date": str(e.start_date),
                        "episode_end_week_ending_date": str(e.end_date),
                        "true_episode": int(1 if e.true_episode else 0),
                        "onset_epiweek": (int(e.onset_epiweek) if e.onset_epiweek is not None else np.nan),
                        "lead_weeks": (int(e.lead_weeks) if e.lead_weeks is not None else np.nan),
                    }
                )

    summary = pd.DataFrame(rows_summary).sort_values(["surveillance_network", "age_group", "exclude_last_weeks", "method"])
    episodes = pd.DataFrame(rows_episode).sort_values(["surveillance_network", "age_group", "exclude_last_weeks", "method", "episode_start_epiweek"])

    ensure_dir(Path(args.out_summary).parent)
    summary.to_csv(args.out_summary, sep="\t", index=False)
    episodes.to_csv(args.out_episodes, sep="\t", index=False)
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
