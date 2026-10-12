#!/usr/bin/env python3
"""Generate the five study figures from numerical surveillance and forecast results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
LABELS = {
    "AR": "Hospitalization history",
    "AR_ED": "+ ED",
    "AR_wILI": "+ wILI",
    "AR_lab": "+ Laboratory positivity",
    "AR_wILI_lab": "+ wILI + laboratory positivity",
    "AR_wILI_ED": "+ wILI + ED",
    "AR_all": "+ wILI + laboratory positivity + ED",
    "AR_ED_lab": "+ Laboratory positivity + ED",
    "AR_lab_ED": "+ Laboratory positivity + ED",
}
ORDER = list(LABELS)
NETS = [("FluSurv-NET", "Influenza", BLUE), ("RSV-NET", "RSV", GREEN)]
SOURCE_NAMES = {"wili": "wILI", "flu_positivity": "Laboratory Influenza",
                "rsv_positivity": "Laboratory RSV", "ed_influenza_pct_65p": "ED influenza",
                "ed_rsv_pct_65p": "ED RSV"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2, "legend.fontsize": 8.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def export(fig, out, name):
    fig.savefig(out / f"{name}.pdf", facecolor="white")
    fig.savefig(out / f"{name}.png", facecolor="white", dpi=300)
    plt.close(fig)


def net_rows(df, net, column="surveillance_network"):
    synonyms = {"FluSurv-NET": ["FluSurv-NET", "Influenza", "influenza", "flu"],
                "RSV-NET": ["RSV-NET", "RSV", "rsv"]}
    return df[df[column].isin(synonyms[net])].copy()


def overview(analysis, out):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.1), sharex=True)
    for i, (net, title, color) in enumerate(NETS):
        ax = axes[i]
        g = net_rows(analysis, net).set_index("week_ending_date").sort_index()
        dates = pd.date_range(g.index.min(), g.index.max(), freq="W-SAT")
        # Explicitly retain missing calendar weeks: lines cannot bridge surveillance gaps.
        g = g.reindex(dates)
        ax.plot(g.index, g.rate_per_100k, color=color, lw=1.2)
        ax.set_title(f"{'AB'[i]}  {title}", loc="left", fontweight="bold", pad=7)
        ax.set_ylabel("Hospitalizations\nper 100,000")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=".9", lw=.5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Week ending date")
    fig.subplots_adjust(left=.14, right=.98, top=.92, bottom=.13, hspace=.35)
    export(fig, out, "Fig1_time_series_overview")


def weekly_segments(ax, dates, y, color, width=6):
    if not len(dates):
        return
    x = np.unique(mdates.date2num(pd.to_datetime(dates)))
    # Join genuinely consecutive observed weeks, while retaining every missing-week gap.
    groups = np.split(x, np.flatnonzero(np.diff(x) > 7.01) + 1)
    segments = [((v[0] - 3.5, y), (v[-1] + 3.5, y)) for v in groups]
    ax.add_collection(LineCollection(segments, colors=color, linewidths=width,
                                    capstyle="butt"))


def availability(analysis, source, out):
    xmin, xmax = analysis.week_ending_date.min(), analysis.week_ending_date.max()
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.7), sharex=True)
    rows_for_export = []
    for i, (net, title, color) in enumerate(NETS):
        ax = axes[i]
        g = net_rows(analysis, net).dropna(subset=["rate_per_100k"])
        outcome_dates = set(g.week_ending_date)
        ed = "ed_influenza_pct_65p" if net == "FluSurv-NET" else "ed_rsv_pct_65p"
        lab = "flu_positivity" if net == "FluSurv-NET" else "rsv_positivity"
        wili_label = "wILI (all ages; coverage only)" if net == "RSV-NET" else "wILI (all ages)"
        variables = [(None, "Hospitalizations (≥65 years)"), (ed, "ED visits (≥65 years)"),
                     (lab, "Laboratory positivity (all ages)"), ("wili", wili_label)]
        for j, (var, label) in enumerate(variables):
            if var is None:
                raw_dates = g.week_ending_date
                overlapping = raw_dates
            else:
                s = source[(source.signal.isin([var, SOURCE_NAMES[var]])) & source.value.notna()]
                raw_dates = s.week_ending_date.drop_duplicates()
                raw_dates = raw_dates[(raw_dates >= xmin) & (raw_dates <= xmax)]
                overlapping = raw_dates[raw_dates.isin(outcome_dates)]
                weekly_segments(ax, raw_dates, j, "#d7dce0")
            weekly_segments(ax, overlapping, j, color if var else "#343434")
            ax.text(1.02, j, str(len(overlapping)), transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8.3)
            rows_for_export.append({"outcome": title, "signal": var or "hospitalization",
                                    "n_source_in_display_window": len(raw_dates),
                                    "n_overlap": len(overlapping),
                                    "outcome_n": len(g)})
        ax.text(1.02, 1.15, "Weeks", transform=ax.transAxes, fontsize=8.3)
        ax.set_title(f"{'AB'[i]}  {title}", loc="left", fontweight="bold", pad=8)
        ax.set_yticks(range(4), [v[1] for v in variables])
        ax.set_ylim(3.6, -.6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(xmin - pd.Timedelta(days=10), xmax + pd.Timedelta(days=10))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(axis="x", color=".93", lw=.5)
    axes[-1].set_xlabel("Week ending date")
    fig.legend(handles=[Line2D([], [], color="#d7dce0", lw=5, label="Indicator observed only"),
                        Line2D([], [], color="#343434", lw=5, label="Outcome observed"),
                        Line2D([], [], color=BLUE, lw=5, label="Influenza indicator + outcome"),
                        Line2D([], [], color=GREEN, lw=5, label="RSV indicator + outcome")],
               loc="lower center", bbox_to_anchor=(.62, .025), ncol=2, frameon=False,
               columnspacing=1.1, handlelength=1.7, handletextpad=.5, labelspacing=.35,
               borderaxespad=0, fontsize=7.6)
    fig.subplots_adjust(left=.34, right=.90, top=.91, bottom=.23, hspace=.45)
    export(fig, out, "Fig2_weekly_availability")
    pd.DataFrame(rows_for_export).to_csv(out / "Fig2_weekly_availability_counts.tsv", sep="\t", index=False)


def common_comparisons(comparisons, out):
    c = comparisons[(comparisons.exclude_last_weeks == 4) &
                    (comparisons.horizon_weeks == 1) &
                    (comparisons.block_len == 4) &
                    (comparisons.inference_method == "calendar_segment_moving_block_bootstrap") &
                    (comparisons.analysis_set == "common_modern") &
                    (comparisons.comparator_model == "AR")].copy()
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2))
    for i, (net, title, color) in enumerate(NETS):
        ax = axes[i]
        g = net_rows(c, net, "outcome")
        g["order"] = g.candidate_model.map({v: j for j, v in enumerate(ORDER)})
        g = g.sort_values("order")
        if not len(g):
            raise ValueError(f"No common-origin comparisons for {net}")
        y = np.arange(len(g))
        ax.axvline(0, color=".5", lw=.8, ls="--")
        for j, (_, r) in enumerate(g.iterrows()):
            ax.plot([r.ci_lower, r.ci_upper], [j, j], color=color, lw=1.2)
            ax.plot(r.loss_difference, j, marker="o", ms=4.5, color=color)
        ax.set_yticks(y, [LABELS[v] for v in g.candidate_model])
        ax.set_ylim(len(g) - .5, -.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{'AB'[i]}  {title} (n = {int(g.matched_n.iloc[0])} forecast weeks)",
                     loc="left", fontweight="bold", pad=8)
        ax.grid(axis="x", color=".92", lw=.5)
        ax.set_xlabel("Reduction in mean absolute error\n(hospitalizations per 100,000)")
    fig.subplots_adjust(left=.40, right=.97, top=.92, bottom=.15, hspace=.60)
    export(fig, out, "Fig3_common_origin_comparisons")
    c.to_csv(out / "Fig3_comparison_data.tsv", sep="\t", index=False)


def descriptive_lag(analysis, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), sharey=True)
    records = []
    for i, (net, title, _) in enumerate(NETS):
        ax = axes[i]
        g = net_rows(analysis, net)
        g = g[g.week_ending_date >= "2022-10-01"].set_index("week_ending_date").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="W-SAT"))
        ed = "ed_influenza_pct_65p" if i == 0 else "ed_rsv_pct_65p"
        lab = "flu_positivity" if i == 0 else "rsv_positivity"
        sigs = [(ed, "ED visits", BLUE, "o"), (lab, "Laboratory positivity", GREEN, "s"),
                ("wili", "wILI", ORANGE, "^")]
        for signal, label, color, marker in sigs:
            vals = []
            ns = []
            for lag in range(9):
                pair = pd.concat([g.rate_per_100k, g[signal].shift(lag)], axis=1).dropna()
                r = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 20 else np.nan
                vals.append(r)
                ns.append(len(pair))
                records.append({"outcome": title, "signal": signal, "lag_weeks": lag,
                                "n": len(pair), "correlation": r})
            ax.plot(range(9), vals, color=color, marker=marker, ms=3, lw=1,
                    label=f"{label} (n = {min(ns)}–{max(ns)})")
        ax.set_title(f"{'AB'[i]}  {title}", loc="left", fontweight="bold")
        ax.set_xlabel("Indicator lead (calendar weeks)")
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.set_ylim(-.15, 1.03)
        ax.grid(axis="y", color=".91", lw=.5)
        ax.legend(loc="lower left", frameon=False, fontsize=8.1)
    axes[0].set_ylabel("Pearson correlation")
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.90, wspace=.20)
    export(fig, out, "FigS1_descriptive_lag")
    pd.DataFrame(records).to_csv(out / "FigS1_descriptive_lag_data.tsv", sep="\t", index=False)


def conditional_comparisons(comparisons, out):
    c = comparisons[(comparisons.exclude_last_weeks == 4) &
                    (comparisons.block_len == 4) &
                    (comparisons.inference_method == "calendar_segment_moving_block_bootstrap") &
                    (comparisons.analysis_set == "common_modern") &
                    (comparisons.comparator_model != "AR")].copy()
    wanted = [("AR_wILI", "AR_wILI_lab", "Laboratory positivity given wILI"),
              ("AR_wILI", "AR_wILI_ED", "ED given wILI"),
              ("AR_lab", "AR_all", "ED given laboratory positivity"),
              ("AR_lab", "AR_ED_lab", "ED given laboratory positivity"),
              ("AR_lab", "AR_lab_ED", "ED given laboratory positivity"),
              ("AR_wILI_lab", "AR_all", "ED given wILI + laboratory positivity"),
              ("AR_ED", "AR_all", "Laboratory positivity given ED"),
              ("AR_ED", "AR_ED_lab", "Laboratory positivity given ED"),
              ("AR_ED", "AR_lab_ED", "Laboratory positivity given ED")]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.7))
    colors, markers = [BLUE, GREEN, ORANGE, PURPLE], ["o", "s", "^", "D"]
    used = []
    for i, (net, title, _) in enumerate(NETS):
        ax = axes[i]
        g = net_rows(c, net, "outcome")
        contrasts = []
        for comparator, candidate, label in wanted:
            # In influenza AR_all includes wILI and cannot isolate ED given lab alone.
            if i == 0 and (comparator, candidate) in [("AR_lab", "AR_all"), ("AR_ED", "AR_all")]:
                continue
            sub = g[(g.comparator_model == comparator) & (g.candidate_model == candidate)]
            if len(sub):
                contrasts.append((label, sub))
        if not contrasts:
            raise ValueError(f"No conditional comparisons for {net}")
        ax.axvline(0, color=".5", lw=.8, ls="--")
        for j, (label, sub) in enumerate(contrasts):
            for _, r in sub.iterrows():
                h = int(r.horizon_weeks)
                y = j + (h - 2.5) * .12
                ax.plot([r.ci_lower, r.ci_upper], [y, y], lw=1, color=colors[h-1])
                ax.plot(r.loss_difference, y, marker=markers[h-1], ms=3.8, color=colors[h-1])
                used.append(r.to_dict())
        ax.set_yticks(range(len(contrasts)), [v[0] for v in contrasts])
        ax.set_ylim(len(contrasts) - .5, -.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{'AB'[i]}  {title}", loc="left", fontweight="bold", pad=7)
        ax.set_xlabel("Reduction in mean absolute error (per 100,000)")
        ax.grid(axis="x", color=".92", lw=.5)
    fig.legend(handles=[Line2D([], [], color=colors[h-1], marker=markers[h-1], ms=4,
                               label=f"{h} week{'s' if h > 1 else ''}") for h in range(1, 5)],
               loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=.44, right=.98, top=.90, bottom=.17, hspace=.65)
    export(fig, out, "FigS2_conditional_comparisons")
    pd.DataFrame(used).to_csv(out / "FigS2_comparison_data.tsv", sep="\t", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--out", type=Path,
                   default=ROOT / "figures")
    p.add_argument("--descriptive-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()
    analysis = pd.read_csv(a.results / "analysis_table.tsv", sep="\t", parse_dates=["week_ending_date"])
    analysis = analysis[(analysis.site == "Overall") & (analysis.age_group == "65+ yr")]
    source = pd.read_csv(a.results / "source_weekly.tsv", sep="\t", parse_dates=["week_ending_date"])
    overview(analysis, a.out)
    availability(analysis, source, a.out)
    descriptive_lag(analysis, a.out)
    if not a.descriptive_only:
        comparisons = pd.read_csv(a.results / "comparisons.tsv", sep="\t")
        common_comparisons(comparisons, a.out)
        conditional_comparisons(comparisons, a.out)
    print(f"Saved study figures to {a.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
