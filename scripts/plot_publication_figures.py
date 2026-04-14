#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_profiles import cohort_from_env, get_cohort
from utils import RESULTS_DIR, ensure_dir


def setup_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "Arial, Helvetica, sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
        }
    )


def method_palette() -> dict[str, tuple[float, float, float]]:
    """Okabe-Ito colorblind-safe palette, mapped to methods for consistency across figures."""
    # Okabe-Ito: orange, sky blue, bluish green, yellow, blue, vermilion, reddish purple, black
    return {
        "seasonal_naive": (0.35, 0.35, 0.35),       # black (baseline)
        "ridge_univariate": (0.00, 0.45, 0.70),      # blue
        "ridge_with_ed": (1.00, 0.50, 0.05),          # orange
        "ridge_with_signals": (0.00, 0.62, 0.45),     # bluish green
        "ridge_with_signals_plus_ed": (0.84, 0.15, 0.16),  # vermilion
        "ridge_with_signals_plus_flu_pos": (0.58, 0.40, 0.74),  # reddish purple
        "ridge_with_signals_plus_flu_pos_plus_ed": (0.95, 0.91, 0.26),  # yellow (dark outline needed)
    }


def nice_method_label(m: str) -> str:
    return {
        "seasonal_naive": "Seasonal baseline",
        "ridge_univariate": "AR seasonal model",
        "ridge_with_ed": "ED-integrated model",
        "ridge_with_signals": "Public-signal model",
        "ridge_with_signals_plus_ed": "Public-signal + ED model",
        "ridge_with_signals_plus_flu_pos": "Public-signal + flu positivity model",
        "ridge_with_signals_plus_flu_pos_plus_ed": "Public-signal + flu positivity + ED model",
    }.get(m, m)


def method_linestyle(m: str) -> str | tuple[int, tuple[int, ...]]:
    # Use distinct line styles so overlapping curves remain distinguishable in print.
    return {
        "ridge_univariate": "-",
        "ridge_with_ed": "--",
        "ridge_with_signals": ":",
        "ridge_with_signals_plus_ed": "-.",
        "ridge_with_signals_plus_flu_pos_plus_ed": (0, (3, 1, 1, 1)),
    }.get(m, "-")


def method_marker(m: str) -> str:
    return {
        "ridge_univariate": "o",
        "ridge_with_ed": "s",
        "ridge_with_signals": "^",
        "ridge_with_signals_plus_ed": "D",
        "ridge_with_signals_plus_flu_pos_plus_ed": "v",
    }.get(m, "o")

def nice_age_label(age_group: str) -> str:
    ag = str(age_group or "").strip()
    return {
        "65+ yr": "Age 65 years and older",
        "65-74 yr": "Age 65–74 years",
        "75-84 yr": "Age 75–84 years",
        "85+ yr": "Age 85 years and older",
        "Overall": "All ages",
    }.get(ag, ag)


def fig1_time_series_overview(time_series: pd.DataFrame, outdir: Path, cohort, *, write_png: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.dates as mdates

    keep = time_series.copy()
    keep = keep[keep["site"].astype(str) == "Overall"]
    keep = keep[keep["surveillance_network"].isin(["FluSurv-NET", "RSV-NET"])]
    keep = keep[keep["age_group"].isin(list(cohort.primary_respnet_ages))]
    keep = keep.sort_values("week_ending_date")
    keep["week_ending_date"] = pd.to_datetime(keep["week_ending_date"], errors="coerce")
    keep["rate_per_100k"] = pd.to_numeric(keep["rate_per_100k"], errors="coerce")

    outcomes = ["FluSurv-NET", "RSV-NET"]
    ages = list(cohort.primary_respnet_ages)

    pdf_path = outdir / "Fig1_time_series_overview.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, len(ages), figsize=(7.2, 3.6 + 1.3 * len(ages)), sharex=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(2, 1)
        for i, net in enumerate(outcomes):
            for j, age in enumerate(ages):
                ax = axes[i, j]
                g = keep[(keep["surveillance_network"] == net) & (keep["age_group"] == age)].copy()
                if g.empty:
                    ax.set_axis_off()
                    continue
                ax.plot(g["week_ending_date"], g["rate_per_100k"], color=(0.00, 0.45, 0.70), linewidth=1.4)
                # Panel label
                ax.text(-0.12, 1.04, "AB"[i], transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="bottom")
                ax.set_title(f"{net} · {nice_age_label(age)}", loc="left", pad=6)
                ax.set_ylabel("Rate per 100,000")
                ax.grid(axis="y", alpha=0.2)
                ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                # Rotate x-tick labels on bottom row only
                if i == 1:
                    for label in ax.get_xticklabels():
                        label.set_rotation(45)
                        label.set_ha("right")
                    ax.set_xlabel("Year")
        fig.suptitle(str(cohort.figure_suptitle), y=0.99, fontsize=9.5)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        pdf.savefig(fig)
        if write_png:
            png_path = outdir / "Fig1_time_series_overview.png"
            fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)


def _signal_coverage_rows(coverage: pd.DataFrame, cohort) -> tuple[pd.DataFrame, str]:
    keep = coverage.copy()
    keep = keep[keep["site"].astype(str) == "Overall"]
    # Use one canonical slice: RSV-NET for the cohort exemplar age highlights coverage limits clearly.
    keep = keep[(keep["surveillance_network"] == "RSV-NET") & (keep["age_group"] == str(cohort.key_exemplar_age))]
    if keep.empty:
        raise SystemExit("Signal coverage: no rows after filtering (expected RSV-NET cohort exemplar age, site=Overall).")

    keep["first_week_ending_date"] = pd.to_datetime(keep["first_week_ending_date"], errors="coerce")
    keep["last_week_ending_date"] = pd.to_datetime(keep["last_week_ending_date"], errors="coerce")

    # Predeclare key signals for feasibility disclosure (avoid appearance of cherry-picking).
    if cohort.profile_id == "older_adult_65plus":
        variables = [
            ("wili", "wILI (FluView)"),
            ("flu_positivity", "Flu test positivity"),
            ("ed_influenza_pct_65p", "ED % influenza\n(age 65 years and older)"),
            ("ed_rsv_pct_65p", "ED % RSV\n(age 65 years and older)"),
            ("rsv_positivity", "RSV test positivity (NREVSS)"),
        ]
        title_slice = "RSV-NET (age 65 years and older)"
    elif cohort.profile_id == "older_adult_strata":
        variables = [
            ("wili", "wILI (FluView)"),
            ("flu_positivity", "Flu test positivity (CDC)"),
            ("rsv_positivity", "RSV test positivity (NREVSS; sensitivity)"),
        ]
        title_slice = "RSV-NET older-adult strata (no ED proxying)"
    else:
        raise SystemExit(f"Signal coverage: unsupported cohort profile_id={cohort.profile_id!r}")

    rows = []
    for var, label in variables:
        sub = keep[keep["variable"] == var]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "variable": var,
                "label": label,
                "start": r["first_week_ending_date"],
                "end": r["last_week_ending_date"],
                "n": int(r["n_nonmissing"]),
            }
        )
    if not rows:
        raise SystemExit("Signal coverage: none of the expected variables were found in coverage report.")

    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    return df, title_slice


def _plot_signal_coverage(ax, coverage: pd.DataFrame, cohort) -> None:
    import matplotlib.dates as mdates

    df, title_slice = _signal_coverage_rows(coverage, cohort)
    y = np.arange(len(df))
    for i, row in df.iterrows():
        ax.plot([row["start"], row["end"]], [i, i], linewidth=7, solid_capstyle="butt",
                color=(0.00, 0.62, 0.45))
        pad = pd.Timedelta(days=14)
        ax.text(row["end"] + pad, i, f"n={row['n']}", va="center", ha="left",
                fontsize=6.5, color=(0.25, 0.25, 0.25))
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=7)
    ax.set_xlabel("Week ending date")
    ax.set_title("Signal availability windows", loc="left", pad=4)
    ax.grid(axis="x", alpha=0.15)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    xmin = pd.to_datetime(df["start"].min()) - pd.Timedelta(days=30)
    xmax = pd.to_datetime(df["end"].max()) + pd.Timedelta(days=90)
    ax.set_xlim(xmin, xmax)


def fig2_lag_and_coverage_board(lag_analysis: pd.DataFrame, coverage: pd.DataFrame, outdir: Path, cohort, *, write_png: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    required_cols = {"age_group", "surveillance_network", "signal", "lag_weeks", "corr"}
    if lag_analysis is None or not required_cols.issubset(set(lag_analysis.columns)):
        lag = pd.DataFrame()
    else:
        lag = lag_analysis.copy()
        lag = lag[lag["age_group"].astype(str) == str(cohort.key_exemplar_age)].copy()
        lag = lag[lag["surveillance_network"].isin(["FluSurv-NET", "RSV-NET"])].copy()
        lag["lag_weeks"] = pd.to_numeric(lag["lag_weeks"], errors="coerce")
        lag["corr"] = pd.to_numeric(lag["corr"], errors="coerce")
        lag = lag.dropna(subset=["lag_weeks", "corr"])

    signals = ["rsv_positivity", "wili"]
    sig_label = {"rsv_positivity": "RSV test positivity", "wili": "wILI"}
    sig_color = {"rsv_positivity": (0.00, 0.45, 0.70), "wili": (0.84, 0.15, 0.16)}

    pdf_path = outdir / "Fig2_lag_and_coverage.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(7.2, 4.5))
        outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)

        left = outer[0, 0].subgridspec(2, 1, hspace=0.35)
        ax_flu = fig.add_subplot(left[0, 0])
        ax_rsv = fig.add_subplot(left[1, 0], sharex=ax_flu, sharey=ax_flu)

        if lag.empty:
            for ax, net, row_title in [(ax_flu, "FluSurv-NET", None), (ax_rsv, "RSV-NET", None)]:
                ax.axhline(0.0, color=(0.6, 0.6, 0.6), linewidth=1.0)
                ax.set_title(row_title if row_title else f"{net} · {nice_age_label(cohort.key_exemplar_age)}", loc="left", pad=4)
                if row_title:
                    ax.text(-0.12, 1.04, "AB"[0], transform=ax.transAxes,
                            fontsize=9, fontweight="bold", va="bottom")
                ax.text(
                    0.02, 0.50,
                    "Lag-analysis table missing for this cohort run.\nRe-run: python3 scripts/lag_analysis.py",
                    transform=ax.transAxes, ha="left", va="center", fontsize=7, color=(0.4, 0.4, 0.4),
                )
                ax.set_ylabel("Correlation coefficient (r)")
                ax.grid(axis="y", alpha=0.15)
        else:
            for ax_idx, (ax, net, title) in enumerate([(ax_flu, "FluSurv-NET", None), (ax_rsv, "RSV-NET", None)]):
                g = lag[lag["surveillance_network"].astype(str) == net]
                if g.empty:
                    ax.set_axis_off()
                    continue
                for s in signals:
                    gg = g[g["signal"].astype(str) == s].sort_values("lag_weeks")
                    if gg.empty:
                        continue
                    x = gg["lag_weeks"].to_numpy(dtype=float)
                    y = gg["corr"].to_numpy(dtype=float)
                    ax.plot(
                        x, y, marker="o", markersize=3.5, linewidth=1.6,
                        color=sig_color.get(s, (0.2, 0.2, 0.2)),
                        label=sig_label.get(s, s),
                    )
                ax.axhline(0.0, color=(0.6, 0.6, 0.6), linewidth=0.8)
                if ax_idx == 0:
                    ax.text(-0.12, 1.04, "A", transform=ax.transAxes,
                            fontsize=9, fontweight="bold", va="bottom")
                else:
                    ax.text(-0.12, 1.04, "B", transform=ax.transAxes,
                            fontsize=9, fontweight="bold", va="bottom")
                ax.set_title(f"{net} · {nice_age_label(cohort.key_exemplar_age)}", loc="left", pad=4)
                ax.set_ylabel("Correlation coefficient (r)")
                ax.grid(axis="y", alpha=0.15)
        ax_rsv.set_xlabel("Lag (weeks)")
        handles, labels = ax_flu.get_legend_handles_labels()
        if handles:
            ax_flu.legend(handles, labels, loc="upper right", frameon=False, fontsize=7)

        ax_cov = fig.add_subplot(outer[0, 1])
        _plot_signal_coverage(ax_cov, coverage=coverage, cohort=cohort)
        ax_cov.text(-0.12, 1.04, "C", transform=ax_cov.transAxes,
                     fontsize=9, fontweight="bold", va="bottom")

        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14, wspace=0.32)
        pdf.savefig(fig)
        if write_png:
            fig.savefig(outdir / "Fig2_lag_and_coverage.png", bbox_inches="tight")
        plt.close(fig)


def _paired_benchmark_panel(paired: pd.DataFrame, outdir: Path, cohort, *, exclude_last_weeks: int, stem: str, write_png: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    palette = method_palette()

    def xtick_label(m: str) -> str:
        return {
            "seasonal_naive": "Seasonal\nbaseline",
            "ridge_univariate": "AR seasonal\nmodel",
            "ridge_with_ed": "ED-integrated\nmodel",
            "ridge_with_signals": "Public-signal\nmodel",
            "ridge_with_signals_plus_ed": "Public-signal\n+ ED model",
            "ridge_with_signals_plus_flu_pos": "Public-signal\n+ flu positivity",
            "ridge_with_signals_plus_flu_pos_plus_ed": "Public-signal\n+ flu pos + ED",
        }.get(m, nice_method_label(m))

    keep = paired.copy()
    keep = keep[keep["site"].astype(str) == "Overall"]
    keep = keep[keep["age_group"].isin(list(cohort.primary_respnet_ages))]
    keep = keep[keep["surveillance_network"].isin(["FluSurv-NET", "RSV-NET"])]
    keep = keep[(keep["exclude_last_weeks"] == int(exclude_last_weeks)) & (keep["metric"] == "MAE")]
    keep = keep[keep["horizon_weeks"].isin([1, 2])]
    keep = keep[keep["train_scope"].astype(str) == "within_site"]

    methods = ["ridge_univariate", "ridge_with_ed", "ridge_with_signals", "ridge_with_signals_plus_ed"]

    outcomes = ["FluSurv-NET", "RSV-NET"]
    ages = list(cohort.primary_respnet_ages)

    pdf_path = outdir / f"{stem}.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, len(ages), figsize=(7.2, 3.4 + 1.3 * len(ages)), sharey=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(2, 1)
        for i, net in enumerate(outcomes):
            for j, age in enumerate(ages):
                ax = axes[i, j]
                g = keep[(keep["surveillance_network"] == net) & (keep["age_group"] == age) & (keep["horizon_weeks"] == 1)].copy()
                if g.empty:
                    ax.set_axis_off()
                    continue
                g = g[g["method"].isin(methods)].copy()
                g["method"] = pd.Categorical(g["method"], categories=methods, ordered=True)
                g = g.sort_values("method")
                x = np.arange(len(g))
                y = pd.to_numeric(g["improvement_over_baseline"], errors="coerce").to_numpy(dtype=float)
                lo = pd.to_numeric(g["improvement_ci_lo"], errors="coerce").to_numpy(dtype=float)
                hi = pd.to_numeric(g["improvement_ci_hi"], errors="coerce").to_numpy(dtype=float)
                err = np.vstack([y - lo, hi - y])
                colors = [palette.get(m, (0.0, 0.0, 0.0)) for m in g["method"].astype(str).tolist()]
                ax.bar(x, y, yerr=err, color=colors, alpha=0.9, capsize=3, edgecolor="white", linewidth=0.5)
                ax.axhline(0.0, color=(0.4, 0.4, 0.4), linewidth=0.8, linestyle="--")
                ax.set_xticks(x)
                ax.set_xticklabels([xtick_label(m) for m in g["method"].astype(str).tolist()],
                                   rotation=0, ha="center", fontsize=7)
                ax.tick_params(axis="x", labelsize=7)
                # Panel label
                ax.text(-0.12, 1.04, "AB"[i], transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="bottom")
                ax.set_title(f"{net} · {nice_age_label(age)} · One-week-ahead", loc="left", pad=4)
                if j == 0:
                    ax.set_ylabel("Reduction in forecast error (MAE)\n(hospitalizations per 100,000 per week)\nvs seasonal baseline")
                ax.grid(axis="y", alpha=0.15)
                if i == 0 and j == 0:
                    subtitle = "Finalized-data cutoff (exclude last 4 weeks)" if int(exclude_last_weeks) == 4 else "Approximate real-time (no exclusion)"
                    ax.text(0.0, 1.16, subtitle, transform=ax.transAxes, fontsize=8, color=(0.25, 0.25, 0.25))
        fig.tight_layout(rect=(0, 0.06, 1, 0.96))
        pdf.savefig(fig)
        if write_png:
            png_path = outdir / f"{stem}.png"
            fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)


def fig3_paired_benchmark(paired: pd.DataFrame, outdir: Path, cohort, *, write_png: bool) -> None:
    _paired_benchmark_panel(paired, outdir, cohort, exclude_last_weeks=4, stem="Fig3_paired_benchmark", write_png=write_png)


def figS5_paired_benchmark_realtime(paired: pd.DataFrame, outdir: Path, cohort, *, write_png: bool) -> None:
    _paired_benchmark_panel(paired, outdir, cohort, exclude_last_weeks=0, stem="FigS5_paired_benchmark_realtime", write_png=write_png)

def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    sx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    sy = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if sx.size < 3:
        return float("nan")
    return float(np.corrcoef(sx, sy)[0, 1])


def _moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if block_len <= 1 or n < 2:
        return rng.integers(0, n, size=n, dtype=int)
    block_len = min(block_len, n)
    starts = rng.integers(0, n - block_len + 1, size=int(np.ceil(n / block_len)), dtype=int)
    idx = np.concatenate([np.arange(s, s + block_len, dtype=int) for s in starts])[:n]
    return idx


def _circular_shift_pvalue(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, reps: int = 5000) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(x.size)
    if n < 8:
        return float("nan")
    r_obs = _spearman_r(x, y)
    if not np.isfinite(r_obs):
        return float("nan")
    rs = np.empty(int(reps), dtype=float)
    for i in range(int(reps)):
        k = int(rng.integers(1, n))
        rs[i] = _spearman_r(np.roll(x, k), y)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return float("nan")
    p = float(np.mean(np.abs(rs) >= abs(r_obs)))
    return max(p, 1.0 / float(rs.size))


def _loyo_week_median_expected(df: pd.DataFrame, value_col: str, *, year_col: str, week_col: str) -> pd.Series:
    base = df[[year_col, week_col, value_col]].copy()
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce")
    base = base.dropna(subset=[year_col, week_col, value_col])
    if base.empty:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)

    pairs = df[[year_col, week_col]].drop_duplicates().copy()
    exp_rows: list[dict[str, object]] = []
    for r in pairs.itertuples(index=False):
        y = int(getattr(r, year_col))
        w = int(getattr(r, week_col))
        other = base[(base[week_col] == w) & (base[year_col] != y)][value_col]
        exp_rows.append({year_col: y, week_col: w, f"{value_col}_seasonal_expected": float(other.median()) if len(other) else np.nan})
    exp = pd.DataFrame(exp_rows)
    out = df[[year_col, week_col]].merge(exp, on=[year_col, week_col], how="left")[f"{value_col}_seasonal_expected"]
    out.index = df.index
    return pd.to_numeric(out, errors="coerce")


def figS4_incremental_value_beyond_seasonality(time_series: pd.DataFrame, outdir: Path, cohort, *, write_png: bool) -> None:
    """
    Reviewer-facing diagnostic figure:
      Panel A: deviations from seasonal expectation (ED vs hospitalization).
      Panel B: year × ED-deviation tertile heatmap of hospitalization deviation.

    Primary slice: RSV-NET, adults aged 65 years and older (national aggregation), modern syndromic surveillance era weeks where ED signal is observed.
    Seasonal expectation is leave-one-year-out median by MMWR week within the modern syndromic surveillance era slice.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm

    if cohort.profile_id != "older_adult_65plus":
        return

    keep = time_series.copy()
    keep = keep[
        (keep["site"].astype(str) == "Overall")
        & (keep["surveillance_network"].astype(str) == "RSV-NET")
        & (keep["age_group"].astype(str) == "65+ yr")
    ].copy()
    if keep.empty:
        return

    keep["week_ending_date"] = pd.to_datetime(keep["week_ending_date"], errors="coerce")
    keep["epiweeks"] = pd.to_numeric(keep["epiweeks"], errors="coerce")
    keep["rate_per_100k"] = pd.to_numeric(keep["rate_per_100k"], errors="coerce")
    keep["ed_rsv_pct_65p"] = pd.to_numeric(keep.get("ed_rsv_pct_65p"), errors="coerce")

    keep = keep.dropna(subset=["week_ending_date", "epiweeks", "rate_per_100k", "ed_rsv_pct_65p"]).sort_values("week_ending_date")
    if len(keep) < 30:
        return

    keep["mmwr_year"] = (keep["epiweeks"] // 100).astype(int)
    keep["mmwr_week"] = (keep["epiweeks"] % 100).astype(int)
    # Align with baseline convention: week 53 → week 52 fallback.
    keep["mmwr_week_season"] = keep["mmwr_week"].where(keep["mmwr_week"] != 53, 52).astype(int)

    keep["ed_expected"] = _loyo_week_median_expected(keep, "ed_rsv_pct_65p", year_col="mmwr_year", week_col="mmwr_week_season")
    keep["hosp_expected"] = _loyo_week_median_expected(keep, "rate_per_100k", year_col="mmwr_year", week_col="mmwr_week_season")
    keep["ed_dev"] = keep["ed_rsv_pct_65p"] - keep["ed_expected"]
    keep["hosp_dev"] = keep["rate_per_100k"] - keep["hosp_expected"]
    keep = keep.dropna(subset=["ed_dev", "hosp_dev"]).copy()
    if len(keep) < 30:
        return

    # Early-warning alignment: ED deviation at week t vs hospitalization deviation at week t+1.
    keep = keep.sort_values("week_ending_date").reset_index(drop=True)
    next_date = keep["week_ending_date"].shift(-1)
    contiguous = (next_date - keep["week_ending_date"]).dt.days == 7
    keep["hosp_dev_h1"] = keep["hosp_dev"].shift(-1).where(contiguous)
    keep["target_week_ending_date_h1"] = next_date.where(contiguous)
    keep = keep.dropna(subset=["hosp_dev_h1"]).copy()
    if len(keep) < 30:
        return

    # Save figure source table for auditability.
    fig_src = keep[
        [
            "week_ending_date",
            "target_week_ending_date_h1",
            "epiweeks",
            "mmwr_year",
            "mmwr_week",
            "ed_rsv_pct_65p",
            "ed_expected",
            "ed_dev",
            "rate_per_100k",
            "hosp_expected",
            "hosp_dev",
            "hosp_dev_h1",
        ]
    ].copy()
    fig_src.to_csv(RESULTS_DIR / "figures" / "seasonal_deviation_scatter_rsv65p.tsv", sep="\t", index=False)

    x = keep["ed_dev"].to_numpy(dtype=float)
    y = keep["hosp_dev_h1"].to_numpy(dtype=float)

    r_s = _spearman_r(x, y)
    rng = np.random.default_rng(12345)
    p_shift = _circular_shift_pvalue(x, y, rng=rng, reps=5000)

    # Moving-block bootstrap CI for Spearman r (descriptive; preserves local autocorrelation).
    r_boot = []
    for _ in range(800):
        idx = _moving_block_bootstrap_indices(len(x), block_len=4, rng=rng)
        r_boot.append(_spearman_r(x[idx], y[idx]))
    r_boot = np.asarray([v for v in r_boot if np.isfinite(v)], dtype=float)
    r_lo, r_hi = (np.nan, np.nan)
    if r_boot.size >= 50:
        r_lo, r_hi = np.quantile(r_boot, [0.025, 0.975]).tolist()

    # Reviewer-facing summary row (small, stable table for manuscript text).
    summary_path = RESULTS_DIR / "figures" / "seasonal_deviation_scatter_summary_rsv65p.tsv"
    pd.DataFrame(
        [
            {
                "surveillance_network": "RSV-NET",
                "age_group": "65+ yr",
                "site": "Overall",
                "horizon_weeks": 1,
                "n_weeks": int(len(x)),
                "spearman_r": float(r_s),
                "spearman_r_ci_lo": float(r_lo) if np.isfinite(r_lo) else np.nan,
                "spearman_r_ci_hi": float(r_hi) if np.isfinite(r_hi) else np.nan,
                "shift_permutation_p": float(p_shift) if np.isfinite(p_shift) else np.nan,
                "notes": "ED deviation at week t vs hospitalization deviation at week t+1; both deviations from leave-one-year-out seasonal medians by MMWR week (modern syndromic surveillance era weeks only).",
            }
        ]
    ).to_csv(summary_path, sep="\t", index=False)

    # Regression line + bootstrap band (moving-block; descriptive).
    X = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(X, y, rcond=None)[0].tolist()
    x_grid = np.linspace(np.nanmin(x), np.nanmax(x), 80)
    y_hat = slope * x_grid + intercept
    y_band = []
    for _ in range(600):
        idx = _moving_block_bootstrap_indices(len(x), block_len=4, rng=rng)
        Xi = np.vstack([x[idx], np.ones_like(x[idx])]).T
        si, bi = np.linalg.lstsq(Xi, y[idx], rcond=None)[0].tolist()
        y_band.append(si * x_grid + bi)
    y_band = np.asarray(y_band, dtype=float)
    y_lo = np.nanquantile(y_band, 0.025, axis=0) if y_band.size else None
    y_hi = np.nanquantile(y_band, 0.975, axis=0) if y_band.size else None

    # Era coloring within the modern syndromic surveillance era only; split early vs later years for visual diagnostics.
    keep["era"] = np.where(keep["mmwr_year"] <= 2023, "2022–2023", "2024–2026")
    colors = {"2022–2023": (0.35, 0.35, 0.35), "2024–2026": (0.12, 0.47, 0.71)}

    # Panel B: year × ED-deviation quantile-bin heatmap.
    # Use qcut with duplicates='drop' to remain stable when many deviations are exactly 0.
    try:
        keep["ed_dev_bin"] = pd.qcut(keep["ed_dev"], q=[0.0, 1 / 3, 2 / 3, 1.0], labels=["Low", "Mid", "High"], duplicates="drop")
    except Exception:
        keep["ed_dev_bin"] = pd.qcut(keep["ed_dev"], q=2, labels=["Low", "High"], duplicates="drop")
    heat = (
        keep.groupby(["ed_dev_bin", "mmwr_year"], as_index=False)
        .agg(median_hosp_dev=("hosp_dev_h1", "median"), n_weeks=("hosp_dev_h1", "size"))
        .sort_values(["ed_dev_bin", "mmwr_year"])
    )
    heat.to_csv(RESULTS_DIR / "figures" / "seasonal_deviation_heatmap_rsv65p.tsv", sep="\t", index=False)

    years = sorted(keep["mmwr_year"].unique().tolist())
    bins = [str(x) for x in keep["ed_dev_bin"].cat.categories.tolist()] if hasattr(keep["ed_dev_bin"], "cat") else ["Low", "Mid", "High"]
    mat = np.full((len(bins), len(years)), np.nan, dtype=float)
    nmat = np.zeros((len(bins), len(years)), dtype=int)
    for i, b in enumerate(bins):
        for j, yr in enumerate(years):
            sub = heat[(heat["ed_dev_bin"] == b) & (heat["mmwr_year"] == yr)]
            if len(sub) == 1:
                mat[i, j] = float(sub["median_hosp_dev"].iloc[0])
                nmat[i, j] = int(sub["n_weeks"].iloc[0])

    v = mat[np.isfinite(mat)]
    vmin, vmax = (float(np.min(v)), float(np.max(v))) if v.size else (-1.0, 1.0)
    lim = max(abs(vmin), abs(vmax), 1e-6)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    pdf_path = outdir / "FigS4_incremental_value_beyond_seasonality.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(7.2, 3.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)

        # ── Panel A: scatter ──
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(-0.10, 1.04, "A", transform=ax1.transAxes,
                 fontsize=9, fontweight="bold", va="bottom")
        for era, g in keep.groupby("era"):
            ax1.scatter(
                g["ed_dev"], g["hosp_dev_h1"],
                s=18, alpha=0.7, edgecolors="white", linewidths=0.3,
                color=colors.get(str(era), (0.2, 0.2, 0.2)),
                label=str(era),
            )
        ax1.axvline(0.0, color=(0.5, 0.5, 0.5), linewidth=0.7)
        ax1.axhline(0.0, color=(0.5, 0.5, 0.5), linewidth=0.7)
        ax1.plot(x_grid, y_hat, color=(0.0, 0.0, 0.0), linewidth=1.4)
        if y_lo is not None and y_hi is not None:
            ax1.fill_between(x_grid, y_lo, y_hi, color=(0.0, 0.0, 0.0), alpha=0.10, linewidth=0.0)
        ax1.set_xlabel("Deviation from seasonal median:\nED RSV percent visits")
        ax1.set_ylabel("Deviation from seasonal median:\nnext-week RSV hospitalizations (per 100,000)")
        ax1.set_title("ED deviations and next-week hospitalization deviations", loc="left", pad=4)
        ax1.grid(axis="both", alpha=0.15)
        # Legend above the plot to avoid all four quadrant corners
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=7,
                   title=None, handletextpad=0.3, columnspacing=0.8, ncol=2)

        # Quadrant counts — use transAxes for stable positioning
        q_ru = int(((x > 0) & (y > 0)).sum())
        q_ld = int(((x < 0) & (y < 0)).sum())
        q_lu = int(((x < 0) & (y > 0)).sum())
        q_rd = int(((x > 0) & (y < 0)).sum())
        q_font = 7
        ax1.text(0.97, 0.97, f"Above both medians\n(n={q_ru})", transform=ax1.transAxes,
                 ha="right", va="top", fontsize=q_font, color=(0.35, 0.35, 0.35))
        ax1.text(0.03, 0.03, f"Below both medians\n(n={q_ld})", transform=ax1.transAxes,
                 ha="left", va="bottom", fontsize=q_font, color=(0.35, 0.35, 0.35))
        ax1.text(0.03, 0.97, f"ED↓ Hosp↑\n(n={q_lu})", transform=ax1.transAxes,
                 ha="left", va="top", fontsize=q_font, color=(0.35, 0.35, 0.35))
        ax1.text(0.97, 0.03, f"ED↑ Hosp↓\n(n={q_rd})", transform=ax1.transAxes,
                 ha="right", va="bottom", fontsize=q_font, color=(0.35, 0.35, 0.35))

        # Statistics annotation — bottom-left, multi-line
        ann_lines = [f"Spearman r = {r_s:.2f}"]
        if np.isfinite(r_lo) and np.isfinite(r_hi):
            ann_lines.append(f"95 % CI {r_lo:.2f} to {r_hi:.2f}")
        if np.isfinite(p_shift):
            ann_lines.append(f"Shift-permutation p = {p_shift:.3f}")
        ax1.text(0.03, 0.13, "\n".join(ann_lines), transform=ax1.transAxes,
                 ha="left", va="bottom", fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=(0.7, 0.7, 0.7), alpha=0.85))

        # ── Panel B: heatmap ──
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(-0.18, 1.04, "B", transform=ax2.transAxes,
                 fontsize=9, fontweight="bold", va="bottom")
        im = ax2.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax2.set_yticks(np.arange(len(bins)))
        ax2.set_yticklabels(bins, fontsize=7)
        ax2.set_xticks(np.arange(len(years)))
        ax2.set_xticklabels([str(yr) for yr in years], rotation=0, fontsize=7)
        ax2.set_title("Year × ED deviation level", loc="center", pad=4)
        ax2.set_xlabel("MMWR year")
        ax2.set_ylabel("ED deviation level")
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Median hospitalization deviation", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

        fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.18, wspace=0.30)
        pdf.savefig(fig)
        if write_png:
            fig.savefig(outdir / "FigS4_incremental_value_beyond_seasonality.png", bbox_inches="tight")
        plt.close(fig)


def figS2_expected_cost(expected_cost: pd.DataFrame, outdir: Path, cohort, *, write_png: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import string

    palette = method_palette()
    methods = [
        "ridge_univariate",
        "ridge_with_ed",
        "ridge_with_signals",
        "ridge_with_signals_plus_ed",
        "ridge_with_signals_plus_flu_pos_plus_ed",
    ]

    keep = expected_cost.copy()
    keep = keep[keep["site"].astype(str) == "Overall"]
    keep = keep[keep["age_group"].isin(list(cohort.primary_respnet_ages))]
    keep = keep[keep["horizon_weeks"] == 1]
    keep = keep[keep["threshold_rule"].astype(str) == "train_quantile_0.85"]
    # Primary panel: exclude_last_weeks=4 only (conservative / audit-friendly)
    keep = keep[keep["exclude_last_weeks"] == 4].copy()

    outcomes = ["FluSurv-NET", "RSV-NET"]
    ages = list(cohort.primary_respnet_ages)
    panel_letters = string.ascii_uppercase

    pdf_path = outdir / "FigS2_expected_cost.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, len(ages), figsize=(7.2, 3.4 + 1.3 * len(ages)), sharex=True, sharey=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(2, 1)
        for i, net in enumerate(outcomes):
            for j, age in enumerate(ages):
                ax = axes[i, j]
                idx = i * len(ages) + j
                letter = panel_letters[idx] if idx < len(panel_letters) else f"P{idx+1}"
                ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="bottom")
                g = keep[(keep["surveillance_network"] == net) & (keep["age_group"] == age)].copy()
                if g.empty:
                    ax.set_axis_off()
                    continue
                ax.axhline(1.0, color=palette["seasonal_naive"], linewidth=1.0, linestyle="--", label="Baseline (=1)")
                for m in methods:
                    gg = g[g["method"] == m].sort_values("cost_ratio_fn_to_fp")
                    if gg.empty:
                        continue
                    y = pd.to_numeric(gg.get("expected_cost_ratio_vs_baseline"), errors="coerce").to_numpy(dtype=float)
                    x = pd.to_numeric(gg.get("cost_ratio_fn_to_fp"), errors="coerce").to_numpy(dtype=float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 2:
                        continue
                    ax.plot(
                        x[ok], y[ok],
                        marker=method_marker(m), markersize=3, linewidth=1.4,
                        color=palette.get(m, (0.0, 0.0, 0.0)),
                        linestyle=method_linestyle(m),
                        label=nice_method_label(m),
                    )
                ax.set_xscale("log")
                ax.set_title(f"{net} · {nice_age_label(age)}", loc="left", pad=4)
                ax.grid(axis="y", alpha=0.15)
                if i == 1:
                    ax.set_xlabel("Miss / false-alarm cost ratio (FN/FP)")
                if j == 0:
                    ax.set_ylabel("Expected cost / baseline")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.01), fontsize=7)
        fig.suptitle(
            "Expected cost ratios versus seasonal baseline",
            y=0.98, fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.14, 1, 0.95))
        pdf.savefig(fig)

        if write_png:
            png_path = outdir / "FigS2_expected_cost.png"
            fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)


def figS3_alert_lead_time(alert_lead: pd.DataFrame, outdir: Path, cohort, *, write_png: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import string

    palette = method_palette()
    methods = [
        "seasonal_naive",
        "ridge_univariate",
        "ridge_with_ed",
        "ridge_with_signals",
        "ridge_with_signals_plus_ed",
        "ridge_with_signals_plus_flu_pos_plus_ed",
    ]

    keep = alert_lead.copy()
    keep = keep[keep["site"].astype(str) == "Overall"]
    keep = keep[keep["age_group"].isin(list(cohort.primary_respnet_ages))]
    keep = keep[keep["threshold_rule"].astype(str) == "train_quantile_0.85"]
    # Primary panel: exclude_last_weeks=4 only (conservative / audit-friendly)
    keep = keep[keep["exclude_last_weeks"] == 4].copy()

    outcomes = ["FluSurv-NET", "RSV-NET"]
    ages = list(cohort.primary_respnet_ages)
    panel_letters = string.ascii_uppercase

    pdf_path = outdir / "FigS3_alert_lead_time.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, len(ages), figsize=(7.2, 3.4 + 1.3 * len(ages)), sharex=True, sharey=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(2, 1)
        for i, net in enumerate(outcomes):
            for j, age in enumerate(ages):
                ax = axes[i, j]
                idx = i * len(ages) + j
                letter = panel_letters[idx] if idx < len(panel_letters) else f"P{idx+1}"
                ax.text(-0.12, 1.04, letter, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="bottom")
                g = keep[(keep["surveillance_network"] == net) & (keep["age_group"] == age)].copy()
                if g.empty:
                    ax.set_axis_off()
                    continue
                for m in methods:
                    gg = g[g["method"] == m]
                    if gg.empty:
                        continue
                    x = pd.to_numeric(gg.get("median_lead_weeks"), errors="coerce").to_numpy(dtype=float)
                    y = pd.to_numeric(gg.get("detection_rate"), errors="coerce").to_numpy(dtype=float)
                    x_max = pd.to_numeric(gg.get("max_lead_weeks"), errors="coerce").to_numpy(dtype=float)
                    if not (np.isfinite(x).any() and np.isfinite(y).any()):
                        continue
                    xx = float(x[np.isfinite(x)][0]) if np.isfinite(x).any() else float("nan")
                    yy = float(y[np.isfinite(y)][0]) if np.isfinite(y).any() else float("nan")
                    xm = float(x_max[np.isfinite(x_max)][0]) if np.isfinite(x_max).any() else float("nan")
                    if not (np.isfinite(xx) and np.isfinite(yy)):
                        continue
                    ax.scatter(
                        [xx], [yy], s=50,
                        color=palette.get(m, (0.0, 0.0, 0.0)),
                        label=nice_method_label(m), zorder=3,
                        edgecolors="white", linewidths=0.5,
                    )
                    if np.isfinite(xm) and xm >= xx:
                        ax.plot([xx, xm], [yy, yy], color=palette.get(m, (0.0, 0.0, 0.0)),
                                linewidth=1.4, alpha=0.8)
                ax.set_xlim(-0.1, 4.2)
                ax.set_ylim(-0.02, 1.02)
                ax.set_title(f"{net} · {nice_age_label(age)}", loc="left", pad=4)
                ax.grid(axis="y", alpha=0.15)
                if i == 1:
                    ax.set_xlabel("Median lead time (weeks)")
                if j == 0:
                    ax.set_ylabel("Onset detection rate (proportion)")
                # Annotate n_events for transparency
                total_events = g["n_events"].sum() if "n_events" in g.columns else None
                if total_events is not None and np.isfinite(total_events) and int(total_events) > 0:
                    ax.text(0.97, 0.03, f"n = {int(total_events)} events", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=6, color=(0.4, 0.4, 0.4))

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 0.01), fontsize=7)
        fig.suptitle(
            "Onset detection rate and median lead time",
            y=0.98, fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.14, 1, 0.95))
        pdf.savefig(fig)

        if write_png:
            png_path = outdir / "FigS3_alert_lead_time.png"
            fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate publication-ready plots from benchmark tables.")
    ap.add_argument(
        "--cohort-profile",
        default=cohort_from_env("older_adult_65plus"),
        help="Cohort profile id (default: env COHORT_PROFILE or 'older_adult_65plus').",
    )
    ap.add_argument("--outdir", default="plots/publication", help="Output directory for figures (default: plots/publication)")
    ap.add_argument("--time-series-overview", default=str(RESULTS_DIR / "figures/time_series_overview.tsv"))
    ap.add_argument("--signal-coverage", default=str(RESULTS_DIR / "analysis/signal_coverage_report.tsv"))
    ap.add_argument("--paired-benchmark", default=str(RESULTS_DIR / "benchmarks/paired_benchmark.tsv"))
    ap.add_argument("--expected-cost", default=str(RESULTS_DIR / "benchmarks/expected_cost.tsv"))
    ap.add_argument("--alert-lead-time", default=str(RESULTS_DIR / "benchmarks/alert_lead_time.tsv"))
    ap.add_argument("--time-series-full", default=str(RESULTS_DIR / "analysis/analysis_table.tsv"))
    ap.add_argument("--lag-analysis", default=str(RESULTS_DIR / "figures/lag_analysis.tsv"))
    ap.add_argument("--write-png", action="store_true", help="Also write 300dpi PNG exports (default: PDF only).")
    args = ap.parse_args()
    cohort = get_cohort(args.cohort_profile)

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    setup_style()

    time_series = pd.read_csv(args.time_series_overview, sep="\t") if Path(args.time_series_overview).exists() else pd.DataFrame()
    time_series_full = pd.read_csv(args.time_series_full, sep="\t") if Path(args.time_series_full).exists() else pd.DataFrame()
    coverage = pd.read_csv(args.signal_coverage, sep="\t") if Path(args.signal_coverage).exists() else pd.DataFrame()
    paired = pd.read_csv(args.paired_benchmark, sep="\t") if Path(args.paired_benchmark).exists() else pd.DataFrame()
    expected_cost = pd.read_csv(args.expected_cost, sep="\t") if Path(args.expected_cost).exists() else pd.DataFrame()
    alert_lead = pd.read_csv(args.alert_lead_time, sep="\t") if Path(args.alert_lead_time).exists() else pd.DataFrame()
    lag_analysis = pd.read_csv(args.lag_analysis, sep="\t") if Path(args.lag_analysis).exists() else pd.DataFrame()

    if time_series.empty:
        raise SystemExit(f"Missing or empty time-series overview table: {args.time_series_overview}")
    if time_series_full.empty:
        raise SystemExit(f"Missing or empty analysis table: {args.time_series_full}")
    if coverage.empty:
        raise SystemExit(f"Missing or empty signal coverage table: {args.signal_coverage}")
    if paired.empty:
        raise SystemExit(f"Missing or empty paired benchmark table: {args.paired_benchmark}")
    if expected_cost.empty:
        raise SystemExit(f"Missing or empty expected cost table: {args.expected_cost}")
    if alert_lead.empty:
        raise SystemExit(f"Missing or empty alert lead time table: {args.alert_lead_time}")

    fig1_time_series_overview(time_series, outdir=outdir, cohort=cohort, write_png=bool(args.write_png))
    fig2_lag_and_coverage_board(lag_analysis, coverage, outdir=outdir, cohort=cohort, write_png=bool(args.write_png))
    fig3_paired_benchmark(paired, outdir=outdir, cohort=cohort, write_png=bool(args.write_png))
    figS5_paired_benchmark_realtime(paired, outdir=outdir, cohort=cohort, write_png=bool(args.write_png))
    figS2_expected_cost(expected_cost, outdir=outdir, cohort=cohort, write_png=bool(args.write_png))
    figS3_alert_lead_time(alert_lead, outdir=outdir, cohort=cohort, write_png=bool(args.write_png))
    figS4_incremental_value_beyond_seasonality(time_series_full, outdir=outdir, cohort=cohort, write_png=bool(args.write_png))
    print(f"Wrote figures to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
