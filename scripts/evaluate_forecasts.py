#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import zlib
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from cohort_profiles import cohort_from_env, get_cohort
from utils import RESULTS_DIR, ensure_dir


def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray, lam: float) -> float:
    p = X_train.shape[1]
    XtX = X_train.T @ X_train
    A = XtX + lam * np.eye(p)
    b = X_train.T @ y_train
    beta = np.linalg.solve(A, b)
    return float(x_pred @ beta)


def seasonal_naive(series: np.ndarray, i: int, horizon: int, season_len: int) -> float | None:
    j = i + horizon - season_len
    if j < 0:
        return None
    return float(series[j])


def seasonal_naive_same_week_last_year(y_by_epiweek: dict[int, float], target_epiweek: int) -> float | None:
    """
    Seasonal naïve baseline for seasonal/irregular surveillance series:
      y_hat(target_week) = y(target_week in the prior year, same MMWR week number).

    We use epiweek codes YYYYWW and look up (YYYY-1)WW. If week 53 is not present
    in the prior year, we fall back to (YYYY-1)W52.
    """
    year = int(target_epiweek) // 100
    week = int(target_epiweek) % 100
    prior = (year - 1) * 100 + week
    v = y_by_epiweek.get(prior)
    if v is not None and np.isfinite(v):
        return float(v)
    if week == 53:
        prior2 = (year - 1) * 100 + 52
        v2 = y_by_epiweek.get(prior2)
        if v2 is not None and np.isfinite(v2):
            return float(v2)
    return None


def metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def moving_block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if block_len <= 1 or n < 2:
        return rng.integers(0, n, size=n, dtype=int)
    block_len = min(block_len, n)
    starts = rng.integers(0, n - block_len + 1, size=math.ceil(n / block_len), dtype=int)
    idx = np.concatenate([np.arange(s, s + block_len, dtype=int) for s in starts])[:n]
    return idx


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    n: int,
    reps: int,
    block_len: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if reps <= 0 or n < 20:
        return (float("nan"), float("nan"))
    stats = np.empty(reps, dtype=float)
    for b in range(reps):
        idx = moving_block_bootstrap_indices(n, block_len=block_len, rng=rng)
        stats[b] = float(stat_fn(idx))
    stats = stats[np.isfinite(stats)]
    if len(stats) < max(30, reps // 3):
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def stable_seed(base_seed: int, parts: Iterable[object]) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return (int(base_seed) + zlib.adler32(s)) % (2**32 - 1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Basic seasonal encoding using MMWR week number (1-53).
    out = df.copy()
    week_dt = pd.to_datetime(out.get("week_ending_date"), errors="coerce")
    out["week_ending_date"] = week_dt.dt.date
    # Define contiguous weekly blocks; any gap >7 days starts a new block.
    d = week_dt.diff().dt.days
    out["time_block"] = (d.fillna(7).astype(float) > 7.0).cumsum().astype(int)
    week = pd.to_numeric(out["epiweeks"] % 100, errors="coerce")
    # Normalize to [0, 2pi)
    angle = 2 * math.pi * (week - 1) / 52.0
    out["sin52"] = np.sin(angle)
    out["cos52"] = np.cos(angle)

    out["wili"] = pd.to_numeric(out.get("wili"), errors="coerce")
    out["rsv_positivity"] = pd.to_numeric(out.get("rsv_positivity"), errors="coerce")
    out["flu_positivity"] = pd.to_numeric(out.get("flu_positivity"), errors="coerce")
    out["rate_per_100k"] = pd.to_numeric(out.get("rate_per_100k"), errors="coerce")
    out["ed_influenza_pct_65p"] = pd.to_numeric(out.get("ed_influenza_pct_65p"), errors="coerce")
    out["ed_rsv_pct_65p"] = pd.to_numeric(out.get("ed_rsv_pct_65p"), errors="coerce")

    # Forward-fill short gaps to reduce row loss, but DO NOT bridge long missing spans across
    # surveillance gaps/years. We enforce this by forward-filling within contiguous time blocks.
    ffill_limit = 8
    for col in [
        "wili",
        "rsv_positivity",
        "flu_positivity",
        "ed_influenza_pct_65p",
        "ed_rsv_pct_65p",
    ]:
        out[col] = out.groupby("time_block")[col].ffill(limit=ffill_limit)

    out["y_lag1"] = out.groupby("time_block")["rate_per_100k"].shift(1)
    out["y_lag2"] = out.groupby("time_block")["rate_per_100k"].shift(2)
    out["rsvpos_lag1"] = out.groupby("time_block")["rsv_positivity"].shift(1)
    out["rsvpos_lag2"] = out.groupby("time_block")["rsv_positivity"].shift(2)
    out["wili_lag1"] = out.groupby("time_block")["wili"].shift(1)
    out["wili_lag2"] = out.groupby("time_block")["wili"].shift(2)
    out["flupos_lag1"] = out.groupby("time_block")["flu_positivity"].shift(1)
    out["flupos_lag2"] = out.groupby("time_block")["flu_positivity"].shift(2)

    out["edflu_65p_lag1"] = out.groupby("time_block")["ed_influenza_pct_65p"].shift(1)
    out["edflu_65p_lag2"] = out.groupby("time_block")["ed_influenza_pct_65p"].shift(2)

    out["edrsv_65p_lag1"] = out.groupby("time_block")["ed_rsv_pct_65p"].shift(1)
    out["edrsv_65p_lag2"] = out.groupby("time_block")["ed_rsv_pct_65p"].shift(2)
    return out


def _fit_one_ridge_X(
    X_all: np.ndarray,
    y: np.ndarray,
    time_block: np.ndarray,
    train_end_exclusive: int,
    origin_i: int,
    horizon: int,
    lam: float,
) -> float | None:
    """
    Direct multi-horizon ridge:
      Use rows t in [0, train_end_exclusive) with targets y[t+h].
      Requires t+h < origin_i to avoid label leakage.
    """
    train_idx = np.arange(0, train_end_exclusive)
    target_idx = train_idx + horizon
    ok = target_idx < origin_i
    train_idx = train_idx[ok]
    target_idx = target_idx[ok]
    # Do not train on pairs that cross surveillance gaps (non-contiguous weekly blocks).
    ok2 = time_block[train_idx] == time_block[target_idx]
    train_idx = train_idx[ok2]
    target_idx = target_idx[ok2]

    if len(train_idx) < 30:
        return None

    x_pred = X_all[origin_i]
    if not np.isfinite(x_pred).all():
        return None

    X_train = X_all[train_idx]
    y_train = y[target_idx]
    mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train2 = X_train[mask]
    y_train2 = y_train[mask]
    if len(y_train2) < 30:
        return None

    return ridge_fit_predict(X_train2, y_train2, x_pred, lam=lam)


def tune_ridge_lambda(
    X_all: np.ndarray,
    y: np.ndarray,
    time_block: np.ndarray,
    horizon: int,
    tune_end_exclusive: int,
    tune_min_train: int,
    lam_grid: list[float],
    tune_last_n: int,
    base_lam: float,
) -> tuple[float, int]:
    """
    Nested (within-training) tuning of ridge lambda for time-series forecasting.

    We choose lambda by evaluating prequential MAE on an inner window that ends strictly
    before the outer evaluation window (cutoff=tune_end_exclusive).
    """
    if tune_end_exclusive <= 0 or horizon <= 0:
        return (base_lam, 0)
    # NOTE: This is a *nested* tuning pass that must end strictly before the outer evaluation
    # window. We intentionally allow a smaller "tune_min_train" than the outer min_train so
    # that tuning is feasible even when the outer evaluation starts immediately after min_train.
    start = max(int(tune_min_train), int(tune_end_exclusive) - int(tune_last_n))
    stop = int(tune_end_exclusive)
    origins = list(range(start, stop))
    if len(origins) < 40:
        return (base_lam, 0)

    best_lam = base_lam
    best_mae = float("inf")
    best_n = 0

    for lam in lam_grid:
        preds: list[float] = []
        truth: list[float] = []
        for j in origins:
            train_end_exclusive = j - horizon
            if train_end_exclusive <= 0:
                continue
            p = _fit_one_ridge_X(
                X_all,
                y,
                time_block=time_block,
                train_end_exclusive=train_end_exclusive,
                origin_i=j,
                horizon=horizon,
                lam=float(lam),
            )
            if p is None:
                continue
            if (j + horizon) >= len(y) or time_block[j] != time_block[j + horizon]:
                continue
            yt = y[j + horizon]
            if not np.isfinite(yt):
                continue
            preds.append(float(p))
            truth.append(float(yt))
        if len(truth) < 30:
            continue
        yt_arr = np.asarray(truth, dtype=float)
        yp_arr = np.asarray(preds, dtype=float)
        mae = float(np.mean(np.abs(yt_arr - yp_arr)))
        if mae < best_mae - 1e-12 or (abs(mae - best_mae) <= 1e-12 and float(lam) < float(best_lam)):
            best_mae = mae
            best_lam = float(lam)
            best_n = int(len(truth))

    if not np.isfinite(best_mae):
        return (base_lam, 0)
    return (best_lam, best_n)


def _fit_one_ridge_transfer_X(
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    time_block_train: np.ndarray,
    train_end_exclusive: int,
    origin_train_i: int,
    x_pred: np.ndarray,
    horizon: int,
    lam: float,
) -> float | None:
    """
    Like _fit_one_ridge_X but trains on X_train_all and predicts using x_pred (from another series).
    """
    train_idx = np.arange(0, train_end_exclusive)
    target_idx = train_idx + horizon
    ok = target_idx < origin_train_i
    train_idx = train_idx[ok]
    target_idx = target_idx[ok]
    ok2 = time_block_train[train_idx] == time_block_train[target_idx]
    train_idx = train_idx[ok2]
    target_idx = target_idx[ok2]

    if len(train_idx) < 30:
        return None

    if not np.isfinite(x_pred).all():
        return None

    X_train = X_train_all[train_idx]
    y_train = y_train_all[target_idx]
    mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    X_train2 = X_train[mask]
    y_train2 = y_train[mask]
    if len(y_train2) < 30:
        return None

    return ridge_fit_predict(X_train2, y_train2, x_pred, lam=lam)


def evaluate_within_series(
    df: pd.DataFrame,
    horizons: list[int],
    season_len: int,
    min_train: int,
    eval_last_n: int,
    lam: float,
    surveillance_network: str,
    include_wili: bool,
    include_rsvpos: bool,
    include_flupos: bool,
    include_ed: bool,
    tune_lam: bool,
    lam_grid: list[float],
    tune_min_train: int,
    tune_last_n: int,
    tune_warmup_ed: int,
    exclude_last_weeks: int,
    run_ridge: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("epiweeks").reset_index(drop=True)
    df = build_features(df)

    y = df["rate_per_100k"].to_numpy(dtype=float)
    time_block = df["time_block"].to_numpy(dtype=int) if "time_block" in df.columns else np.zeros(len(df), dtype=int)
    y_by_epiweek = {int(e): float(v) for e, v in zip(df["epiweeks"].astype(int).to_list(), y)}
    n_full = len(df)
    n = max(0, n_full - max(0, int(exclude_last_weeks)))
    if n < 10:
        return pd.DataFrame(), pd.DataFrame()
    eval_start = max(min_train, n - eval_last_n)

    rows_forecast: list[dict[str, object]] = []
    hp_rows: list[dict[str, object]] = []

    base_feats = ["sin52", "cos52", "y_lag1", "y_lag2"]
    sig_feats: list[str] = []
    if include_rsvpos:
        sig_feats += ["rsvpos_lag1", "rsvpos_lag2"]
    if include_wili:
        sig_feats += ["wili_lag1", "wili_lag2"]
    if include_flupos:
        sig_feats_flupos = sig_feats + ["flupos_lag1", "flupos_lag2"]
    else:
        sig_feats_flupos = None

    # ED feature mapping depends on outcome age_group + which pathogen the outcome represents.
    age = str(df["age_group"].iloc[0]) if "age_group" in df.columns and len(df) else ""
    ed_feats: list[str] = []
    if include_ed:
        if surveillance_network == "FluSurv-NET":
            if age == "65+ yr":
                ed_feats = ["edflu_65p_lag1", "edflu_65p_lag2"]
        elif surveillance_network == "RSV-NET":
            if age == "65+ yr":
                ed_feats = ["edrsv_65p_lag1", "edrsv_65p_lag2"]

    sig_feats_plus_ed = (sig_feats + ed_feats) if ed_feats else None
    sig_feats_flupos_plus_ed = (sig_feats_flupos + ed_feats) if (sig_feats_flupos and ed_feats) else None

    X_base = df[base_feats].to_numpy(dtype=float)
    X_sig = df[base_feats + sig_feats].to_numpy(dtype=float) if sig_feats else None
    X_sig_flupos = df[base_feats + sig_feats_flupos].to_numpy(dtype=float) if sig_feats_flupos else None
    X_ed = df[base_feats + ed_feats].to_numpy(dtype=float) if ed_feats else None
    X_sig_plus_ed = df[base_feats + sig_feats_plus_ed].to_numpy(dtype=float) if sig_feats_plus_ed else None
    X_sig_flupos_plus_ed = df[base_feats + sig_feats_flupos_plus_ed].to_numpy(dtype=float) if sig_feats_flupos_plus_ed else None

    def _first_full_row_index(X: np.ndarray | None) -> int | None:
        if X is None or X.size == 0:
            return None
        ok = np.isfinite(X).all(axis=1)
        idx = np.flatnonzero(ok)
        return int(idx[0]) if len(idx) else None

    # Nested lambda tuning (within training window), per horizon and per design matrix.
    #
    # For ED-inclusive feature sets, signals are often missing in the early portion of the time
    # series. If we tune lambda using the same cutoff as the global evaluation start, tuning is
    # frequently infeasible (all-missing ED) and/or risks leakage if tuned-on weeks are evaluated.
    # To avoid leakage, we reserve a short "warmup" period after ED signals begin for tuning and
    # only start evaluating ED-inclusive methods after that cutoff.
    tune_cutoff_base = int(eval_start)
    tune_cutoff_sig = int(eval_start)
    tune_cutoff_sig_flupos = int(eval_start)
    first_ed = _first_full_row_index(X_ed)
    first_sig_ed = _first_full_row_index(X_sig_plus_ed)
    first_sig_flupos_ed = _first_full_row_index(X_sig_flupos_plus_ed)

    def _cutoff_after_sparse_signals(first_full: int | None) -> int:
        if first_full is None:
            return int(eval_start)
        # _fit_one_ridge_X requires >=30 complete training rows; and tune_ridge_lambda requires
        # >=40 inner origins to proceed. For ED-inclusive matrices, we therefore need a cutoff
        # far enough after ED begins to satisfy BOTH constraints.
        min_complete_train = 30
        min_inner_origins = 40
        base = int(first_full) + int(tune_warmup_ed)
        min_needed = int(first_full) + int(min_complete_train + min_inner_origins)
        return int(max(eval_start, base, min_needed))

    tune_cutoff_ed = _cutoff_after_sparse_signals(first_ed)
    tune_cutoff_sig_ed = _cutoff_after_sparse_signals(first_sig_ed)
    tune_cutoff_sig_flupos_ed = _cutoff_after_sparse_signals(first_sig_flupos_ed)

    # Safety clamp: cutoff indices beyond n leave no evaluation window; tuning will fall back.
    max_h = max(horizons) if horizons else 1
    max_cutoff = max(0, int(n - max_h))
    tune_cutoff_ed = int(min(tune_cutoff_ed, max_cutoff))
    tune_cutoff_sig_ed = int(min(tune_cutoff_sig_ed, max_cutoff))
    tune_cutoff_sig_flupos_ed = int(min(tune_cutoff_sig_flupos_ed, max_cutoff))
    lam_uni: dict[int, float] = {h: float(lam) for h in horizons}
    lam_ed: dict[int, float] = {h: float(lam) for h in horizons}
    lam_sig: dict[int, float] = {h: float(lam) for h in horizons}
    lam_sig_ed: dict[int, float] = {h: float(lam) for h in horizons}
    lam_sig_flupos: dict[int, float] = {h: float(lam) for h in horizons}
    lam_sig_flupos_ed: dict[int, float] = {h: float(lam) for h in horizons}

    if tune_lam and run_ridge:
        for h in horizons:
            lu, nu = tune_ridge_lambda(
                X_base,
                y,
                time_block=time_block,
                horizon=h,
                tune_end_exclusive=tune_cutoff_base,
                tune_min_train=tune_min_train,
                lam_grid=lam_grid,
                tune_last_n=tune_last_n,
                base_lam=float(lam),
            )
            lam_uni[h] = lu
            hp_rows.append(
                {
                    "train_scope": "within_site",
                    "method": "ridge_univariate",
                    "horizon_weeks": int(h),
                    "lambda": float(lu),
                    "tune_n": int(nu),
                    "lam_grid": ",".join(str(x) for x in lam_grid),
                    "tune_min_train": int(tune_min_train),
                    "tune_last_n": int(tune_last_n),
                    "tune_cutoff_index": int(tune_cutoff_base),
                }
            )

            if X_ed is not None:
                le, ne = tune_ridge_lambda(
                    X_ed,
                    y,
                    time_block=time_block,
                    horizon=h,
                    tune_end_exclusive=tune_cutoff_ed,
                    tune_min_train=tune_min_train,
                    lam_grid=lam_grid,
                    tune_last_n=tune_last_n,
                    base_lam=float(lam),
                )
                lam_ed[h] = le
                hp_rows.append(
                    {
                        "train_scope": "within_site",
                        "method": "ridge_with_ed",
                        "horizon_weeks": int(h),
                        "lambda": float(le),
                        "tune_n": int(ne),
                        "lam_grid": ",".join(str(x) for x in lam_grid),
                        "tune_min_train": int(tune_min_train),
                        "tune_last_n": int(tune_last_n),
                        "tune_cutoff_index": int(tune_cutoff_ed),
                    }
                )

            if X_sig is not None:
                ls, ns = tune_ridge_lambda(
                    X_sig,
                    y,
                    time_block=time_block,
                    horizon=h,
                    tune_end_exclusive=tune_cutoff_sig,
                    tune_min_train=tune_min_train,
                    lam_grid=lam_grid,
                    tune_last_n=tune_last_n,
                    base_lam=float(lam),
                )
                lam_sig[h] = ls
                hp_rows.append(
                    {
                        "train_scope": "within_site",
                        "method": "ridge_with_signals",
                        "horizon_weeks": int(h),
                        "lambda": float(ls),
                        "tune_n": int(ns),
                        "lam_grid": ",".join(str(x) for x in lam_grid),
                        "tune_min_train": int(tune_min_train),
                        "tune_last_n": int(tune_last_n),
                        "tune_cutoff_index": int(tune_cutoff_sig),
                    }
                )

            if X_sig_plus_ed is not None:
                lse, nse = tune_ridge_lambda(
                    X_sig_plus_ed,
                    y,
                    time_block=time_block,
                    horizon=h,
                    tune_end_exclusive=tune_cutoff_sig_ed,
                    tune_min_train=tune_min_train,
                    lam_grid=lam_grid,
                    tune_last_n=tune_last_n,
                    base_lam=float(lam),
                )
                lam_sig_ed[h] = lse
                hp_rows.append(
                    {
                        "train_scope": "within_site",
                        "method": "ridge_with_signals_plus_ed",
                        "horizon_weeks": int(h),
                        "lambda": float(lse),
                        "tune_n": int(nse),
                        "lam_grid": ",".join(str(x) for x in lam_grid),
                        "tune_min_train": int(tune_min_train),
                        "tune_last_n": int(tune_last_n),
                        "tune_cutoff_index": int(tune_cutoff_sig_ed),
                    }
                )

            if X_sig_flupos is not None:
                lsp, nsp = tune_ridge_lambda(
                    X_sig_flupos,
                    y,
                    time_block=time_block,
                    horizon=h,
                    tune_end_exclusive=tune_cutoff_sig_flupos,
                    tune_min_train=tune_min_train,
                    lam_grid=lam_grid,
                    tune_last_n=tune_last_n,
                    base_lam=float(lam),
                )
                lam_sig_flupos[h] = lsp
                hp_rows.append(
                    {
                        "train_scope": "within_site",
                        "method": "ridge_with_signals_plus_flu_pos",
                        "horizon_weeks": int(h),
                        "lambda": float(lsp),
                        "tune_n": int(nsp),
                        "lam_grid": ",".join(str(x) for x in lam_grid),
                        "tune_min_train": int(tune_min_train),
                        "tune_last_n": int(tune_last_n),
                        "tune_cutoff_index": int(tune_cutoff_sig_flupos),
                    }
                )

            if X_sig_flupos_plus_ed is not None:
                lspe, nspe = tune_ridge_lambda(
                    X_sig_flupos_plus_ed,
                    y,
                    time_block=time_block,
                    horizon=h,
                    tune_end_exclusive=tune_cutoff_sig_flupos_ed,
                    tune_min_train=tune_min_train,
                    lam_grid=lam_grid,
                    tune_last_n=tune_last_n,
                    base_lam=float(lam),
                )
                lam_sig_flupos_ed[h] = lspe
                hp_rows.append(
                    {
                        "train_scope": "within_site",
                        "method": "ridge_with_signals_plus_flu_pos_plus_ed",
                        "horizon_weeks": int(h),
                        "lambda": float(lspe),
                        "tune_n": int(nspe),
                        "lam_grid": ",".join(str(x) for x in lam_grid),
                        "tune_min_train": int(tune_min_train),
                        "tune_last_n": int(tune_last_n),
                        "tune_cutoff_index": int(tune_cutoff_sig_flupos_ed),
                    }
                )

    eval_start_ed = int(tune_cutoff_ed) if (tune_lam and run_ridge) else int(eval_start)
    eval_start_sig_ed = int(tune_cutoff_sig_ed) if (tune_lam and run_ridge) else int(eval_start)
    eval_start_sig_flupos_ed = int(tune_cutoff_sig_flupos_ed) if (tune_lam and run_ridge) else int(eval_start)

    for h in horizons:
        for i in range(eval_start, n - h):
            if time_block[i] != time_block[i + h]:
                continue
            y_true = float(y[i + h])
            origin_epiweek = int(df.loc[i, "epiweeks"])
            target_epiweek = int(df.loc[i + h, "epiweeks"])
            origin_date = str(df.loc[i, "week_ending_date"])
            target_date = str(df.loc[i + h, "week_ending_date"])

            pred_sn = seasonal_naive_same_week_last_year(y_by_epiweek, target_epiweek)
            if pred_sn is not None:
                rows_forecast.append(
                    {
                        "train_scope": "within_site",
                        "method": "seasonal_naive",
                        "horizon_weeks": h,
                        "origin_idx": i,
                        "origin_epiweek": origin_epiweek,
                        "origin_week_ending_date": origin_date,
                        "target_epiweek": target_epiweek,
                        "target_week_ending_date": target_date,
                        "y_true": y_true,
                        "y_pred": pred_sn,
                    }
                )

            if not run_ridge:
                continue

            # Ridge regression with expanding window (direct multi-horizon).
            train_end_exclusive = i - h
            if train_end_exclusive <= 0:
                continue

            pred_uni = _fit_one_ridge_X(
                X_base,
                y,
                time_block=time_block,
                train_end_exclusive=train_end_exclusive,
                origin_i=i,
                horizon=h,
                lam=float(lam_uni[h]),
            )
            if pred_uni is not None:
                rows_forecast.append(
                    {
                        "train_scope": "within_site",
                        "method": "ridge_univariate",
                        "horizon_weeks": h,
                        "origin_idx": i,
                        "origin_epiweek": origin_epiweek,
                        "origin_week_ending_date": origin_date,
                        "target_epiweek": target_epiweek,
                        "target_week_ending_date": target_date,
                        "y_true": y_true,
                        "y_pred": pred_uni,
                    }
                )

            if X_ed is not None:
                if i >= eval_start_ed:
                    pred_ed = _fit_one_ridge_X(
                        X_ed,
                        y,
                        time_block=time_block,
                        train_end_exclusive=train_end_exclusive,
                        origin_i=i,
                        horizon=h,
                        lam=float(lam_ed[h]),
                    )
                    if pred_ed is not None:
                        rows_forecast.append(
                            {
                                "train_scope": "within_site",
                                "method": "ridge_with_ed",
                                "horizon_weeks": h,
                                "origin_idx": i,
                                "origin_epiweek": origin_epiweek,
                                "origin_week_ending_date": origin_date,
                                "target_epiweek": target_epiweek,
                                "target_week_ending_date": target_date,
                                "y_true": y_true,
                                "y_pred": pred_ed,
                            }
                        )

            if X_sig is not None:
                pred_sig = _fit_one_ridge_X(
                    X_sig,
                    y,
                    time_block=time_block,
                    train_end_exclusive=train_end_exclusive,
                    origin_i=i,
                    horizon=h,
                    lam=float(lam_sig[h]),
                )
                if pred_sig is not None:
                    rows_forecast.append(
                        {
                            "train_scope": "within_site",
                            "method": "ridge_with_signals",
                            "horizon_weeks": h,
                            "origin_idx": i,
                            "origin_epiweek": origin_epiweek,
                            "origin_week_ending_date": origin_date,
                            "target_epiweek": target_epiweek,
                            "target_week_ending_date": target_date,
                            "y_true": y_true,
                            "y_pred": pred_sig,
                        }
                    )

            if X_sig_flupos is not None:
                pred_sig2 = _fit_one_ridge_X(
                    X_sig_flupos,
                    y,
                    time_block=time_block,
                    train_end_exclusive=train_end_exclusive,
                    origin_i=i,
                    horizon=h,
                    lam=float(lam_sig_flupos[h]),
                )
                if pred_sig2 is not None:
                    rows_forecast.append(
                        {
                            "train_scope": "within_site",
                            "method": "ridge_with_signals_plus_flu_pos",
                            "horizon_weeks": h,
                            "origin_idx": i,
                            "origin_epiweek": origin_epiweek,
                            "origin_week_ending_date": origin_date,
                            "target_epiweek": target_epiweek,
                            "target_week_ending_date": target_date,
                            "y_true": y_true,
                            "y_pred": pred_sig2,
                        }
                    )

            if X_sig_plus_ed is not None:
                if i >= eval_start_sig_ed:
                    pred_sig_ed = _fit_one_ridge_X(
                        X_sig_plus_ed,
                        y,
                        time_block=time_block,
                        train_end_exclusive=train_end_exclusive,
                        origin_i=i,
                        horizon=h,
                        lam=float(lam_sig_ed[h]),
                    )
                    if pred_sig_ed is not None:
                        rows_forecast.append(
                            {
                                "train_scope": "within_site",
                                "method": "ridge_with_signals_plus_ed",
                                "horizon_weeks": h,
                                "origin_idx": i,
                                "origin_epiweek": origin_epiweek,
                                "origin_week_ending_date": origin_date,
                                "target_epiweek": target_epiweek,
                                "target_week_ending_date": target_date,
                                "y_true": y_true,
                                "y_pred": pred_sig_ed,
                            }
                        )

            if X_sig_flupos_plus_ed is not None:
                if i >= eval_start_sig_flupos_ed:
                    pred_sig_ed2 = _fit_one_ridge_X(
                        X_sig_flupos_plus_ed,
                        y,
                        time_block=time_block,
                        train_end_exclusive=train_end_exclusive,
                        origin_i=i,
                        horizon=h,
                        lam=float(lam_sig_flupos_ed[h]),
                    )
                    if pred_sig_ed2 is not None:
                        rows_forecast.append(
                            {
                                "train_scope": "within_site",
                                "method": "ridge_with_signals_plus_flu_pos_plus_ed",
                                "horizon_weeks": h,
                                "origin_idx": i,
                                "origin_epiweek": origin_epiweek,
                                "origin_week_ending_date": origin_date,
                                "target_epiweek": target_epiweek,
                                "target_week_ending_date": target_date,
                                "y_true": y_true,
                                "y_pred": pred_sig_ed2,
                            }
                        )

    pred_df = pd.DataFrame(rows_forecast)
    hp_df = pd.DataFrame(hp_rows)
    return pred_df, hp_df


def evaluate_transfer_overall_to_site(
    overall_df: pd.DataFrame,
    site_df: pd.DataFrame,
    horizons: list[int],
    season_len: int,
    min_train: int,
    eval_last_n: int,
    lam: float,
    include_wili: bool,
    include_rsvpos: bool,
    include_flupos: bool,
    exclude_last_weeks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_df = overall_df.sort_values("epiweeks").drop_duplicates("epiweeks").reset_index(drop=True)
    site_df = site_df.sort_values("epiweeks").drop_duplicates("epiweeks").reset_index(drop=True)
    if exclude_last_weeks > 0:
        overall_df = overall_df.iloc[: max(0, len(overall_df) - exclude_last_weeks)].reset_index(drop=True)
        site_df = site_df.iloc[: max(0, len(site_df) - exclude_last_weeks)].reset_index(drop=True)

    overall_df = build_features(overall_df)
    site_df = build_features(site_df)

    y_overall = overall_df["rate_per_100k"].to_numpy(dtype=float)
    y_site = site_df["rate_per_100k"].to_numpy(dtype=float)
    time_block_overall = (
        overall_df["time_block"].to_numpy(dtype=int) if "time_block" in overall_df.columns else np.zeros(len(overall_df), dtype=int)
    )
    time_block_site = site_df["time_block"].to_numpy(dtype=int) if "time_block" in site_df.columns else np.zeros(len(site_df), dtype=int)

    epi_overall = overall_df["epiweeks"].to_numpy(dtype=int)
    epi_site = site_df["epiweeks"].to_numpy(dtype=int)

    n_site = len(site_df)
    if n_site < 10:
        return pd.DataFrame(), pd.DataFrame()
    eval_start = max(min_train, n_site - eval_last_n)

    base_feats = ["sin52", "cos52", "y_lag1", "y_lag2"]
    sig_feats: list[str] = []
    if include_rsvpos:
        sig_feats += ["rsvpos_lag1", "rsvpos_lag2"]
    if include_wili:
        sig_feats += ["wili_lag1", "wili_lag2"]
    if include_flupos:
        sig_feats_flupos = sig_feats + ["flupos_lag1", "flupos_lag2"]
    else:
        sig_feats_flupos = None

    X_overall_base = overall_df[base_feats].to_numpy(dtype=float)
    X_site_base = site_df[base_feats].to_numpy(dtype=float)
    X_overall_sig = overall_df[base_feats + sig_feats].to_numpy(dtype=float) if sig_feats else None
    X_site_sig = site_df[base_feats + sig_feats].to_numpy(dtype=float) if sig_feats else None
    X_overall_sig_flupos = overall_df[base_feats + sig_feats_flupos].to_numpy(dtype=float) if sig_feats_flupos else None
    X_site_sig_flupos = site_df[base_feats + sig_feats_flupos].to_numpy(dtype=float) if sig_feats_flupos else None

    rows_forecast: list[dict[str, object]] = []

    for h in horizons:
        for i_site in range(eval_start, n_site - h):
            if time_block_site[i_site] != time_block_site[i_site + h]:
                continue
            origin_epiweek = int(epi_site[i_site])
            target_epiweek = int(epi_site[i_site + h])
            origin_date = str(site_df.loc[i_site, "week_ending_date"])
            target_date = str(site_df.loc[i_site + h, "week_ending_date"])
            y_true = float(y_site[i_site + h])

            # Find how much Overall history exists before the origin epiweek.
            i_overall = int(np.searchsorted(epi_overall, origin_epiweek, side="left"))
            train_end_exclusive = i_overall - h
            if train_end_exclusive <= 0:
                continue

            pred_uni = _fit_one_ridge_transfer_X(
                X_train_all=X_overall_base,
                y_train_all=y_overall,
                time_block_train=time_block_overall,
                train_end_exclusive=train_end_exclusive,
                origin_train_i=i_overall,
                x_pred=X_site_base[i_site],
                horizon=h,
                lam=lam,
            )
            if pred_uni is not None:
                rows_forecast.append(
                    {
                        "train_scope": "train_on_overall",
                        "method": "ridge_univariate",
                        "horizon_weeks": h,
                        "origin_idx": i_site,
                        "origin_epiweek": origin_epiweek,
                        "origin_week_ending_date": origin_date,
                        "target_epiweek": target_epiweek,
                        "target_week_ending_date": target_date,
                        "y_true": y_true,
                        "y_pred": pred_uni,
                    }
                )

            if X_overall_sig is not None and X_site_sig is not None:
                pred_sig = _fit_one_ridge_transfer_X(
                    X_train_all=X_overall_sig,
                    y_train_all=y_overall,
                    time_block_train=time_block_overall,
                    train_end_exclusive=train_end_exclusive,
                    origin_train_i=i_overall,
                    x_pred=X_site_sig[i_site],
                    horizon=h,
                    lam=lam,
                )
                if pred_sig is not None:
                    rows_forecast.append(
                        {
                            "train_scope": "train_on_overall",
                            "method": "ridge_with_signals",
                            "horizon_weeks": h,
                            "origin_idx": i_site,
                            "origin_epiweek": origin_epiweek,
                            "origin_week_ending_date": origin_date,
                            "target_epiweek": target_epiweek,
                            "target_week_ending_date": target_date,
                            "y_true": y_true,
                            "y_pred": pred_sig,
                        }
                    )

            if X_overall_sig_flupos is not None and X_site_sig_flupos is not None:
                pred_sig2 = _fit_one_ridge_transfer_X(
                    X_train_all=X_overall_sig_flupos,
                    y_train_all=y_overall,
                    time_block_train=time_block_overall,
                    train_end_exclusive=train_end_exclusive,
                    origin_train_i=i_overall,
                    x_pred=X_site_sig_flupos[i_site],
                    horizon=h,
                    lam=lam,
                )
                if pred_sig2 is not None:
                    rows_forecast.append(
                        {
                            "train_scope": "train_on_overall",
                            "method": "ridge_with_signals_plus_flu_pos",
                            "horizon_weeks": h,
                            "origin_idx": i_site,
                            "origin_epiweek": origin_epiweek,
                            "origin_week_ending_date": origin_date,
                            "target_epiweek": target_epiweek,
                            "target_week_ending_date": target_date,
                            "y_true": y_true,
                            "y_pred": pred_sig2,
                        }
                    )

    pred_df = pd.DataFrame(rows_forecast)
    return pred_df, pd.DataFrame()


def _site_generalization_with_ci(
    pred_long: pd.DataFrame,
    baseline_method: str,
    reps: int,
    block_len: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pair_keys = [
        "exclude_last_weeks",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "horizon_weeks",
        "origin_epiweek",
        "target_epiweek",
    ]
    base = (
        pred_long[pred_long["method"] == baseline_method][pair_keys + ["y_true", "y_pred"]]
        .rename(columns={"y_pred": "y_pred_baseline"})
        .copy()
    )
    rows: list[dict[str, object]] = []

    for (excl, net, age, site, scope, h, method), g in pred_long.groupby(
        ["exclude_last_weeks", "surveillance_network", "age_group", "site", "train_scope", "horizon_weeks", "method"]
    ):
        if method == baseline_method:
            continue
        gg = g[pair_keys + ["y_true", "y_pred"]].merge(base, on=pair_keys + ["y_true"], how="inner")
        if gg.empty:
            continue
        gg = gg.sort_values("origin_epiweek")
        yt = gg["y_true"].to_numpy(dtype=float)
        yp = gg["y_pred"].to_numpy(dtype=float)
        yb = gg["y_pred_baseline"].to_numpy(dtype=float)

        abs_base = np.abs(yt - yb)
        abs_method = np.abs(yt - yp)
        se_base = (yt - yb) ** 2
        se_method = (yt - yp) ** 2

        mae_base = float(np.mean(abs_base))
        mae_method = float(np.mean(abs_method))
        rmse_base = float(math.sqrt(np.mean(se_base)))
        rmse_method = float(math.sqrt(np.mean(se_method)))

        n = len(gg)

        def mae_imp_stat(idx: np.ndarray) -> float:
            return float(np.mean(abs_base[idx]) - np.mean(abs_method[idx]))

        def rmse_imp_stat(idx: np.ndarray) -> float:
            return float(math.sqrt(np.mean(se_base[idx])) - math.sqrt(np.mean(se_method[idx])))

        mae_imp_lo, mae_imp_hi = bootstrap_ci(mae_imp_stat, n=n, reps=reps, block_len=block_len, rng=rng)

        rmse_imp_lo, rmse_imp_hi = bootstrap_ci(rmse_imp_stat, n=n, reps=reps, block_len=block_len, rng=rng)

        rows.append(
            {
                "exclude_last_weeks": int(excl),
                "surveillance_network": net,
                "age_group": age,
                "site": site,
                "train_scope": scope,
                "method": method,
                "horizon_weeks": int(h),
                "paired_n": int(len(gg)),
                "metric": "MAE",
                "baseline_value": float(mae_base),
                "method_value": float(mae_method),
                "improvement_over_baseline": float(mae_base - mae_method),
                "improvement_ci_lo": mae_imp_lo,
                "improvement_ci_hi": mae_imp_hi,
            }
        )
        rows.append(
            {
                "exclude_last_weeks": int(excl),
                "surveillance_network": net,
                "age_group": age,
                "site": site,
                "train_scope": scope,
                "method": method,
                "horizon_weeks": int(h),
                "paired_n": int(len(gg)),
                "metric": "RMSE",
                "baseline_value": float(rmse_base),
                "method_value": float(rmse_method),
                "improvement_over_baseline": float(rmse_base - rmse_method),
                "improvement_ci_lo": rmse_imp_lo,
                "improvement_ci_hi": rmse_imp_hi,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["exclude_last_weeks", "surveillance_network", "age_group", "train_scope", "site", "method", "horizon_weeks", "metric"]
    )


def _alert_utility(
    pred_long: pd.DataFrame,
    series_df: pd.DataFrame,
    method: str,
    horizons: list[int],
    threshold_quantiles: list[float],
    min_train: int,
    eval_last_n: int,
    reps: int,
    block_len: int,
    rng: np.random.Generator,
    exclude_last_weeks: int,
) -> pd.DataFrame:
    series_df = series_df.sort_values("epiweeks").drop_duplicates("epiweeks").reset_index(drop=True)
    y = pd.to_numeric(series_df["rate_per_100k"], errors="coerce").to_numpy(dtype=float)
    n_full = len(series_df)
    n = max(0, n_full - max(0, int(exclude_last_weeks)))
    if n < 10:
        return pd.DataFrame()
    eval_start = max(min_train, n - eval_last_n)
    train_y = y[:eval_start]
    train_y = train_y[np.isfinite(train_y)]
    if len(train_y) < 30:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for h in horizons:
        g = pred_long[(pred_long["method"] == method) & (pred_long["horizon_weeks"] == h)].copy()
        if g.empty:
            continue
        g = g.sort_values("origin_epiweek")
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        ok = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[ok]
        yp = yp[ok]
        if len(yt) < 30:
            continue
        n_eval = len(yt)

        for q in threshold_quantiles:
            thr = float(np.quantile(train_y, q))
            alert = yp >= thr
            event = yt >= thr
            tp = int((alert & event).sum())
            fp = int((alert & ~event).sum())
            tn = int((~alert & ~event).sum())
            fn = int((~alert & event).sum())
            n_all = tp + fp + tn + fn
            if n_all == 0:
                continue
            sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
            f1 = 2 * ppv * sens / (ppv + sens) if np.isfinite(ppv) and np.isfinite(sens) and (ppv + sens) > 0 else float("nan")

            def stat_from_idx(idx: np.ndarray) -> tuple[float, float]:
                a = yp[idx] >= thr
                e = yt[idx] >= thr
                tp2 = int((a & e).sum())
                fp2 = int((a & ~e).sum())
                tn2 = int((~a & ~e).sum())
                fn2 = int((~a & e).sum())
                n2 = tp2 + fp2 + tn2 + fn2
                if n2 == 0:
                    return (float("nan"), float("nan"))
                sens2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else float("nan")
                spec2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else float("nan")
                return (sens2, spec2)

            def ci_of(component: int) -> tuple[float, float]:
                def one(idx: np.ndarray) -> float:
                    return float(stat_from_idx(idx)[component])

                return bootstrap_ci(one, n=n_eval, reps=reps, block_len=block_len, rng=rng)

            sens_lo, sens_hi = ci_of(0)
            spec_lo, spec_hi = ci_of(1)

            rows.append(
                {
                    "method": method,
                    "horizon_weeks": int(h),
                    "threshold_rule": f"train_quantile_{q:.2f}",
                    "threshold_value": thr,
                    "n": int(n_all),
                    "event_frequency": float((tp + fn) / n_all),
                    "alert_frequency": float((tp + fp) / n_all),
                    "sensitivity": float(sens),
                    "specificity": float(spec),
                    "ppv": float(ppv),
                    "npv": float(npv),
                    "f1": float(f1),
                    "sensitivity_ci_lo": sens_lo,
                    "sensitivity_ci_hi": sens_hi,
                    "specificity_ci_lo": spec_lo,
                    "specificity_ci_hi": spec_hi,
                }
            )
    return pd.DataFrame(rows)


def _alert_utility_paired(
    pred_long: pd.DataFrame,
    series_df: pd.DataFrame,
    method: str,
    baseline_method: str,
    horizons: list[int],
    threshold_quantiles: list[float],
    min_train: int,
    eval_last_n: int,
    reps: int,
    block_len: int,
    rng: np.random.Generator,
    exclude_last_weeks: int,
) -> pd.DataFrame:
    """
    Alert utility computed on a *paired* evaluation set: baseline and method are aligned on the same
    (origin_epiweek, target_epiweek) pairs to avoid unfair comparisons when methods have different
    coverage due to missing signals.
    """
    series_df = series_df.sort_values("epiweeks").drop_duplicates("epiweeks").reset_index(drop=True)
    y = pd.to_numeric(series_df["rate_per_100k"], errors="coerce").to_numpy(dtype=float)
    n_full = len(series_df)
    n = max(0, n_full - max(0, int(exclude_last_weeks)))
    if n < 10:
        return pd.DataFrame()
    eval_start = max(min_train, n - eval_last_n)
    train_y = y[:eval_start]
    train_y = train_y[np.isfinite(train_y)]
    if len(train_y) < 30:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    key_cols = ["origin_epiweek", "target_epiweek", "y_true"]

    for h in horizons:
        gm = pred_long[(pred_long["method"] == method) & (pred_long["horizon_weeks"] == h)].copy()
        gb = pred_long[(pred_long["method"] == baseline_method) & (pred_long["horizon_weeks"] == h)].copy()
        if gm.empty or gb.empty:
            continue
        gm = gm[key_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_method"}).copy()
        gb = gb[key_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_baseline"}).copy()
        gg = gm.merge(gb, on=key_cols, how="inner")
        if gg.empty:
            continue

        yt = pd.to_numeric(gg["y_true"], errors="coerce").to_numpy(dtype=float)
        yp_m = pd.to_numeric(gg["y_pred_method"], errors="coerce").to_numpy(dtype=float)
        yp_b = pd.to_numeric(gg["y_pred_baseline"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(yt) & np.isfinite(yp_m) & np.isfinite(yp_b)
        yt = yt[ok]
        yp_m = yp_m[ok]
        yp_b = yp_b[ok]
        if len(yt) < 30:
            continue
        n_eval = int(len(yt))

        for q in threshold_quantiles:
            thr = float(np.quantile(train_y, q))
            event = yt >= thr

            alert_m = yp_m >= thr
            tp = int((alert_m & event).sum())
            fp = int((alert_m & ~event).sum())
            tn = int((~alert_m & ~event).sum())
            fn = int((~alert_m & event).sum())
            n_all = tp + fp + tn + fn
            if n_all == 0:
                continue
            sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
            f1 = 2 * ppv * sens / (ppv + sens) if np.isfinite(ppv) and np.isfinite(sens) and (ppv + sens) > 0 else float("nan")

            alert_b = yp_b >= thr
            tp0 = int((alert_b & event).sum())
            fp0 = int((alert_b & ~event).sum())
            tn0 = int((~alert_b & ~event).sum())
            fn0 = int((~alert_b & event).sum())
            sens0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else float("nan")
            spec0 = tn0 / (tn0 + fp0) if (tn0 + fp0) > 0 else float("nan")
            alert_freq0 = float((tp0 + fp0) / n_all)

            def stat_from_idx(idx: np.ndarray, which: str, component: int) -> float:
                if which == "method":
                    a = yp_m[idx] >= thr
                else:
                    a = yp_b[idx] >= thr
                e = yt[idx] >= thr
                tp2 = int((a & e).sum())
                fp2 = int((a & ~e).sum())
                tn2 = int((~a & ~e).sum())
                fn2 = int((~a & e).sum())
                n2 = tp2 + fp2 + tn2 + fn2
                if n2 == 0:
                    return float("nan")
                sens2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else float("nan")
                spec2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else float("nan")
                return float(sens2 if component == 0 else spec2)

            def ci_of(which: str, component: int) -> tuple[float, float]:
                def one(idx: np.ndarray) -> float:
                    return float(stat_from_idx(idx, which=which, component=component))

                return bootstrap_ci(one, n=n_eval, reps=reps, block_len=block_len, rng=rng)

            sens_lo, sens_hi = ci_of("method", 0)
            spec_lo, spec_hi = ci_of("method", 1)

            rows.append(
                {
                    "method": method,
                    "baseline_method": baseline_method,
                    "horizon_weeks": int(h),
                    "threshold_rule": f"train_quantile_{q:.2f}",
                    "threshold_value": thr,
                    "paired_n": int(n_all),
                    "event_frequency": float(event.mean()),
                    "alert_frequency": float((tp + fp) / n_all),
                    "sensitivity": float(sens),
                    "specificity": float(spec),
                    "ppv": float(ppv),
                    "npv": float(npv),
                    "f1": float(f1),
                    "sensitivity_ci_lo": sens_lo,
                    "sensitivity_ci_hi": sens_hi,
                    "specificity_ci_lo": spec_lo,
                    "specificity_ci_hi": spec_hi,
                    "baseline_sensitivity": float(sens0),
                    "baseline_specificity": float(spec0),
                    "baseline_alert_frequency": float(alert_freq0),
                }
            )

    return pd.DataFrame(rows)


def add_prequential_intervals(
    pred_long: pd.DataFrame,
    levels: list[float] = [0.5, 0.9],
    min_history: int = 30,
) -> pd.DataFrame:
    """
    Add empirical prequential prediction intervals based on past absolute errors:
      q_t(level) = quantile(|y_true - y_pred| for origins < t, prob=level)
      interval(level) = [y_pred - q_t, y_pred + q_t]
    This is leakage-safe because it uses only past forecast errors for the same series/method/horizon.
    """
    out = pred_long.copy()
    for lvl in levels:
        lo = f"pi{int(lvl*100)}_lo"
        hi = f"pi{int(lvl*100)}_hi"
        out[lo] = np.nan
        out[hi] = np.nan

    grp_cols = [
        "exclude_last_weeks",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "method",
        "horizon_weeks",
    ]
    for key, g in out.groupby(grp_cols):
        g = g.sort_values("origin_epiweek")
        abs_err = np.abs(pd.to_numeric(g["y_true"], errors="coerce") - pd.to_numeric(g["y_pred"], errors="coerce")).to_numpy(dtype=float)
        yp = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=float)
        idx = g.index.to_numpy(dtype=int)
        history: list[float] = []
        for i in range(len(g)):
            if len(history) >= min_history and np.isfinite(yp[i]):
                hist = np.asarray(history, dtype=float)
                hist = hist[np.isfinite(hist)]
                if len(hist) >= min_history:
                    for lvl in levels:
                        q = float(np.quantile(hist, lvl))
                        out.loc[idx[i], f"pi{int(lvl*100)}_lo"] = yp[i] - q
                        out.loc[idx[i], f"pi{int(lvl*100)}_hi"] = yp[i] + q
            if np.isfinite(abs_err[i]):
                history.append(float(abs_err[i]))
    return out


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> np.ndarray:
    width = hi - lo
    below = (y < lo).astype(float)
    above = (y > hi).astype(float)
    penalty = (2.0 / alpha) * ((lo - y) * below + (y - hi) * above)
    return width + penalty


def expected_cost_from_rates(event_freq: float, sensitivity: float, specificity: float, cost_ratio_fn_to_fp: float) -> float | None:
    if not (np.isfinite(event_freq) and np.isfinite(sensitivity) and np.isfinite(specificity) and np.isfinite(cost_ratio_fn_to_fp)):
        return None
    if event_freq < 0 or event_freq > 1:
        return None
    fp_rate = (1.0 - event_freq) * (1.0 - specificity)
    fn_rate = event_freq * (1.0 - sensitivity)
    return float(cost_ratio_fn_to_fp * fn_rate + 1.0 * fp_rate)


def onset_weeks(event: np.ndarray) -> np.ndarray:
    if len(event) == 0:
        return np.array([], dtype=bool)
    prev = np.concatenate([[False], event[:-1]])
    return event & (~prev)

def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lo, hi). Uses z=1.96 by default (approx 95%).
    """
    n = int(n)
    k = int(k)
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    adj = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    lo = max(0.0, center - adj)
    hi = min(1.0, center + adj)
    return (float(lo), float(hi))


def lead_time_summary_for_threshold(
    pred_long: pd.DataFrame,
    threshold_value: float,
    methods: list[str],
) -> pd.DataFrame:
    """
    For each method, compute event-onset detection and lead time based on forecasts that target the onset week.
    Lead time is the maximum horizon (in weeks) that correctly alerts an onset week.
    """
    # Build a target-week outcome series on the prediction grid (unique target_epiweek).
    base = pred_long.drop_duplicates(subset=["target_epiweek"])[["target_epiweek", "y_true"]].copy()
    base = base.sort_values("target_epiweek")
    y = pd.to_numeric(base["y_true"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 30:
        return pd.DataFrame()
    event = (y >= threshold_value) & ok
    onset = onset_weeks(event)
    onset_weeks_list = base.loc[onset, "target_epiweek"].to_numpy(dtype=int)
    n_events = int(len(onset_weeks_list))
    if n_events == 0:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for method in methods:
        g = pred_long[pred_long["method"] == method].copy()
        if g.empty:
            continue
        # For each onset week, find max horizon among correct alerts.
        leads: list[int] = []
        detected = 0
        for w in onset_weeks_list:
            gg = g[g["target_epiweek"] == w].copy()
            if gg.empty:
                continue
            gg["y_pred"] = pd.to_numeric(gg["y_pred"], errors="coerce")
            gg = gg[np.isfinite(gg["y_pred"])]
            if gg.empty:
                continue
            alerts = gg[gg["y_pred"] >= threshold_value]
            if alerts.empty:
                continue
            detected += 1
            leads.append(int(alerts["horizon_weeks"].max()))
        if detected == 0:
            lo_ci, hi_ci = wilson_ci(0, n_events)
            rows.append(
                {
                    "method": method,
                    "n_events": n_events,
                    "n_detected": 0,
                    "detection_rate": 0.0,
                    "detection_rate_ci_lo": float(lo_ci),
                    "detection_rate_ci_hi": float(hi_ci),
                    "median_lead_weeks": float("nan"),
                    "p25_lead_weeks": float("nan"),
                    "p75_lead_weeks": float("nan"),
                    "max_lead_weeks": float("nan"),
                }
            )
            continue
        arr = np.asarray(leads, dtype=float)
        lo_ci, hi_ci = wilson_ci(int(detected), n_events)
        rows.append(
            {
                "method": method,
                "n_events": n_events,
                "n_detected": int(detected),
                "detection_rate": float(detected / n_events),
                "detection_rate_ci_lo": float(lo_ci),
                "detection_rate_ci_hi": float(hi_ci),
                "median_lead_weeks": float(np.median(arr)),
                "p25_lead_weeks": float(np.quantile(arr, 0.25)),
                "p75_lead_weeks": float(np.quantile(arr, 0.75)),
                "max_lead_weeks": float(np.max(arr)),
            }
        )

    return pd.DataFrame(rows).sort_values(["method"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Rolling-origin evaluation for RESP-NET hospitalization forecasting (public data).")
    ap.add_argument(
        "--cohort-profile",
        default=cohort_from_env("older_adult_65plus"),
        help="Cohort profile id (default: env COHORT_PROFILE or 'older_adult_65plus').",
    )
    ap.add_argument("--input", default=str(RESULTS_DIR / "analysis/analysis_table.tsv"), help="Input analysis table (default: results/analysis/analysis_table.tsv)")
    ap.add_argument("--outdir", default=str(RESULTS_DIR / "benchmarks"), help="Output directory for benchmark TSVs (default: results/benchmarks)")
    ap.add_argument("--horizons", default="1,2,3,4", help="Forecast horizons in weeks (default: 1,2,3,4)")
    ap.add_argument("--season-len", type=int, default=52, help="Season length in weeks (default: 52)")
    ap.add_argument("--min-train", type=int, default=156, help="Minimum training points before evaluation (default: 156 ~ 3 years)")
    ap.add_argument("--eval-last-n", type=int, default=260, help="Evaluate on last N weeks (default: 260 ~ 5 years)")
    ap.add_argument("--lam", type=float, default=1.0, help="Ridge penalty lambda fallback (used when tuning is off; default: 1.0)")
    ap.add_argument("--tune-lam", type=int, default=1, choices=[0, 1], help="Whether to tune ridge lambda via nested time-series CV within training window (default: 1)")
    ap.add_argument("--lam-grid", default="0.01,0.1,1,10,100", help="Comma-separated lambda candidates for tuning (default: 0.01,0.1,1,10,100)")
    ap.add_argument(
        "--tune-min-train",
        type=int,
        default=104,
        help="Minimum training points for *inner* lambda tuning (default: 104 ~ 2 years; may be smaller than --min-train)",
    )
    ap.add_argument("--tune-last-n", type=int, default=104, help="Inner tuning window length in weeks (default: 104 ~ 2 years)")
    ap.add_argument(
        "--tune-warmup-ed",
        type=int,
        default=52,
        help="Warmup weeks after ED signals begin to reserve for tuning; ED-inclusive methods start evaluation after this (default: 52)",
    )
    ap.add_argument("--exclude-last-weeks", default="0,4", help="Comma-separated sensitivity cutoffs; drops last N weeks before evaluation (default: 0,4)")
    ap.add_argument("--bootstrap-reps", type=int, default=600, help="Moving-block bootstrap reps for CIs (default: 600)")
    ap.add_argument("--bootstrap-block-len", type=int, default=4, help="Moving-block length in weeks (default: 4)")
    ap.add_argument("--bootstrap-seed", type=int, default=20260326, help="Base RNG seed for bootstrap (default: 20260326)")
    args = ap.parse_args()
    cohort = get_cohort(args.cohort_profile)

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    exclude_last_weeks_list = [int(x.strip()) for x in args.exclude_last_weeks.split(",") if x.strip()]
    lam_grid = [float(x.strip()) for x in args.lam_grid.split(",") if x.strip()]
    tune_lam = bool(int(args.tune_lam))

    df = pd.read_csv(args.input, sep="\t")
    df = df.dropna(subset=["rate_per_100k"])
    if "site" not in df.columns:
        df["site"] = "Overall"

    bench_dir = Path(args.outdir)
    ensure_dir(bench_dir)

    all_pred_long: list[pd.DataFrame] = []
    all_hyperparams: list[pd.DataFrame] = []

    for excl in exclude_last_weeks_list:
        overall = df[df["site"].astype(str) == "Overall"].copy()

        for (net, age, site), g in df.groupby(["surveillance_network", "age_group", "site"]):
            g = g.sort_values("epiweeks").drop_duplicates("epiweeks")
            include_wili = net == "FluSurv-NET"
            include_rsvpos = net == "RSV-NET"
            include_flupos = net == "FluSurv-NET"

            pred_df, hp_df = evaluate_within_series(
                g,
                horizons=horizons,
                season_len=args.season_len,
                min_train=args.min_train,
                eval_last_n=args.eval_last_n,
                lam=args.lam,
                surveillance_network=str(net),
                include_wili=include_wili,
                include_rsvpos=include_rsvpos,
                include_flupos=include_flupos,
                include_ed=True,
                tune_lam=tune_lam,
                lam_grid=lam_grid,
                tune_min_train=int(args.tune_min_train),
                tune_last_n=int(args.tune_last_n),
                tune_warmup_ed=int(args.tune_warmup_ed),
                exclude_last_weeks=excl,
                run_ridge=(str(site) == "Overall"),
            )
            if str(site) == "Overall" and not hp_df.empty:
                hp_df = hp_df.copy()
                hp_df.insert(0, "exclude_last_weeks", int(excl))
                hp_df.insert(1, "surveillance_network", net)
                hp_df.insert(2, "age_group", age)
                hp_df.insert(3, "site", site)
                all_hyperparams.append(hp_df)
            if not pred_df.empty:
                pred_df.insert(0, "exclude_last_weeks", int(excl))
                pred_df.insert(1, "surveillance_network", net)
                pred_df.insert(2, "age_group", age)
                pred_df.insert(3, "site", site)
                all_pred_long.append(pred_df)

            if site != "Overall":
                g_overall = overall[(overall["surveillance_network"] == net) & (overall["age_group"] == age)].copy()
                if not g_overall.empty:
                    pred_t, _ = evaluate_transfer_overall_to_site(
                        overall_df=g_overall,
                        site_df=g,
                        horizons=horizons,
                        season_len=args.season_len,
                        min_train=args.min_train,
                        eval_last_n=args.eval_last_n,
                        lam=args.lam,
                        include_wili=include_wili,
                        include_rsvpos=include_rsvpos,
                        include_flupos=include_flupos,
                        exclude_last_weeks=excl,
                    )
                    if not pred_t.empty:
                        pred_t.insert(0, "exclude_last_weeks", int(excl))
                        pred_t.insert(1, "surveillance_network", net)
                        pred_t.insert(2, "age_group", age)
                        pred_t.insert(3, "site", site)
                        all_pred_long.append(pred_t)

    if not all_pred_long:
        raise SystemExit("No per-origin predictions produced.")

    pred_long = pd.concat(all_pred_long, ignore_index=True)
    pred_long = add_prequential_intervals(pred_long, levels=[0.5, 0.9], min_history=30)
    pred_long.to_csv(bench_dir / "predictions_long.tsv", sep="\t", index=False)

    hyperparams = pd.concat(all_hyperparams, ignore_index=True) if all_hyperparams else pd.DataFrame()
    hyperparams.to_csv(bench_dir / "hyperparams.tsv", sep="\t", index=False)

    # Forecast metrics + CIs from per-origin predictions.
    rows_eval: list[dict[str, object]] = []
    group_cols = [
        "exclude_last_weeks",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "method",
        "horizon_weeks",
    ]
    for key, g in pred_long.groupby(group_cols):
        excl, net, age, site, scope, method, h = key
        g = g.sort_values("origin_epiweek")
        yt = g["y_true"].to_numpy(dtype=float)
        yp = g["y_pred"].to_numpy(dtype=float)
        ok = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[ok]
        yp = yp[ok]
        if len(yt) < 30:
            continue
        abs_err = np.abs(yt - yp)
        se = (yt - yp) ** 2
        n = len(yt)

        do_ci = str(site) == "Overall"
        if do_ci:
            rng_mae = np.random.default_rng(stable_seed(args.bootstrap_seed, [*key, "MAE"]))
            mae_lo, mae_hi = bootstrap_ci(
                lambda idx: float(np.mean(abs_err[idx])),
                n=n,
                reps=args.bootstrap_reps,
                block_len=args.bootstrap_block_len,
                rng=rng_mae,
            )
        else:
            mae_lo, mae_hi = (float("nan"), float("nan"))
        mae = float(np.mean(abs_err))
        rows_eval.append(
            {
                "exclude_last_weeks": int(excl),
                "surveillance_network": net,
                "age_group": age,
                "site": site,
                "train_scope": scope,
                "method": method,
                "horizon_weeks": int(h),
                "split_scheme": "rolling_origin",
                "metric": "MAE",
                "value": mae,
                "ci_lo": mae_lo,
                "ci_hi": mae_hi,
                "n_forecasts": int(n),
            }
        )

        if do_ci:
            rng_rmse = np.random.default_rng(stable_seed(args.bootstrap_seed, [*key, "RMSE"]))
            rmse_lo, rmse_hi = bootstrap_ci(
                lambda idx: float(math.sqrt(np.mean(se[idx]))),
                n=n,
                reps=args.bootstrap_reps,
                block_len=args.bootstrap_block_len,
                rng=rng_rmse,
            )
        else:
            rmse_lo, rmse_hi = (float("nan"), float("nan"))
        rmse = float(math.sqrt(np.mean(se)))
        rows_eval.append(
            {
                "exclude_last_weeks": int(excl),
                "surveillance_network": net,
                "age_group": age,
                "site": site,
                "train_scope": scope,
                "method": method,
                "horizon_weeks": int(h),
                "split_scheme": "rolling_origin",
                "metric": "RMSE",
                "value": rmse,
                "ci_lo": rmse_lo,
                "ci_hi": rmse_hi,
                "n_forecasts": int(n),
            }
        )

    forecast_eval = pd.DataFrame(rows_eval).sort_values(group_cols + ["metric"])
    forecast_eval.to_csv(bench_dir / "forecast_eval.tsv", sep="\t", index=False)

    method_benchmark = forecast_eval.rename(columns={"value": "metric_value"})
    method_benchmark.to_csv(bench_dir / "method_benchmark.tsv", sep="\t", index=False)

    # Paired benchmark vs seasonal naive on matched (origin,target) pairs to avoid bias from missing-signal coverage.
    rng_pair = np.random.default_rng(stable_seed(args.bootstrap_seed, ["paired_benchmark"]))
    paired_input = pred_long[(pred_long["site"].astype(str) == "Overall") & (pred_long["train_scope"] == "within_site")].copy()
    paired_benchmark = _site_generalization_with_ci(
        paired_input,
        baseline_method="seasonal_naive",
        reps=int(args.bootstrap_reps),
        block_len=args.bootstrap_block_len,
        rng=rng_pair,
    )
    paired_benchmark.to_csv(bench_dir / "paired_benchmark.tsv", sep="\t", index=False)

    # Alert utility (decision-relevant): focus site=Overall + cohort primary age group(s).
    util_rows: list[pd.DataFrame] = []
    overall_series = df[df["site"].astype(str) == "Overall"].copy()
    overall_series = overall_series[overall_series["age_group"].isin(list(cohort.primary_respnet_ages))].copy()
    utility_methods = [
        "seasonal_naive",
        "ridge_univariate",
        "ridge_with_ed",
        "ridge_with_signals",
        "ridge_with_signals_plus_ed",
        "ridge_with_signals_plus_flu_pos",
        "ridge_with_signals_plus_flu_pos_plus_ed",
    ]
    for excl in exclude_last_weeks_list:
        for (net, age), series in overall_series.groupby(["surveillance_network", "age_group"]):
            series = series.sort_values("epiweeks").drop_duplicates("epiweeks")
            preds = pred_long[
                (pred_long["exclude_last_weeks"] == excl)
                & (pred_long["site"].astype(str) == "Overall")
                & (pred_long["surveillance_network"] == net)
                & (pred_long["age_group"] == age)
                & (pred_long["train_scope"] == "within_site")
            ].copy()
            if preds.empty:
                continue
            for method in utility_methods:
                if method not in preds["method"].unique():
                    continue
                rng_util = np.random.default_rng(stable_seed(args.bootstrap_seed, [excl, net, age, method, "alert"]))
                util = _alert_utility_paired(
                    preds,
                    series_df=series,
                    method=method,
                    baseline_method="seasonal_naive",
                    horizons=horizons,
                    threshold_quantiles=[0.85, 0.90, 0.95],
                    min_train=args.min_train,
                    eval_last_n=args.eval_last_n,
                    reps=int(args.bootstrap_reps),
                    block_len=args.bootstrap_block_len,
                    rng=rng_util,
                    exclude_last_weeks=int(excl),
                )
                if util.empty:
                    continue
                util.insert(0, "exclude_last_weeks", int(excl))
                util.insert(1, "surveillance_network", net)
                util.insert(2, "age_group", age)
                util.insert(3, "site", "Overall")
                util_rows.append(util)

    alert_utility = pd.concat(util_rows, ignore_index=True) if util_rows else pd.DataFrame()
    alert_utility.to_csv(bench_dir / "alert_utility.tsv", sep="\t", index=False)

    # Expected cost grid (decision closure): derived from alert utility rates.
    cost_ratios = [0.5, 1.0, 2.0, 5.0, 10.0]
    cost_rows: list[dict[str, object]] = []
    if not alert_utility.empty:
        for _, r in alert_utility.iterrows():
            for cr in cost_ratios:
                event_freq = float(r.get("event_frequency"))
                c = expected_cost_from_rates(
                    event_freq=event_freq,
                    sensitivity=float(r.get("sensitivity")),
                    specificity=float(r.get("specificity")),
                    cost_ratio_fn_to_fp=float(cr),
                )
                c0 = expected_cost_from_rates(
                    event_freq=event_freq,
                    sensitivity=float(r.get("baseline_sensitivity")),
                    specificity=float(r.get("baseline_specificity")),
                    cost_ratio_fn_to_fp=float(cr),
                )
                if c is None or c0 is None:
                    continue
                cost_rows.append(
                    {
                        "exclude_last_weeks": int(r["exclude_last_weeks"]),
                        "surveillance_network": r["surveillance_network"],
                        "age_group": r["age_group"],
                        "site": r["site"],
                        "method": r["method"],
                        "horizon_weeks": int(r["horizon_weeks"]),
                        "threshold_rule": r["threshold_rule"],
                        "threshold_value": float(r["threshold_value"]),
                        "cost_ratio_fn_to_fp": float(cr),
                        "expected_cost": float(c),
                        "baseline_expected_cost": float(c0),
                        "expected_cost_diff_vs_baseline": float(c - c0),
                        "expected_cost_ratio_vs_baseline": float(c / c0) if c0 > 0 else float("nan"),
                    }
                )
    expected_cost = pd.DataFrame(cost_rows)
    expected_cost.to_csv(bench_dir / "expected_cost.tsv", sep="\t", index=False)

    # Alert lead time to event onset (decision closure): for the national primary series.
    lead_rows: list[pd.DataFrame] = []
    if not alert_utility.empty:
        key_cols = ["exclude_last_weeks", "surveillance_network", "age_group", "site", "threshold_rule", "threshold_value"]
        lead_keys = alert_utility[key_cols].drop_duplicates()
        for _, kk in lead_keys.iterrows():
            excl = int(kk["exclude_last_weeks"])
            net = kk["surveillance_network"]
            age = kk["age_group"]
            site = kk["site"]
            thr_rule = kk["threshold_rule"]
            thr = float(kk["threshold_value"])
            sub = pred_long[
                (pred_long["exclude_last_weeks"] == excl)
                & (pred_long["surveillance_network"] == net)
                & (pred_long["age_group"] == age)
                & (pred_long["site"] == site)
                & (pred_long["train_scope"] == "within_site")
            ].copy()
            if sub.empty:
                continue
            methods_here = [m for m in utility_methods if m in sub["method"].unique()]
            lt = lead_time_summary_for_threshold(sub, threshold_value=thr, methods=methods_here)
            if lt.empty:
                continue
            lt.insert(0, "exclude_last_weeks", excl)
            lt.insert(1, "surveillance_network", net)
            lt.insert(2, "age_group", age)
            lt.insert(3, "site", site)
            lt.insert(4, "threshold_rule", thr_rule)
            lt.insert(5, "threshold_value", thr)
            lead_rows.append(lt)
    alert_lead_time = pd.concat(lead_rows, ignore_index=True) if lead_rows else pd.DataFrame()
    alert_lead_time.to_csv(bench_dir / "alert_lead_time.tsv", sep="\t", index=False)

    # Probabilistic evaluation (intervals): focus national primary series (site=Overall, cohort primary ages).
    interval_rows: list[dict[str, object]] = []
    primary_ages = list(cohort.primary_respnet_ages)
    p = pred_long[
        (pred_long["site"].astype(str) == "Overall")
        & (pred_long["age_group"].isin(primary_ages))
        & (pred_long["train_scope"] == "within_site")
    ].copy()
    grp_cols_int = [
        "exclude_last_weeks",
        "surveillance_network",
        "age_group",
        "site",
        "train_scope",
        "method",
        "horizon_weeks",
    ]
    for key, g in p.groupby(grp_cols_int):
        g = g.sort_values("origin_epiweek")
        y = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float)
        yhat = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(yhat)
        if ok.sum() < 30:
            continue
        # 50% and 90% intervals
        for lvl, alpha in [(0.5, 0.5), (0.9, 0.1)]:
            lo = pd.to_numeric(g.get(f"pi{int(lvl*100)}_lo"), errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(g.get(f"pi{int(lvl*100)}_hi"), errors="coerce").to_numpy(dtype=float)
            ok2 = ok & np.isfinite(lo) & np.isfinite(hi)
            if ok2.sum() < 30:
                continue
            yy = y[ok2]
            ll = lo[ok2]
            hh = hi[ok2]
            cov = float(((yy >= ll) & (yy <= hh)).mean())
            width = float(np.mean(hh - ll))
            iscore = interval_score(yy, ll, hh, alpha=alpha)
            interval_rows.append(
                {
                    **{k: v for k, v in zip(grp_cols_int, key)},
                    "interval_level": float(lvl),
                    "alpha": float(alpha),
                    "n": int(ok2.sum()),
                    "coverage": cov,
                    "mean_width": width,
                    "mean_interval_score": float(np.mean(iscore)),
                }
            )

        # WIS using median=y_pred and available intervals.
        lo50 = pd.to_numeric(g.get("pi50_lo"), errors="coerce").to_numpy(dtype=float)
        hi50 = pd.to_numeric(g.get("pi50_hi"), errors="coerce").to_numpy(dtype=float)
        lo90 = pd.to_numeric(g.get("pi90_lo"), errors="coerce").to_numpy(dtype=float)
        hi90 = pd.to_numeric(g.get("pi90_hi"), errors="coerce").to_numpy(dtype=float)
        ok_w = ok & np.isfinite(lo50) & np.isfinite(hi50) & np.isfinite(lo90) & np.isfinite(hi90)
        if ok_w.sum() >= 30:
            yy = y[ok_w]
            m = yhat[ok_w]
            is50 = interval_score(yy, lo50[ok_w], hi50[ok_w], alpha=0.5)
            is90 = interval_score(yy, lo90[ok_w], hi90[ok_w], alpha=0.1)
            wis = (0.5 * np.abs(yy - m) + 0.25 * is50 + 0.05 * is90) / 2.5
            interval_rows.append(
                {
                    **{k: v for k, v in zip(grp_cols_int, key)},
                    "interval_level": float("nan"),
                    "alpha": float("nan"),
                    "n": int(ok_w.sum()),
                    "coverage": float("nan"),
                    "mean_width": float("nan"),
                    "mean_interval_score": float("nan"),
                    "mean_wis": float(np.mean(wis)),
                }
            )

    interval_eval = pd.DataFrame(interval_rows)
    if not interval_eval.empty and "mean_wis" not in interval_eval.columns:
        interval_eval["mean_wis"] = np.nan
    interval_eval.to_csv(bench_dir / "interval_eval.tsv", sep="\t", index=False)

    # External validity: per-site improvement vs seasonal naive, with CIs (moving-block bootstrap).
    base_transfer = pred_long[(pred_long["train_scope"] == "within_site") & (pred_long["method"] == "seasonal_naive")].copy()
    if not base_transfer.empty:
        base_transfer["train_scope"] = "train_on_overall"
        pred_for_generalization = pd.concat([pred_long, base_transfer], ignore_index=True)
    else:
        pred_for_generalization = pred_long

    rng_gen = np.random.default_rng(args.bootstrap_seed)
    site_generalization = _site_generalization_with_ci(
        pred_for_generalization,
        baseline_method="seasonal_naive",
        reps=args.bootstrap_reps,
        block_len=args.bootstrap_block_len,
        rng=rng_gen,
    )
    site_generalization.to_csv(bench_dir / "site_generalization.tsv", sep="\t", index=False)

    print(f"Wrote {bench_dir / 'predictions_long.tsv'}")
    print(f"Wrote {bench_dir / 'hyperparams.tsv'}")
    print(f"Wrote {bench_dir / 'forecast_eval.tsv'}")
    print(f"Wrote {bench_dir / 'method_benchmark.tsv'}")
    print(f"Wrote {bench_dir / 'paired_benchmark.tsv'}")
    print(f"Wrote {bench_dir / 'alert_utility.tsv'}")
    print(f"Wrote {bench_dir / 'expected_cost.tsv'}")
    print(f"Wrote {bench_dir / 'alert_lead_time.tsv'}")
    print(f"Wrote {bench_dir / 'interval_eval.tsv'}")
    print(f"Wrote {bench_dir / 'site_generalization.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
