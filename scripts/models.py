"""Ridge forecasts with chronological fitting and validation."""
from __future__ import annotations
import math
import zlib
from typing import Callable, Iterable
import numpy as np
import pandas as pd

def ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
    lam: float,
    *,
    ridge_preprocess: str,
    nonnegative_handling: str,
) -> float:
    if ridge_preprocess == "legacy":
        p = X_train.shape[1]
        XtX = X_train.T @ X_train
        A = XtX + lam * np.eye(p)
        b = X_train.T @ y_train
        beta = np.linalg.solve(A, b)
        pred = float(x_pred @ beta)
    elif ridge_preprocess == "standardized_intercept":
        mu = np.nanmean(X_train, axis=0)
        sd = np.nanstd(X_train, axis=0, ddof=0)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        Xz = (X_train - mu) / sd
        xz = (x_pred - mu) / sd
        Z = np.column_stack([np.ones(Xz.shape[0]), Xz])
        z_pred = np.concatenate([[1.0], xz])
        p = Z.shape[1]
        penalty = lam * np.eye(p)
        penalty[0, 0] = 0.0
        A = Z.T @ Z + penalty
        b = Z.T @ y_train
        beta = np.linalg.solve(A, b)
        pred = float(z_pred @ beta)
    else:
        raise ValueError(f"Unknown ridge_preprocess: {ridge_preprocess}")

    if nonnegative_handling == "truncate":
        return float(max(0.0, pred))
    if nonnegative_handling == "none":
        return pred
    raise ValueError(f"Unknown nonnegative_handling: {nonnegative_handling}")

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
    ridge_preprocess: str,
    nonnegative_handling: str,
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

    return ridge_fit_predict(
        X_train2,
        y_train2,
        x_pred,
        lam=lam,
        ridge_preprocess=ridge_preprocess,
        nonnegative_handling=nonnegative_handling,
    )

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
    ridge_preprocess: str,
    nonnegative_handling: str,
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
                ridge_preprocess=ridge_preprocess,
                nonnegative_handling=nonnegative_handling,
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
    tune_per_origin: bool,
    lam_grid: list[float],
    tune_min_train: int,
    tune_last_n: int,
    tune_warmup_ed: int,
    exclude_last_weeks: int,
    ridge_preprocess: str,
    nonnegative_handling: str,
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
                ridge_preprocess=ridge_preprocess,
                nonnegative_handling=nonnegative_handling,
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
                    ridge_preprocess=ridge_preprocess,
                    nonnegative_handling=nonnegative_handling,
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
                    ridge_preprocess=ridge_preprocess,
                    nonnegative_handling=nonnegative_handling,
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
                    ridge_preprocess=ridge_preprocess,
                    nonnegative_handling=nonnegative_handling,
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
                    ridge_preprocess=ridge_preprocess,
                    nonnegative_handling=nonnegative_handling,
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
                    ridge_preprocess=ridge_preprocess,
                    nonnegative_handling=nonnegative_handling,
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

    origin_lam_cache: dict[tuple[str, int, int], tuple[float, int]] = {}

    def _lambda_for_origin(
        *,
        method: str,
        X: np.ndarray,
        horizon: int,
        origin_i: int,
        fallback_lam: float,
    ) -> float:
        if not (tune_lam and tune_per_origin and run_ridge):
            return float(fallback_lam)
        key = (method, int(horizon), int(origin_i))
        if key not in origin_lam_cache:
            train_end_exclusive = int(origin_i - horizon)
            if train_end_exclusive <= 0:
                origin_lam_cache[key] = (float(fallback_lam), 0)
            else:
                origin_lam_cache[key] = tune_ridge_lambda(
                    X,
                    y,
                    time_block=time_block,
                    horizon=int(horizon),
                    tune_end_exclusive=train_end_exclusive,
                    tune_min_train=tune_min_train,
                    lam_grid=lam_grid,
                    tune_last_n=tune_last_n,
                    base_lam=float(fallback_lam),
                    ridge_preprocess=ridge_preprocess,
                    nonnegative_handling=nonnegative_handling,
                )
        tuned_lam, tuned_n = origin_lam_cache[key]
        hp_rows.append(
            {
                "train_scope": "within_site",
                "method": method,
                "horizon_weeks": int(horizon),
                "origin_idx": int(origin_i),
                "origin_epiweek": int(df.loc[origin_i, "epiweeks"]),
                "lambda": float(tuned_lam),
                "tune_n": int(tuned_n),
                "lam_grid": ",".join(str(x) for x in lam_grid),
                "tune_min_train": int(tune_min_train),
                "tune_last_n": int(tune_last_n),
                "tune_cutoff_index": int(origin_i - horizon),
            }
        )
        return float(tuned_lam)

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
                lam=_lambda_for_origin(
                    method="ridge_univariate",
                    X=X_base,
                    horizon=h,
                    origin_i=i,
                    fallback_lam=float(lam_uni[h]),
                ),
                ridge_preprocess=ridge_preprocess,
                nonnegative_handling=nonnegative_handling,
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
                        lam=_lambda_for_origin(
                            method="ridge_with_ed",
                            X=X_ed,
                            horizon=h,
                            origin_i=i,
                            fallback_lam=float(lam_ed[h]),
                        ),
                        ridge_preprocess=ridge_preprocess,
                        nonnegative_handling=nonnegative_handling,
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
                lam=_lambda_for_origin(
                    method="ridge_with_signals",
                    X=X_sig,
                    horizon=h,
                    origin_i=i,
                    fallback_lam=float(lam_sig[h]),
                ),
                ridge_preprocess=ridge_preprocess,
                nonnegative_handling=nonnegative_handling,
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
                lam=_lambda_for_origin(
                    method="ridge_with_signals_plus_flu_pos",
                    X=X_sig_flupos,
                    horizon=h,
                    origin_i=i,
                    fallback_lam=float(lam_sig_flupos[h]),
                ),
                ridge_preprocess=ridge_preprocess,
                nonnegative_handling=nonnegative_handling,
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
                        lam=_lambda_for_origin(
                            method="ridge_with_signals_plus_ed",
                            X=X_sig_plus_ed,
                            horizon=h,
                            origin_i=i,
                            fallback_lam=float(lam_sig_ed[h]),
                        ),
                        ridge_preprocess=ridge_preprocess,
                        nonnegative_handling=nonnegative_handling,
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
                        lam=_lambda_for_origin(
                            method="ridge_with_signals_plus_flu_pos_plus_ed",
                            X=X_sig_flupos_plus_ed,
                            horizon=h,
                            origin_i=i,
                            fallback_lam=float(lam_sig_flupos_ed[h]),
                        ),
                        ridge_preprocess=ridge_preprocess,
                        nonnegative_handling=nonnegative_handling,
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