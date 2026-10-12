"""Paired forecast comparisons and deterministic resampling seeds."""
from __future__ import annotations
import zlib
from typing import Iterable
import pandas as pd
MATCH_KEYS=["origin_epiweek","origin_week_ending_date","target_epiweek","target_week_ending_date"]

def stable_seed(base_seed: int, parts: Iterable[object]) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return (int(base_seed) + zlib.adler32(s)) % (2**32 - 1)

def matched_pair(
    sub: pd.DataFrame,
    *,
    comparator: str,
    candidate: str,
) -> pd.DataFrame:
    keep = MATCH_KEYS + ["y_true", "y_pred"]
    comp = (
        sub[sub["method"] == comparator][keep]
        .rename(columns={"y_pred": "comparator_pred"})
        .dropna(subset=MATCH_KEYS + ["y_true", "comparator_pred"])
    )
    cand = (
        sub[sub["method"] == candidate][MATCH_KEYS + ["y_pred"]]
        .rename(columns={"y_pred": "candidate_pred"})
        .dropna(subset=MATCH_KEYS + ["candidate_pred"])
    )
    merged = comp.merge(cand, on=MATCH_KEYS, how="inner")
    if merged.empty:
        return merged
    merged = merged.sort_values(["origin_week_ending_date", "target_week_ending_date", "target_epiweek"])
    return merged.reset_index(drop=True)