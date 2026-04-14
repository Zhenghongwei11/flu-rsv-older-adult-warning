from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CohortProfile:
    profile_id: str
    display_name: str
    primary_respnet_ages: tuple[str, ...]
    respnet_age_groups_for_tables: tuple[str, ...]
    nssp_ed_age_map: Mapping[str, str]  # NSSP demographics_values -> suffix used in feature names
    figure_suptitle: str
    key_exemplar_age: str


COHORTS: dict[str, CohortProfile] = {
    "older_adult_65plus": CohortProfile(
        profile_id="older_adult_65plus",
        display_name="Older adults (65 years and older)",
        primary_respnet_ages=("65+ yr",),
        respnet_age_groups_for_tables=("65+ yr", "Overall"),
        nssp_ed_age_map={"65+ years": "65p"},
        figure_suptitle="Older-adult (65 years and older) hospitalization burden time series (RESP-NET)",
        key_exemplar_age="65+ yr",
    ),
    "older_adult_strata": CohortProfile(
        profile_id="older_adult_strata",
        display_name="Older adults (65–74 / 75–84 / 85 years and older)",
        primary_respnet_ages=("65-74 yr", "75-84 yr", "85+ yr"),
        respnet_age_groups_for_tables=("65-74 yr", "75-84 yr", "85+ yr", "Overall"),
        # Public NSSP ED export used in this repo (7xva-uux8 snapshot) does NOT provide these strata.
        # Keep ED mapping empty to avoid any implicit proxying.
        nssp_ed_age_map={},
        figure_suptitle="Older-adult hospitalization burden time series by age strata (RESP-NET)",
        key_exemplar_age="75-84 yr",
    ),
}


def cohort_from_env(default: str = "older_adult_65plus") -> str:
    return os.environ.get("COHORT_PROFILE", default)


def get_cohort(profile_id: str) -> CohortProfile:
    pid = str(profile_id).strip()
    if pid not in COHORTS:
        raise ValueError(f"Unknown cohort profile: {pid!r}. Expected one of: {', '.join(sorted(COHORTS))}")
    return COHORTS[pid]
