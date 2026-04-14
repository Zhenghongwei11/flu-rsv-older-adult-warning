#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import time
from typing import Any

import requests

from utils import RESULTS_DIR, SNAPSHOT_DIR, append_tsv, relpath, sha256_file, utc_timestamp, write_gzip_csv


def fetch_all_rows(
    dataset_id: str,
    where: str | None,
    select: str | None,
    order: str | None,
    limit: int,
    app_token: str | None,
) -> list[dict[str, Any]]:
    base = f"https://data.cdc.gov/resource/{dataset_id}.json"
    headers: dict[str, str] = {}
    if app_token:
        headers["X-App-Token"] = app_token

    out: list[dict[str, Any]] = []
    offset = 0

    while True:
        params: dict[str, str] = {"$limit": str(limit), "$offset": str(offset)}
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select
        if order:
            params["$order"] = order

        r = requests.get(base, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        offset += limit
        time.sleep(0.1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch CDC Open Data (Socrata) dataset and store a snapshot + retrieval log.")
    ap.add_argument("--dataset-id", required=True, help="Socrata 4x4 dataset id, e.g. kvib-3txy")
    ap.add_argument("--outdir", default=str(SNAPSHOT_DIR), help="Base snapshot directory (default: data/snapshots/)")
    ap.add_argument("--where", default=None, help="Socrata $where filter (optional)")
    ap.add_argument("--select", default=None, help="Socrata $select projection (optional)")
    ap.add_argument("--order", default=None, help="Socrata $order (optional)")
    ap.add_argument("--page-size", type=int, default=50000, help="Pagination size (default: 50000)")
    ap.add_argument("--format", choices=["csv"], default="csv", help="Snapshot format (default: csv)")
    args = ap.parse_args()

    app_token = None
    # Optional: user can set SOCRATA_APP_TOKEN to be nicer to the API.
    import os

    app_token = os.environ.get("SOCRATA_APP_TOKEN")

    ts = utc_timestamp()
    snapshot_base = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    outdir = SNAPSHOT_DIR / f"cdc_{args.dataset_id}"
    outpath = outdir / f"{ts}.csv.gz"

    rows = fetch_all_rows(
        dataset_id=args.dataset_id,
        where=args.where,
        select=args.select,
        order=args.order,
        limit=args.page_size,
        app_token=app_token,
    )

    if not rows:
        raise SystemExit(f"No rows returned for {args.dataset_id}. Check filters.")

    # Stable field list: if user provided --select, use those aliases; otherwise write all keys union.
    if args.select:
        # For select with aliases, we cannot easily infer column order without parsing.
        # Keep a stable order by sorting observed keys.
        keys = sorted({k for r in rows for k in r.keys()})
    else:
        keys = sorted({k for r in rows for k in r.keys()})

    n = write_gzip_csv(outpath, rows, fieldnames=keys)
    checksum = sha256_file(outpath)

    log_fields = [
        "dataset_id",
        "retrieval_datetime_utc",
        "source",
        "version_or_revision_tag",
        "snapshot_path",
        "sha256",
        "n_rows",
        "where",
        "select",
        "order",
    ]
    append_tsv(
        RESULTS_DIR / "dataset_retrieval_log.tsv",
        {
            "dataset_id": f"cdc_{args.dataset_id}",
            "retrieval_datetime_utc": ts,
            "source": "data.cdc.gov (Socrata)",
            "version_or_revision_tag": "",
            "snapshot_path": str(outpath.relative_to(RESULTS_DIR.parent)),
            "sha256": checksum,
            "n_rows": n,
            "where": args.where or "",
            "select": args.select or "",
            "order": args.order or "",
        },
        fieldnames=log_fields,
    )

    print(f"Wrote snapshot: {relpath(outpath)} ({n} rows)")
    print(f"Updated retrieval log: {relpath(RESULTS_DIR / 'dataset_retrieval_log.tsv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
