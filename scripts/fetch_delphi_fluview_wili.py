#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import requests

from utils import RESULTS_DIR, SNAPSHOT_DIR, append_tsv, relpath, sha256_file, utc_timestamp, write_gzip_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Delphi Epidata fluview wILI (national) and store a snapshot + retrieval log.")
    ap.add_argument("--regions", default="nat", help="Epidata region code (default: nat)")
    ap.add_argument("--epiweeks", default="199740-202652", help="Epiweek range like 201001-202652 (default: wide)")
    args = ap.parse_args()

    ts = utc_timestamp()
    outdir = SNAPSHOT_DIR / "delphi_fluview_wili"
    outpath = outdir / f"{ts}.json.gz"

    params = {"source": "fluview", "regions": args.regions, "epiweeks": args.epiweeks}
    r = requests.get("https://api.delphi.cmu.edu/epidata/api.php", params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    write_gzip_json(outpath, payload)
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
    n_rows = len(payload.get("epidata", []) or [])
    append_tsv(
        RESULTS_DIR / "dataset_retrieval_log.tsv",
        {
            "dataset_id": "delphi_fluview_wili",
            "retrieval_datetime_utc": ts,
            "source": "api.delphi.cmu.edu/epidata (fluview)",
            "version_or_revision_tag": "",
            "snapshot_path": str(outpath.relative_to(RESULTS_DIR.parent)),
            "sha256": checksum,
            "n_rows": n_rows,
            "where": "",
            "select": "",
            "order": "",
        },
        fieldnames=log_fields,
    )

    print(f"Wrote snapshot: {relpath(outpath)} ({n_rows} rows)")
    print(f"Updated retrieval log: {relpath(RESULTS_DIR / 'dataset_retrieval_log.tsv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
