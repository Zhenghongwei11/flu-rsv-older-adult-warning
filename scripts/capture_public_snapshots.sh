#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
RUN_ID="$(date -u +"%Y-%m-%dT%H%M%SZ")"
LOG="logs/snapshot_${RUN_ID}.log"

{
  echo "[capture_public_snapshots] run_id=${RUN_ID}"

  echo "[capture_public_snapshots] source=cdc_kvib-3txy"
  python3 scripts/fetch_socrata.py \
    --dataset-id kvib-3txy \
    --where "rate_type='Observed' AND surveillance_network in ('FluSurv-NET','RSV-NET')" \
    --order "date ASC"

  echo "[capture_public_snapshots] source=cdc_seuz-s2cv"
  python3 scripts/fetch_socrata.py \
    --dataset-id seuz-s2cv \
    --order "week_end ASC"

  echo "[capture_public_snapshots] source=cdc_7xva-uux8"
  python3 scripts/fetch_socrata.py \
    --dataset-id 7xva-uux8 \
    --where "geography='United States' AND demographics_type='Age Group' AND pathogen in ('Influenza','RSV')" \
    --order "week_end ASC"

  echo "[capture_public_snapshots] source=delphi_fluview_wili"
  python3 scripts/fetch_delphi_fluview_wili.py --regions nat --epiweeks 201001-202652

  echo "[capture_public_snapshots] done"
} | tee "$LOG"

echo "Log: $LOG"
