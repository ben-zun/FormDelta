#!/usr/bin/env bash
set -euo pipefail

paths=(
  "data/cache/composition_features.csv"
  "data/cache/structure_metadata.csv"
  "data/manifests/historical_dataset_manifest_20260626.csv"
)

for path in "${paths[@]}"; do
  gz="${path}.gz"
  if [[ -f "${path}" ]]; then
    echo "present: ${path}"
  elif [[ -f "${gz}" ]]; then
    gzip -dk "${gz}"
    echo "materialized: ${path}"
  else
    echo "missing: ${gz}" >&2
    exit 1
  fi
done
