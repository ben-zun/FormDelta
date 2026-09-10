#!/usr/bin/env bash
set -euo pipefail
: "${FORMDELTA_DEVICE:=cuda}"
PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" scripts/reproduce_megnet.py   --prediction-csv data/foundation_predictions/megnet_frozen_predictions.csv   --dataset-root data/splits/megnet_medium_same_split_269   --feature-cache data/cache/composition_features.csv   --structure-cache data/cache/structure_metadata.csv   --out-dir outputs/megnet_retrain_seed20260528   --workspace-root .   --seed 20260528   --device "${FORMDELTA_DEVICE}"
"${PYTHON_BIN}" scripts/reproduce_cross_backbone.py --snapshot-root .
"${PYTHON_BIN}" scripts/reproduce_r2scan.py --snapshot-root .
"${PYTHON_BIN}" scripts/reproduce_wbm.py --snapshot-root .
"${PYTHON_BIN}" scripts/reproduce_ranking.py --snapshot-root .
