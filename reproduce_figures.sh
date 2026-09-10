#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" figures/scripts/plot_megnet_summary.py
