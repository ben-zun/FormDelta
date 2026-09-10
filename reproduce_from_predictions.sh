#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON:-python3}"
"${PYTHON_BIN}" scripts/evaluate.py --snapshot-root . --check-only
"${PYTHON_BIN}" scripts/audit_config.py --snapshot-root .
"${PYTHON_BIN}" scripts/audit_leakage.py --snapshot-root .
