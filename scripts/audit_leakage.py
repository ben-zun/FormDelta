#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.snapshot_root
    for path in sorted((root / "audits").glob("*leakage*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        central_failures = 0
        for row in rows:
            is_fail_key = str(row.get("is_fail_key", "")).lower() in {"true", "1", "yes"}
            try:
                n_overlap = float(row.get("n_overlap", 0) or 0)
            except ValueError:
                n_overlap = 0
            if is_fail_key and n_overlap > 0:
                central_failures += 1
        print(f"{path.relative_to(root)}: central leakage failures={central_failures}")
        if central_failures:
            raise SystemExit(f"Leakage audit failed for {path}")


if __name__ == "__main__":
    main()
