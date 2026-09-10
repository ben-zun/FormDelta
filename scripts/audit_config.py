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
    failures = []
    for path in sorted((root / "audits").glob("*config*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        pass_cols = [c for c in rows[0].keys()] if rows else []
        pass_col = ""
        for candidate in ("canonical_pass", "protocol_pass", "audit_pass", "pass", "config_pass"):
            if candidate in pass_cols:
                pass_col = candidate
                break
        if pass_col:
            bad = [r for r in rows if str(r.get(pass_col, "")).lower() not in {"true", "1", "yes"}]
            print(f"{path.relative_to(root)}: {len(rows) - len(bad)}/{len(rows)} pass")
            if bad:
                failures.append(path)
    if failures:
        raise SystemExit(f"Config audit failures: {failures}")


if __name__ == "__main__":
    main()
