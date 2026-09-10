#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.snapshot_root
    result_dir = root / "results/cross_backbone"
    print(f"cross_backbone:")
    for path in sorted(result_dir.glob("*.csv")):
        print(f"  {path.relative_to(root)}")
    print("Use retrain_adapters.sh for full adapter retraining and reproduce_from_predictions.sh for read-only metric checks.")


if __name__ == "__main__":
    main()
