#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED = {
    "MEGNet base": 83.372,
    "MEGNet FormDelta": 62.090,
    "MatRIS FormDelta": 29.27,
    "EqV3 FormDelta": 30.06,
    "MACE FormDelta": 33.01,
    "SevenNet FormDelta": 61.43,
    "CHGNet FormDelta": 28.15,
    "EqV3-LoRA FormDelta": 11.99,
    "ALIGNN FormDelta": 68.18,
    "r2SCAN FormDelta full-label": 17.81,
    "r2SCAN classical delta full-label": 16.42,
}


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mae(rows, pred_col):
    vals = []
    for row in rows:
        vals.append(abs(float(row[pred_col]) - float(row["y_true"])) * 1000.0)
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=Path("."))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.snapshot_root
    pred = root / "results/megnet/predictions/megnet_main_test_predictions.csv"
    if pred.exists():
        rows = read_rows(pred)
        print(f"MEGNet test rows: {len(rows)}")
        print(f"MEGNet base MAE: {mae(rows, 'y_base'):.3f} meV/atom")
        print(f"MEGNet simple residual MAE: {mae(rows, 'y_simple'):.3f} meV/atom")
        print(f"MEGNet FormDelta MAE: {mae(rows, 'y_formdelta'):.3f} meV/atom")
    for label, expected in EXPECTED.items():
        print(f"expected {label}: {expected}")


if __name__ == "__main__":
    main()
