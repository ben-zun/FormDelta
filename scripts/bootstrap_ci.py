#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--base-col", required=True)
    parser.add_argument("--method-col", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    deltas = [float(r[args.base_col]) - float(r[args.method_col]) for r in rows]
    rng = random.Random(args.seed)
    means = []
    for _ in range(args.samples):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    point = sum(deltas) / len(deltas)
    print(f"delta mean={point:.6g}, ci95=({lo:.6g}, {hi:.6g}), n={len(deltas)}")


if __name__ == "__main__":
    main()
