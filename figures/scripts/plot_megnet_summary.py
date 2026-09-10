#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    rows = []
    with (ROOT / "results/megnet/predictions/megnet_main_test_predictions.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    vals = []
    for label, col in [("MEGNet", "abs_error_base_mev_atom"), ("Simple", "abs_error_simple_mev_atom"), ("FormDelta", "abs_error_formdelta_mev_atom")]:
        errs = [float(r[col]) for r in rows]
        vals.append((label, sum(errs) / len(errs)))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([v[0] for v in vals], [v[1] for v in vals], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("MAE (meV/atom)")
    ax.set_title("MEGNet same-split test MAE")
    fig.tight_layout()
    out = ROOT / "figures/rendered/megnet_main_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
