#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formdelta.reference import regression_metrics


ROOT = Path(os.environ.get("FORMDELTA_ROOT", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_OUT = ROOT / "outputs" / "invdesflow_ranking_suite_20260904"
CURRENT_DATE = "2026-09-04"


@dataclass(frozen=True)
class Spec:
    label: str
    path: Path
    pred_col: str
    target_col: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-composition ranking and screening metrics for one or more prediction files.")
    parser.add_argument("--spec", action="append", required=True, help="label=csv_path::pred_col::target_col")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--near-degenerate-threshold-mev", type=float, default=25.0)
    return parser.parse_args()


def parse_spec(text: str) -> Spec:
    if "=" not in text:
        raise ValueError(f"Spec must look like label=path::pred::target, got: {text}")
    label, payload = text.split("=", 1)
    parts = payload.split("::")
    if len(parts) != 3:
        raise ValueError(f"Spec payload must look like path::pred::target, got: {payload}")
    return Spec(label=label.strip(), path=Path(parts[0]).resolve(), pred_col=parts[1].strip(), target_col=parts[2].strip())


def load_frame(spec: Spec, split: str) -> pd.DataFrame:
    frame = pd.read_csv(spec.path)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == str(split)].copy()
    if spec.pred_col not in frame.columns:
        raise KeyError(f"{spec.pred_col!r} not found in {spec.path}")
    if spec.target_col not in frame.columns:
        raise KeyError(f"{spec.target_col!r} not found in {spec.path}")
    formula_col = "formula" if "formula" in frame.columns else "reduced_formula_from_cif"
    if formula_col not in frame.columns:
        raise KeyError(f"No formula-like column found in {spec.path}")
    out = frame.copy()
    out["formula_group"] = out[formula_col].astype(str)
    out["y_true_eval"] = pd.to_numeric(out[spec.target_col], errors="coerce")
    out["y_pred_eval"] = pd.to_numeric(out[spec.pred_col], errors="coerce")
    out = out[np.isfinite(out["y_true_eval"]) & np.isfinite(out["y_pred_eval"])].copy()
    out["id"] = out["id"].astype(str) if "id" in out.columns else np.arange(len(out)).astype(str)
    return out


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    return float(frame["y_true"].corr(frame["y_pred"], method="spearman"))


def ranking_summary(frame: pd.DataFrame, *, threshold_mev: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_pairs = 0
    total_correct = 0
    total_ties = 0
    total_top1 = 0
    top1_correct = 0
    near_pairs = 0
    near_correct = 0
    near_ties = 0
    spearmans: list[float] = []
    spearman_pair_weights: list[int] = []

    for formula, group in frame.groupby("formula_group", sort=False):
        if len(group) < 2:
            continue
        y = group["y_true_eval"].to_numpy(dtype=np.float64)
        pred = group["y_pred_eval"].to_numpy(dtype=np.float64)
        all_pairs = []
        near_deg = []
        correct = 0
        ties = 0
        near_ok = 0
        near_eq = 0
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                delta_true = y[right] - y[left]
                if abs(delta_true) <= 1e-12:
                    continue
                all_pairs.append((left, right))
                delta_pred = pred[right] - pred[left]
                if abs(delta_pred) <= 1e-12:
                    ties += 1
                elif np.sign(delta_pred) == np.sign(delta_true):
                    correct += 1
                if abs(delta_true) * 1000.0 <= float(threshold_mev):
                    near_deg.append((left, right))
                    if abs(delta_pred) <= 1e-12:
                        near_eq += 1
                    elif np.sign(delta_pred) == np.sign(delta_true):
                        near_ok += 1
        if not all_pairs:
            continue
        top1_ok = int(np.argmin(y) == np.argmin(pred))
        rho = safe_spearman(y, pred)
        pair_count = len(all_pairs)
        near_count = len(near_deg)
        rows.append(
            {
                "formula_group": formula,
                "n_structures": int(len(group)),
                "pair_count": int(pair_count),
                "pairwise_order_accuracy": float(correct / pair_count),
                "pairwise_tie_rate": float(ties / pair_count),
                "top1_correct": bool(top1_ok),
                "spearman_rho": float(rho),
                "near_degenerate_pair_count": int(near_count),
                "near_degenerate_pair_accuracy": float(near_ok / near_count) if near_count else float("nan"),
                "near_degenerate_tie_rate": float(near_eq / near_count) if near_count else float("nan"),
                "true_energy_range_mev_atom": float((np.max(y) - np.min(y)) * 1000.0),
                "pred_energy_range_mev_atom": float((np.max(pred) - np.min(pred)) * 1000.0),
            }
        )
        total_pairs += pair_count
        total_correct += correct
        total_ties += ties
        total_top1 += 1
        top1_correct += top1_ok
        near_pairs += near_count
        near_correct += near_ok
        near_ties += near_eq
        if np.isfinite(rho):
            spearmans.append(float(rho))
            spearman_pair_weights.append(pair_count)

    long = pd.DataFrame(rows)
    weighted_spearman = float(np.average(spearmans, weights=spearman_pair_weights)) if spearmans else float("nan")
    summary = {
        "n_formula_groups": int(len(long)),
        "pair_count": int(total_pairs),
        "pairwise_order_accuracy": float(total_correct / total_pairs) if total_pairs else float("nan"),
        "pairwise_tie_rate": float(total_ties / total_pairs) if total_pairs else float("nan"),
        "top1_groups": int(total_top1),
        "top1_accuracy": float(top1_correct / total_top1) if total_top1 else float("nan"),
        "near_degenerate_pair_count": int(near_pairs),
        "near_degenerate_pair_accuracy": float(near_correct / near_pairs) if near_pairs else float("nan"),
        "near_degenerate_tie_rate": float(near_ties / near_pairs) if near_pairs else float("nan"),
        "spearman_group_mean": float(np.mean(spearmans)) if spearmans else float("nan"),
        "spearman_group_median": float(np.median(spearmans)) if spearmans else float("nan"),
        "spearman_pair_weighted_mean": weighted_spearman,
    }
    return long.sort_values(["pairwise_order_accuracy", "pair_count"], ascending=[True, False]).reset_index(drop=True), summary


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if len(frame) else "_empty_"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = [parse_spec(item) for item in args.spec]

    summary_rows: list[dict[str, Any]] = []
    long_rows: list[pd.DataFrame] = []

    for idx, spec in enumerate(specs):
        frame = load_frame(spec, args.split)
        reg = regression_metrics(
            frame["y_true_eval"].to_numpy(dtype=np.float64),
            frame["y_pred_eval"].to_numpy(dtype=np.float64),
        )
        group_long, rank = ranking_summary(frame, threshold_mev=args.near_degenerate_threshold_mev)
        group_long.insert(0, "method", spec.label)
        long_rows.append(group_long)
        summary_rows.append(
            {
                "method": spec.label,
                "n_test_rows": int(len(frame)),
                "mae_mev_atom": float(reg["mae_mev_atom"]),
                "rmse_mev_atom": float(reg["rmse_mev_atom"]),
                "median_abs_mev_atom": float(reg["median_abs_mev_atom"]),
                "p95_abs_mev_atom": float(reg["p95_abs_mev_atom"]),
                "bias_mev_atom": float(reg["bias_mev_atom"]),
                **rank,
                "predictions_csv": str(spec.path),
                "pred_col": spec.pred_col,
                "target_col": spec.target_col,
                "order": idx,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["order", "mae_mev_atom"]).drop(columns=["order"]).reset_index(drop=True)
    long_frame = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    summary.to_csv(args.out_dir / "ranking_summary.csv", index=False)
    long_frame.to_csv(args.out_dir / "formula_group_metrics.csv", index=False)

    lines = [
        "# Ranking Suite",
        "",
        f"Generated: {CURRENT_DATE}",
        "",
        f"Near-degenerate threshold: `{args.near_degenerate_threshold_mev:.1f}` meV/atom",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "generated": CURRENT_DATE,
        "split": str(args.split),
        "near_degenerate_threshold_mev": float(args.near_degenerate_threshold_mev),
        "summary_csv": str((args.out_dir / "ranking_summary.csv").resolve()),
        "formula_group_metrics_csv": str((args.out_dir / "formula_group_metrics.csv").resolve()),
        "methods": [spec.label for spec in specs],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
