#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formdelta.direct_reference_fegx import write_direct_reference_contract


ROOT = Path(os.environ.get("FORMDELTA_ROOT", Path(__file__).resolve().parents[1])).resolve()
ADAPTER = ROOT / "scripts/train_simple_residual.py"
TRAIN_FEGX = ROOT / "scripts/train_formdelta.py"
CURRENT_DATE = date.today().isoformat()
METRIC_KEYS = (
    "n",
    "mae_mev_atom",
    "rmse_mev_atom",
    "median_abs_mev_atom",
    "p90_abs_mev_atom",
    "p95_abs_mev_atom",
    "p99_abs_mev_atom",
    "bias_mev_atom",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-model direct-reference FE-GX: frozen prediction -> fixed reference -> residual adapter."
    )
    parser.add_argument("--prediction-csv", action="append", required=True, help="Prediction CSV. Repeatable.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--structure-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workspace-root", default=str(ROOT))
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--split-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--energy-column", default=None)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-subset-seed", type=int, default=20260529)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--adapter-arg",
        action="append",
        default=[],
        help="Extra token appended to the adapter command. Repeat per token.",
    )
    parser.add_argument(
        "--fegx-arg",
        action="append",
        default=[],
        help="Extra token appended to the FE-GX command. Repeat per token.",
    )
    return parser.parse_args()


def run_logged(cmd: list[str], log_path: Path, *, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(cmd)}\n\n")
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - start
        log.write(f"\n[run_logged] exit_code={proc.returncode} runtime_s={elapsed:.3f}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}\nSee log: {log_path}")


def maybe_run(cmd: list[str], log_path: Path, done_path: Path, *, cwd: Path, force: bool) -> None:
    if done_path.exists() and not force:
        return
    run_logged(cmd, log_path, cwd=cwd)
    if not done_path.exists():
        raise FileNotFoundError(f"Expected output was not produced: {done_path}")


def metric_row(stage: str, variant: str, split: str, stats: dict[str, Any], source: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": stage,
        "variant": variant,
        "split": split,
        "reference_mode": "direct_reference",
        "source_metrics": str(source),
    }
    for key in METRIC_KEYS:
        row[key] = stats.get(key)
    return row


def add_relative_columns(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    direct_lookup = {
        str(row["split"]): float(row["mae_mev_atom"])
        for _, row in summary[summary["stage"] == "direct_raw"].dropna(subset=["mae_mev_atom"]).iterrows()
    }
    out = summary.copy()
    out["delta_vs_direct_mae_mev_atom"] = out.apply(
        lambda row: (float(row["mae_mev_atom"]) - direct_lookup[str(row["split"])])
        if pd.notna(row["mae_mev_atom"]) and str(row["split"]) in direct_lookup
        else None,
        axis=1,
    )
    return out


def build_report(summary: pd.DataFrame, out_path: Path) -> None:
    test = summary[summary["split"] == "test"].copy()
    lines = [
        "# Direct-Reference Single-Model FE-GX",
        "",
        f"Generated: {CURRENT_DATE}",
        "",
        "Pipeline:",
        "",
        "1. frozen single-model prediction z(x);",
        "2. fixed direct reference R(x)=z(x);",
        "3. simple residual adapter on target data;",
        "4. single-model FE-GX residual training with the reference path fixed.",
        "",
        "## Test Summary",
        "",
        "| stage | variant | test MAE (meV/atom) | delta vs direct |",
        "|---|---|---:|---:|",
    ]
    for row in test.sort_values(["mae_mev_atom", "stage", "variant"], na_position="last").itertuples(index=False):
        mae = "" if pd.isna(row.mae_mev_atom) else f"{float(row.mae_mev_atom):.3f}"
        delta = "" if pd.isna(row.delta_vs_direct_mae_mev_atom) else f"{float(row.delta_vs_direct_mae_mev_atom):.3f}"
        lines.append(f"| {row.stage} | {row.variant} | {mae} | {delta} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    contract_dir = out_dir / "direct_reference"
    contract_summary = contract_dir / "contract_summary.json"
    if args.force or not contract_summary.exists():
        write_direct_reference_contract(
            prediction_csvs=[Path(path).resolve() for path in args.prediction_csv],
            dataset_root=Path(args.dataset_root).resolve(),
            workspace_root=Path(args.workspace_root).resolve(),
            feature_cache=Path(args.feature_cache).resolve(),
            out_dir=contract_dir,
            id_column=args.id_column,
            split_column=args.split_column,
            target_column=args.target_column,
            energy_column=args.energy_column,
        )
    contract_metrics = json.loads(contract_summary.read_text(encoding="utf-8"))

    calibrated_predictions = contract_dir / "calibrated_predictions.csv"
    coefficient_csv = contract_dir / "calibration_coefficients.csv"
    primary_missing_rows = int(contract_metrics["input"]["primary_missing_rows"])

    adapter_dir = out_dir / "adapter"
    adapter_metrics = adapter_dir / "metrics.json"
    adapter_cmd = [
        py,
        str(ADAPTER),
        "--calibrated-predictions",
        str(calibrated_predictions),
        "--feature-cache",
        str(Path(args.feature_cache).resolve()),
        "--out-dir",
        str(adapter_dir),
        "--baseline-col",
        "pred_element_reference",
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--train-fraction",
        str(args.train_fraction),
        "--train-subset-seed",
        str(args.train_subset_seed),
    ]
    adapter_cmd.extend(args.adapter_arg)
    maybe_run(adapter_cmd, out_dir / "logs/adapter.log", adapter_metrics, cwd=ROOT, force=args.force)
    adapter_payload = json.loads(adapter_metrics.read_text(encoding="utf-8"))

    fegx_dir = out_dir / "fegx"
    fegx_metrics = fegx_dir / "metrics.json"
    exp_name = f"direct_reference_fegx_single_seed{args.seed}"
    fegx_cmd = [
        py,
        str(TRAIN_FEGX),
        "--calibrated-predictions",
        str(calibrated_predictions),
        "--feature-cache",
        str(Path(args.feature_cache).resolve()),
        "--structure-cache",
        str(Path(args.structure_cache).resolve()),
        "--coefficient-csv",
        str(coefficient_csv),
        "--out-dir",
        str(fegx_dir),
        "--exp-name",
        exp_name,
        "--reference-lr",
        "0.0",
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--train-fraction",
        str(args.train_fraction),
        "--train-subset-seed",
        str(args.train_subset_seed),
    ]
    if primary_missing_rows > 0 and "--use-primary-missing-branch" not in args.fegx_arg:
        fegx_cmd.append("--use-primary-missing-branch")
    fegx_cmd.extend(args.fegx_arg)
    maybe_run(fegx_cmd, out_dir / "logs/fegx.log", fegx_metrics, cwd=ROOT, force=args.force)
    fegx_payload = json.loads(fegx_metrics.read_text(encoding="utf-8"))

    summary_rows: list[dict[str, Any]] = []
    for split, stats in contract_metrics["raw_model_energy_as_formation"].items():
        summary_rows.append(metric_row("direct_raw", "raw_model_energy_as_formation", str(split), dict(stats), contract_summary))
    for split, stats in contract_metrics["filled_direct_reference"].items():
        summary_rows.append(metric_row("direct_reference_contract", "direct_reference", str(split), dict(stats), contract_summary))
    for split, stats in adapter_payload.get("baseline", {}).items():
        summary_rows.append(metric_row("adapter_baseline", "direct_reference_simple_residual_baseline", str(split), dict(stats), adapter_metrics))
    for split, block in adapter_payload.get("adapter", {}).items():
        summary_rows.append(metric_row("simple_residual_adapter", "direct_reference_simple_residual", str(split), block.get("overall", {}), adapter_metrics))
    for split, stats in fegx_payload.get("baseline", {}).items():
        summary_rows.append(metric_row("fegx_baseline", "direct_reference_fegx_single_baseline", str(split), dict(stats), fegx_metrics))
    for split, block in fegx_payload.get(exp_name, {}).items():
        summary_rows.append(metric_row("single_model_fegx", "direct_reference_fegx_single", str(split), block.get("overall", {}), fegx_metrics))

    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary = add_relative_columns(summary)
        summary = summary.sort_values(["split", "mae_mev_atom", "stage", "variant"], na_position="last").reset_index(drop=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

    summary_json = {
        "generated": CURRENT_DATE,
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "device": str(args.device),
        "train_fraction": float(args.train_fraction),
        "train_subset_seed": int(args.train_subset_seed),
        "prediction_csv": [str(Path(path).resolve()) for path in args.prediction_csv],
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "feature_cache": str(Path(args.feature_cache).resolve()),
        "structure_cache": str(Path(args.structure_cache).resolve()),
        "contract_summary": str(contract_summary),
        "summary_csv": str((out_dir / "summary.csv").resolve()),
        "best_test_rows": summary[summary["split"] == "test"].sort_values("mae_mev_atom").head(10).to_dict("records")
        if len(summary)
        else [],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_report(summary, out_dir / "README.md")

    if len(summary):
        test = summary[summary["split"] == "test"].sort_values("mae_mev_atom")
        best = test.iloc[0]
        print(
            "direct-reference FE-GX complete | "
            f"best={best['variant']} | stage={best['stage']} | "
            f"test_mae={float(best['mae_mev_atom']):.3f} meV/atom",
            flush=True,
        )


if __name__ == "__main__":
    main()
