#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition


ROOT = Path(os.environ.get("FORMDELTA_ROOT", Path(__file__).resolve().parents[1])).resolve()
RUNS = ROOT / "results"
DEFAULT_RUN_DIR = RUNS / "direct_reference_fegx_single_invdesflow_20260903"
DEFAULT_OVERLAP_DIR = RUNS / "checkpoint_target_overlap_audit_20260904"
DEFAULT_SOURCE_JSON = ROOT / "external_data" / "formation_energy" / "mp.2018.6.1.json"
DEFAULT_OUT = ROOT / "outputs" / "invdesflow_megnet_source_overlap_subset_audit_20260904"
CURRENT_DATE = "2026-09-04"

FORMULA_STRUCTURAL_RE = re.compile(r"_chemical_formula_structural\s+([^\n\r]+)")
FORMULA_SUM_RE = re.compile(r"_chemical_formula_sum\s+'([^']+)'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit current MEGNet FE-GX predictions on source-overlap seen/unseen target subsets.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--overlap-dir", type=Path, default=DEFAULT_OVERLAP_DIR)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def configure_warnings() -> None:
    warnings.filterwarnings("ignore", message="No Pauling electronegativity for .*", category=UserWarning)


def reduced_formula(formula: str) -> str:
    return str(Composition(formula).reduced_formula)


def chemical_system(formula: str) -> str:
    return "-".join(sorted(str(el) for el in Composition(formula).elements))


def parse_source_formula(structure_text: str) -> str | None:
    match = FORMULA_STRUCTURAL_RE.search(structure_text) or FORMULA_SUM_RE.search(structure_text)
    if match is None:
        return None
    formula = str(match.group(1)).strip().strip("'").strip('"')
    return formula or None


def load_source_sets(source_json: Path) -> tuple[set[str], set[str]]:
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    reduced: set[str] = set()
    chemsys: set[str] = set()
    for record in payload:
        if not isinstance(record, dict):
            continue
        structure_text = record.get("structure")
        if not isinstance(structure_text, str):
            continue
        formula = parse_source_formula(structure_text)
        if not formula:
            continue
        try:
            reduced_formula_value = reduced_formula(formula)
            chemical_system_value = chemical_system(formula)
        except Exception:
            continue
        reduced.add(reduced_formula_value)
        chemsys.add(chemical_system_value)
    return reduced, chemsys


def load_current_predictions(run_dir: Path) -> pd.DataFrame:
    frozen = pd.read_csv(run_dir / "direct_reference" / "calibrated_predictions.csv")
    frozen = frozen.loc[frozen["split"].astype(str) == "test", ["id", "formula", "y_true", "pred_raw_model_energy"]].copy()
    simple = pd.read_csv(run_dir / "adapter" / "fe_adapter_predictions.csv")
    simple = simple.loc[simple["split"].astype(str) == "test", ["id", "pred_fe_adapter"]].copy()
    fegx = pd.read_csv(run_dir / "fegx" / "direct_reference_fegx_single_seed20260528_predictions.csv")
    fegx = fegx.loc[
        fegx["split"].astype(str) == "test",
        ["id", "pred_direct_reference_fegx_single_seed20260528"],
    ].copy()
    merged = frozen.merge(simple, on="id", how="inner", validate="one_to_one").merge(fegx, on="id", how="inner", validate="one_to_one")
    merged["id"] = merged["id"].astype(str)
    merged["formula"] = merged["formula"].astype(str)
    merged["reduced_formula"] = merged["formula"].map(reduced_formula)
    merged["target_chemical_system"] = merged["formula"].map(chemical_system)
    return merged


def load_structure_seen_ids(overlap_dir: Path) -> set[str]:
    hits = pd.read_csv(overlap_dir / "checkpoint_overlap_hits.csv")
    mask = (
        (hits["model"].astype(str) == "MEGNet")
        & (hits["split_scope"].astype(str) == "test")
        & (hits["overlap_kind"].astype(str) == "structure_matcher")
    )
    return set(hits.loc[mask, "target_id"].astype(str))


def verify_against_overlap_audit(frame: pd.DataFrame, overlap_dir: Path) -> None:
    audit = pd.read_csv(overlap_dir / "checkpoint_overlap_audit.csv")
    row = audit[(audit["model"].astype(str) == "MEGNet") & (audit["split_scope"].astype(str) == "test")]
    if len(row) != 1:
        raise RuntimeError("Expected exactly one MEGNet/test overlap-audit row.")
    item = row.iloc[0]
    expected = {
        "source_structure_seen": int(item["structure_overlap"]),
        "source_formula_seen": int(item["composition_overlap"]),
        "source_chemical_system_seen": int(item["chemical_system_overlap"]),
    }
    observed = {
        "source_structure_seen": int(frame["source_structure_seen"].sum()),
        "source_formula_seen": int(frame["source_formula_seen"].sum()),
        "source_chemical_system_seen": int(frame["source_chemical_system_seen"].sum()),
    }
    if expected != observed:
        raise RuntimeError(f"Source-overlap subset counts do not match the checkpoint audit. expected={expected} observed={observed}")


def regression_row(subset: pd.DataFrame, *, subset_name: str, method: str, pred_col: str) -> dict[str, object]:
    errors = (pd.to_numeric(subset[pred_col], errors="coerce") - pd.to_numeric(subset["y_true"], errors="coerce")) * 1000.0
    abs_errors = np.abs(errors.to_numpy(dtype=np.float64))
    err = errors.to_numpy(dtype=np.float64)
    return {
        "subset": subset_name,
        "method": method,
        "n_test_rows": int(len(subset)),
        "mae_mev_atom": float(abs_errors.mean()),
        "rmse_mev_atom": float(np.sqrt(np.mean(np.square(err)))),
        "bias_mev_atom": float(err.mean()),
        "p95_abs_mev_atom": float(np.quantile(abs_errors, 0.95)),
    }


def build_subset_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    subset_defs = (
        ("source_structure_seen", frame["source_structure_seen"]),
        ("source_structure_unseen", ~frame["source_structure_seen"]),
        ("source_formula_seen", frame["source_formula_seen"]),
        ("source_formula_unseen", ~frame["source_formula_seen"]),
        ("source_chemical_system_seen", frame["source_chemical_system_seen"]),
        ("source_chemical_system_unseen", ~frame["source_chemical_system_seen"]),
    )
    rows: list[dict[str, object]] = []
    for subset_name, mask in subset_defs:
        subset = frame.loc[mask].copy()
        rows.append(regression_row(subset, subset_name=subset_name, method="frozen", pred_col="pred_raw_model_energy"))
        rows.append(regression_row(subset, subset_name=subset_name, method="simple_residual", pred_col="pred_fe_adapter"))
        rows.append(
            regression_row(
                subset,
                subset_name=subset_name,
                method="fegx",
                pred_col="pred_direct_reference_fegx_single_seed20260528",
            )
        )
    return pd.DataFrame(rows)


def build_delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(index="subset", columns="method", values="mae_mev_atom")
    rows = []
    for subset_name in pivot.index:
        rows.append(
            {
                "subset": subset_name,
                "fegx_minus_frozen_mae_mev_atom": float(pivot.loc[subset_name, "fegx"] - pivot.loc[subset_name, "frozen"]),
                "fegx_minus_simple_residual_mae_mev_atom": float(pivot.loc[subset_name, "fegx"] - pivot.loc[subset_name, "simple_residual"]),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    *,
    out_path: Path,
    flags: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
) -> None:
    lines = [
        "# InvDesFlow MEGNet Source-Overlap Subset Audit",
        "",
        f"Generated: {CURRENT_DATE}",
        "",
        "This audit slices the current MEGNet target test set by whether the MP2018 pretraining source already contains the same structure, reduced formula, or chemical system.",
        "",
        "## Test overlap flag counts",
        "",
        flags[
            [
                "source_structure_seen",
                "source_formula_seen",
                "source_chemical_system_seen",
            ]
        ].sum()
        .rename("count")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "flag"})
        .to_markdown(index=False),
        "",
        "## Subset metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## FE-GX MAE deltas",
        "",
        deltas.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    configure_warnings()
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_current_predictions(args.run_dir)
    structure_seen_ids = load_structure_seen_ids(args.overlap_dir)
    source_reduced, source_chemsys = load_source_sets(args.source_json)
    frame["source_structure_seen"] = frame["id"].isin(structure_seen_ids)
    frame["source_formula_seen"] = frame["reduced_formula"].isin(source_reduced)
    frame["source_chemical_system_seen"] = frame["target_chemical_system"].isin(source_chemsys)
    verify_against_overlap_audit(frame, args.overlap_dir)

    flags = frame[
        [
            "id",
            "formula",
            "reduced_formula",
            "target_chemical_system",
            "source_structure_seen",
            "source_formula_seen",
            "source_chemical_system_seen",
        ]
    ].sort_values("id")
    metrics = build_subset_metrics(frame).sort_values(["subset", "method"]).reset_index(drop=True)
    deltas = build_delta_table(metrics).sort_values("subset").reset_index(drop=True)

    flags_path = args.out_dir / "test_overlap_flags.csv"
    metrics_path = args.out_dir / "subset_metrics.csv"
    delta_path = args.out_dir / "subset_mae_deltas.csv"
    report_path = args.out_dir / "README.md"
    manifest_path = args.out_dir / "manifest.json"

    flags.to_csv(flags_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    deltas.to_csv(delta_path, index=False)
    build_report(out_path=report_path, flags=flags, metrics=metrics, deltas=deltas)
    manifest = {
        "generated": CURRENT_DATE,
        "run_dir": str(args.run_dir.resolve()),
        "overlap_dir": str(args.overlap_dir.resolve()),
        "source_json": str(args.source_json.resolve()),
        "artifacts": {
            "test_overlap_flags": str(flags_path.resolve()),
            "subset_metrics": str(metrics_path.resolve()),
            "subset_mae_deltas": str(delta_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
