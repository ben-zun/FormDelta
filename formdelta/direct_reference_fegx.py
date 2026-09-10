from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from formdelta.data import build_feature_frame, normalize_split, read_split_records
from formdelta.reference import regression_metrics


PASSTHROUGH_COLS = [
    "formula",
    "natoms",
    "nelements",
    "official_fold_split",
    "validation_bucket",
    "chemistry_group",
    "elements",
    "is_positive",
    "is_near_zero_50",
    "is_near_zero_100",
    "quaternary_plus",
]
CURRENT_DATE = "2026-09-04"


def pick_column(frame: pd.DataFrame, explicit: str | None, candidates: tuple[str, ...], label: str) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise KeyError(f"{label} column {explicit!r} not found. Available: {list(frame.columns)}")
        return explicit
    for col in candidates:
        if col in frame.columns:
            return col
    raise KeyError(f"Could not infer {label} column. Candidates={candidates}; available={list(frame.columns)}")


def load_prediction_frames(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["prediction_file"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        raise ValueError("At least one prediction CSV is required.")
    return pd.concat(frames, ignore_index=True)


def load_split_payload(dataset_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        path = dataset_root / split / f"{split}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        frame["split"] = split
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No split CSVs found below {dataset_root}")
    out = pd.concat(frames, ignore_index=True)
    out["id"] = out["id"].astype(str)
    return out


def write_identity_coefficients(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for z in range(1, 119):
        rows.append({"calibration": "element_reference", "coefficient": f"mu_Z{z}", "value": 0.0})
    rows.append({"calibration": "element_reference", "coefficient": "bias", "value": 0.0})
    rows.append({"calibration": "affine_energy_composition", "coefficient": "alpha_energy", "value": 1.0})
    for z in range(1, 119):
        rows.append({"calibration": "affine_energy_composition", "coefficient": f"beta_Z{z}", "value": 0.0})
    rows.append({"calibration": "affine_energy_composition", "coefficient": "bias", "value": 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def _safe_float_mean(values: np.ndarray) -> float:
    mean = float(np.nanmean(values)) if values.size else float("nan")
    return mean if np.isfinite(mean) else 0.0


def build_direct_reference_contract_frame(
    *,
    prediction_csvs: list[Path],
    dataset_root: Path,
    workspace_root: Path,
    feature_cache: Path,
    id_column: str | None = None,
    split_column: str | None = None,
    target_column: str | None = None,
    energy_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pred = load_prediction_frames(prediction_csvs)
    id_col = pick_column(pred, id_column, ("id", "jid", "material_id"), "id")
    split_col = pick_column(pred, split_column, ("split", "source_split"), "split")
    target_col_in = pick_column(
        pred,
        target_column,
        ("target_per_atom_ev", "y_true", "formation_energy_per_atom_ev", "formation_energy_peratom"),
        "target",
    )
    energy_col = pick_column(
        pred,
        energy_column,
        (
            "energy_per_atom",
            "model_energy_per_atom",
            "model_energy_per_atom_ev",
            "matris_energy_per_atom_ev",
            "eqv3_energy_per_atom_ev",
            "y_model",
            "y_pred",
        ),
        "model energy",
    )

    pred = pred.rename(
        columns={
            id_col: "id",
            split_col: "split_raw",
            target_col_in: "y_true_prediction",
            energy_col: "model_energy_per_atom_unfilled",
        }
    )
    pred["id"] = pred["id"].astype(str)
    pred["prediction_split"] = pred["split_raw"].map(normalize_split)
    pred["y_true_prediction"] = pd.to_numeric(pred["y_true_prediction"], errors="coerce")
    pred["model_energy_per_atom_unfilled"] = pd.to_numeric(pred["model_energy_per_atom_unfilled"], errors="coerce")
    pred["_has_energy"] = pred["model_energy_per_atom_unfilled"].notna().astype(int)
    pred["_has_target"] = pred["y_true_prediction"].notna().astype(int)
    pred = pred.sort_values(["_has_energy", "_has_target"], ascending=[False, False]).copy()
    before = len(pred)
    pred = pred.drop_duplicates(subset=["id"], keep="first").copy()
    duplicates_dropped = before - len(pred)

    split_records = read_split_records(dataset_root=dataset_root, workspace_root=workspace_root)
    if split_records[["split", "id"]].duplicated().any():
        raise RuntimeError("Dataset split records contain duplicate (split, id) keys; cannot align raw predictions safely.")
    if split_records["id"].duplicated().any():
        duplicate_ids = (
            split_records.loc[split_records["id"].duplicated(keep=False), "id"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        preview = ", ".join(duplicate_ids[:10])
        raise RuntimeError(
            "Dataset split records reuse ids across splits, so direct-reference alignment by id would be ambiguous. "
            f"Example duplicate ids: {preview}"
        )
    split_payload = load_split_payload(dataset_root)
    merged = split_records[["split", "id", "cif_path", "target_from_dataset"]].merge(
        pred[
            [
                "id",
                "prediction_split",
                "y_true_prediction",
                "model_energy_per_atom_unfilled",
                "prediction_file",
            ]
        ],
        on=["id"],
        how="left",
    )
    merged["prediction_row_present"] = merged["prediction_file"].notna().astype(np.float32)
    merged["prediction_split_match"] = (
        merged["prediction_split"].fillna("").astype(str) == merged["split"].fillna("").astype(str)
    ).astype(np.float32)
    merged["target_from_dataset"] = pd.to_numeric(merged["target_from_dataset"], errors="coerce")
    merged["y_true_prediction"] = pd.to_numeric(merged["y_true_prediction"], errors="coerce")
    merged["y_true"] = merged["y_true_prediction"].where(np.isfinite(merged["y_true_prediction"]), merged["target_from_dataset"])
    if merged["y_true"].isna().any():
        raise RuntimeError("Some dataset rows remain without target labels after direct-reference alignment.")

    available_passthrough = [col for col in PASSTHROUGH_COLS if col in split_payload.columns]
    if available_passthrough:
        merged = merged.merge(
            split_payload[["split", "id", *available_passthrough]],
            on=["split", "id"],
            how="left",
        )

    merged["target_delta_vs_dataset"] = merged["y_true_prediction"] - merged["target_from_dataset"]
    merged["primary_missing"] = (~np.isfinite(merged["model_energy_per_atom_unfilled"])).astype(np.float32)
    merged["primary_available"] = 1.0 - merged["primary_missing"]
    merged = build_feature_frame(merged, cache_path=feature_cache)

    available_train_mask = (merged["split"].to_numpy() == "train") & (merged["primary_missing"].to_numpy() < 0.5)
    if not bool(np.any(available_train_mask)):
        raise RuntimeError("No train rows with finite frozen-model predictions are available for direct-reference FE-GX.")
    fill_energy = _safe_float_mean(merged.loc[available_train_mask, "model_energy_per_atom_unfilled"].to_numpy(dtype=np.float64))

    merged["model_energy_per_atom"] = pd.to_numeric(merged["model_energy_per_atom_unfilled"], errors="coerce").fillna(fill_energy)
    merged["pred_raw_model_energy_unfilled"] = merged["model_energy_per_atom_unfilled"]
    merged["pred_raw_model_energy"] = pd.to_numeric(merged["pred_raw_model_energy_unfilled"], errors="coerce").fillna(fill_energy)
    merged["pred_element_reference_unfilled"] = merged["model_energy_per_atom_unfilled"]
    merged["pred_element_reference"] = merged["model_energy_per_atom"].astype(float)
    merged["pred_affine_energy_composition_unfilled"] = merged["model_energy_per_atom_unfilled"]
    merged["pred_affine_energy_composition"] = merged["model_energy_per_atom"].astype(float)

    metrics: dict[str, Any] = {
        "generated": CURRENT_DATE,
        "input": {
            "prediction_csv": [str(path.resolve()) for path in prediction_csvs],
            "dataset_root": str(dataset_root.resolve()),
            "feature_cache": str(feature_cache.resolve()),
            "n_rows": int(len(merged)),
            "missing_prediction_rows": int(merged["prediction_file"].isna().sum()),
            "prediction_split_mismatch_rows": int(
                np.sum((merged["prediction_row_present"].to_numpy() > 0.5) & (merged["prediction_split_match"].to_numpy() < 0.5))
            ),
            "primary_available_rows": int(np.sum(merged["primary_missing"].to_numpy() < 0.5)),
            "primary_missing_rows": int(np.sum(merged["primary_missing"].to_numpy() >= 0.5)),
            "duplicates_dropped": int(duplicates_dropped),
            "fill_values": {
                "model_energy_per_atom": float(fill_energy),
                "pred_raw_model_energy": float(fill_energy),
                "pred_element_reference": float(fill_energy),
                "pred_affine_energy_composition": float(fill_energy),
            },
        },
        "raw_model_energy_as_formation": {},
        "filled_direct_reference": {},
    }
    for split, group in merged.groupby("split"):
        available = group.loc[group["primary_missing"] < 0.5].copy()
        metrics["raw_model_energy_as_formation"][str(split)] = regression_metrics(
            available["y_true"].to_numpy(dtype=np.float64),
            available["pred_raw_model_energy_unfilled"].to_numpy(dtype=np.float64),
        )
        metrics["filled_direct_reference"][str(split)] = regression_metrics(
            group["y_true"].to_numpy(dtype=np.float64),
            group["pred_element_reference"].to_numpy(dtype=np.float64),
        )

    output_cols = [
        "split",
        "id",
        "formula",
        "cif_path",
        "natoms",
        "nelements",
        "y_true",
        "y_true_prediction",
        "target_from_dataset",
        "prediction_row_present",
        "prediction_split_match",
        "primary_missing",
        "primary_available",
        "model_energy_per_atom",
        "model_energy_per_atom_unfilled",
        "pred_raw_model_energy",
        "pred_raw_model_energy_unfilled",
        "pred_element_reference",
        "pred_element_reference_unfilled",
        "pred_affine_energy_composition",
        "pred_affine_energy_composition_unfilled",
        "prediction_file",
        "prediction_split",
    ]
    output_cols.extend([col for col in available_passthrough if col not in output_cols])
    feature_cols = [f"x_Z{z}" for z in range(1, 119)]
    output_cols.extend([col for col in feature_cols if col in merged.columns and col not in output_cols])
    contract = merged[output_cols].copy()
    return contract, metrics


def write_direct_reference_contract(
    *,
    prediction_csvs: list[Path],
    dataset_root: Path,
    workspace_root: Path,
    feature_cache: Path,
    out_dir: Path,
    id_column: str | None = None,
    split_column: str | None = None,
    target_column: str | None = None,
    energy_column: str | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    contract, metrics = build_direct_reference_contract_frame(
        prediction_csvs=prediction_csvs,
        dataset_root=dataset_root,
        workspace_root=workspace_root,
        feature_cache=feature_cache,
        id_column=id_column,
        split_column=split_column,
        target_column=target_column,
        energy_column=energy_column,
    )
    contract_path = out_dir / "calibrated_predictions.csv"
    coeff_path = out_dir / "calibration_coefficients.csv"
    contract.to_csv(contract_path, index=False)
    write_identity_coefficients(coeff_path)
    metrics["contract"] = {
        "type": "direct_reference",
        "calibrated_predictions": str(contract_path.resolve()),
        "coefficient_csv": str(coeff_path.resolve()),
    }
    (out_dir / "contract_summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics
