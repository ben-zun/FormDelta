from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class FeatureRule:
    feature_name: str
    feature_group: str
    source: str
    allowed_in_forward: bool
    allowed_in_training_sampler: bool
    allowed_in_loss: bool
    allowed_in_validation_slice: bool
    requires_y_true: bool
    requires_test_label: bool
    requires_external_label: bool
    deployability_class: str
    notes: str


FEATURE_RULES: dict[str, FeatureRule] = {
    "composition": FeatureRule("composition", "base", "CIF", True, True, False, True, False, False, False, "A", "CIF-derived composition vector."),
    "structure_metadata": FeatureRule("structure_metadata", "base", "CIF", True, True, False, True, False, False, False, "A", "CIF-derived lattice/symmetry/neighborhood metadata."),
    "element_env": FeatureRule("element_env", "base", "CIF", True, True, False, True, False, False, False, "A", "CIF-derived element environment features."),
    "pair_env": FeatureRule("pair_env", "base", "CIF", True, True, False, True, False, False, False, "A", "CIF-derived pair environment features."),
    "chem_pair_env": FeatureRule("chem_pair_env", "base", "CIF + element properties", True, True, False, True, False, False, False, "A", "CIF-derived chemistry-normalized pair environment features."),
    "matris_energy": FeatureRule("matris_energy", "foundation", "MatRIS forward", True, True, False, True, False, False, False, "A", "Foundation model energy available at inference."),
    "eqv3_energy": FeatureRule("eqv3_energy", "foundation", "EqV3 forward", True, True, False, True, False, False, False, "A", "Foundation model energy available at inference; may be missing with aux_missing flag."),
    "foundation_disagreement": FeatureRule("foundation_disagreement", "tail", "MatRIS/EqV3 forward", True, True, True, True, False, False, False, "A", "Absolute MatRIS/EqV3 energy disagreement; deployable if both foundation energies are available."),
    "pair_frequency_score": FeatureRule("pair_frequency_score", "tail", "train-only pair counts", True, True, True, True, False, False, False, "A", "Train-only pair-frequency feature. Must not use held-out/test counts."),
    "rare_pair_flag": FeatureRule("rare_pair_flag", "tail", "train-only pair counts", True, True, True, True, False, False, False, "A", "Train-only rare-pair flag. Must not use held-out/test counts."),
    "pair_geometry_anomaly_score": FeatureRule("pair_geometry_anomaly_score", "tail", "CIF geometry", True, True, True, True, False, False, False, "A", "Inference-available pair geometry anomaly summary."),
    "chem_pair_anomaly_score": FeatureRule("chem_pair_anomaly_score", "tail", "CIF geometry + element properties", True, True, True, True, False, False, False, "A", "Inference-available chemistry-normalized geometry anomaly summary."),
    "bond_geometry_zscore": FeatureRule("bond_geometry_zscore", "tail", "CIF geometry", True, True, True, True, False, False, False, "A", "Inference-available normalized bond geometry score."),
    "cov_radius_normalized_bond_zscore": FeatureRule("cov_radius_normalized_bond_zscore", "tail", "CIF geometry + covalent radii", True, True, True, True, False, False, False, "A", "Inference-available covalent-radius normalized bond score."),
    "reference_residual_risk": FeatureRule("reference_residual_risk", "tail", "label/reference residual", False, True, True, True, True, True, True, "C", "Allowed for sampler/loss/slices only. Not a forward feature unless predicted by a separate deployable model."),
    "reference_residual_top_flag": FeatureRule("reference_residual_top_flag", "tail", "label/reference residual", False, True, True, True, True, True, True, "C", "Allowed for sampler/loss/slices only. Not a forward feature."),
    "positive_likelihood": FeatureRule("positive_likelihood", "tail", "label-derived unless auxiliary predictor", False, True, True, True, True, True, True, "B", "Forward-allowed only if produced by a train-only auxiliary classifier. Current robust caches use label-derived values."),
    "test_error_bucket": FeatureRule("test_error_bucket", "diagnostic", "test error", False, False, False, True, True, True, True, "D", "Never allowed in training or forward."),
}


def manifest_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(rule) for rule in FEATURE_RULES.values()])


def rule_for(feature_name: str) -> FeatureRule:
    return FEATURE_RULES.get(
        feature_name,
        FeatureRule(
            feature_name=feature_name,
            feature_group="unknown",
            source="unknown",
            allowed_in_forward=False,
            allowed_in_training_sampler=False,
            allowed_in_loss=False,
            allowed_in_validation_slice=True,
            requires_y_true=True,
            requires_test_label=True,
            requires_external_label=True,
            deployability_class="D",
            notes="Unknown feature defaults to non-deployable until added to the feature contract.",
        ),
    )


def audit_forward_features(feature_names: Iterable[str]) -> pd.DataFrame:
    rows = []
    for name in feature_names:
        rule = rule_for(str(name))
        row = asdict(rule)
        row["feature_name"] = str(name)
        row["forward_legal"] = bool(rule.allowed_in_forward and not rule.requires_external_label)
        rows.append(row)
    return pd.DataFrame(rows)


def illegal_forward_features(feature_names: Iterable[str]) -> list[str]:
    audit = audit_forward_features(feature_names)
    if audit.empty:
        return []
    return audit.loc[~audit["forward_legal"], "feature_name"].astype(str).tolist()


def checkpoint_tail_feature_cols(checkpoint: str | Path) -> list[str]:
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    cols = state.get("tail_feature_cols") or []
    return [str(col) for col in cols]


def checkpoint_deployability(checkpoint: str | Path) -> dict[str, object]:
    cols = checkpoint_tail_feature_cols(checkpoint)
    illegal = illegal_forward_features(cols)
    return {
        "checkpoint": str(Path(checkpoint)),
        "tail_feature_cols": ";".join(cols),
        "illegal_forward_features": ";".join(illegal),
        "deployable_feature_pass": len(illegal) == 0,
    }


def feature_values_for_online_forward(
    feature_names: list[str],
    *,
    features: dict[str, object],
    matris_energy: float,
    eqv3_energy: float | None,
) -> np.ndarray:
    illegal = illegal_forward_features(feature_names)
    if illegal:
        raise ValueError(f"Checkpoint requires non-deployable forward features: {illegal}")
    metadata = features.get("structure_metadata", {}) if isinstance(features.get("structure_metadata"), dict) else {}
    bond_mean = float(metadata.get("bond_dist_mean", 0.0) or 0.0)
    bond_std = float(metadata.get("bond_dist_std", 0.0) or 0.0)
    bond_min = float(metadata.get("bond_dist_min", 0.0) or 0.0)
    natoms = float(features.get("natoms", 0.0) or 0.0)
    pair_rows = features.get("pair_env_raw", [])
    if isinstance(pair_rows, list) and pair_rows:
        pair_fracs = [float(row.get("pair_frac", 0.0) or 0.0) for row in pair_rows if isinstance(row, dict)]
        rare_pair_flag = 1.0 if pair_fracs and max(pair_fracs) < 0.05 else 0.0
        pair_frequency_score = float(np.log1p(sum(pair_fracs) * max(natoms, 1.0)))
    else:
        rare_pair_flag = 0.0
        pair_frequency_score = 0.0
    foundation_disagreement = abs(float(matris_energy) - float(eqv3_energy)) if eqv3_energy is not None else 0.0
    geometry_z = (bond_mean - 2.5) / max(bond_std, 0.25)
    anomaly = abs(geometry_z) + max(0.0, 1.0 - bond_min)
    values = {
        "pair_frequency_score": pair_frequency_score,
        "rare_pair_flag": rare_pair_flag,
        "pair_geometry_anomaly_score": anomaly,
        "chem_pair_anomaly_score": anomaly,
        "foundation_disagreement": foundation_disagreement,
        "bond_geometry_zscore": geometry_z,
        "cov_radius_normalized_bond_zscore": geometry_z,
    }
    return np.asarray([values.get(name, 0.0) for name in feature_names], dtype=np.float32)
