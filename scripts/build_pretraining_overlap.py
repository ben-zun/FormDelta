#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from pymatgen.core import Composition, Structure

try:
    from pymatgen.core.structure_matcher import StructureMatcher
except Exception:
    from pymatgen.analysis.structure_matcher import StructureMatcher


ROOT = Path(os.environ.get("FORMDELTA_ROOT", Path(__file__).resolve().parents[1])).resolve()
RUNS = ROOT / "outputs"
CURRENT_DATE = "2026-09-04"

DEFAULT_TARGET_ROOT = ROOT / "data" / "medium_same_split_269"
DEFAULT_MEGNET_SOURCE_JSON = ROOT / "external_data" / "formation_energy" / "mp.2018.6.1.json"
DEFAULT_MEGNET_CHECKPOINT = ROOT / "models" / "megnet_formation_energy.hdf5"
DEFAULT_EQV3_CHECKPOINT = ROOT / "models" / "eqv3" / "omat24-mptrj-salex_gradient.pt"
DEFAULT_OUT_DIR = RUNS / "checkpoint_target_overlap_audit_20260904"

MATCHER = StructureMatcher(
    ltol=0.2,
    stol=0.3,
    angle_tol=5.0,
    primitive_cell=True,
    scale=True,
    attempt_supercell=False,
    allow_subset=False,
)
MATCHER_CONFIG = {
    "ltol": 0.2,
    "stol": 0.3,
    "angle_tol": 5.0,
    "primitive_cell": True,
    "scale": True,
    "attempt_supercell": False,
    "allow_subset": False,
}


@dataclass
class TargetEntry:
    split: str
    sample_id: str
    formula: str
    reduced_formula: str
    chemical_system: str
    natoms: int
    cif_path: Path
    structure: Structure


@dataclass
class OverlapHit:
    model: str
    split_scope: str
    target_id: str
    target_formula: str
    target_reduced_formula: str
    target_chemical_system: str
    target_natoms: int
    source_id: str
    source_formula: str
    source_reduced_formula: str
    source_chemical_system: str
    source_natoms: int
    overlap_kind: str
    source_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit checkpoint-target overlap for current FE-GX experiments.")
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--megnet-source-json", type=Path, default=DEFAULT_MEGNET_SOURCE_JSON)
    parser.add_argument("--megnet-checkpoint", type=Path, default=DEFAULT_MEGNET_CHECKPOINT)
    parser.add_argument("--eqv3-checkpoint", type=Path, default=DEFAULT_EQV3_CHECKPOINT)
    parser.add_argument("--skip-eqv3", action="store_true", help="Do not load EqV3 checkpoint provenance.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_json_array(path: Path) -> Iterable[Any]:
    try:
        import ijson  # type: ignore
    except Exception:
        ijson = None
    if ijson is not None:
        with path.open("rb") as handle:
            yield from ijson.items(handle, "item")
        return
    payload = read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list")
    yield from payload


def safe_reduced_formula(formula: str) -> str:
    try:
        return str(Composition(formula).reduced_formula)
    except Exception:
        return str(formula)


def safe_chemsys(formula: str) -> str:
    try:
        elements = sorted(str(el) for el in Composition(formula).elements)
    except Exception:
        return ""
    return "-".join(elements)


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Issues encountered while parsing CIF: .*fractional coordinates rounded to ideal values.*",
        category=UserWarning,
    )


def load_target_entries(target_root: Path) -> list[TargetEntry]:
    entries: list[TargetEntry] = []
    loaded = 0
    for split in ("train", "val", "test"):
        csv_path = target_root / split / f"{split}.csv"
        frame = pd.read_csv(csv_path)
        for row in frame.itertuples(index=False):
            cif_path = Path(str(row.cif_path))
            if not cif_path.is_absolute():
                cif_path = ROOT / cif_path
            cif_path = cif_path.resolve()
            formula = str(row.formula)
            entries.append(
                TargetEntry(
                    split=split,
                    sample_id=str(row.id),
                    formula=formula,
                    reduced_formula=safe_reduced_formula(formula),
                    chemical_system=safe_chemsys(formula),
                    natoms=safe_int(row.natoms),
                    cif_path=cif_path,
                    structure=Structure.from_file(cif_path),
                )
            )
            loaded += 1
            if loaded % 1000 == 0:
                print(f"[target] loaded {loaded} structures", flush=True)
    return entries


def split_scopes(entries: list[TargetEntry]) -> dict[str, list[TargetEntry]]:
    scopes: dict[str, list[TargetEntry]] = {"all": entries}
    for split in ("train", "val", "test"):
        scopes[split] = [entry for entry in entries if entry.split == split]
    return scopes


def overlap_label(
    *,
    pretraining_datasets: list[str],
    auditable: bool,
    structure_overlap_samples: int | None,
    composition_overlap_samples: int | None,
    chemical_system_overlap_samples: int | None,
) -> str:
    datasets_lower = {item.lower() for item in pretraining_datasets}
    if not auditable:
        return "cross-dataset adaptation"
    if auditable and structure_overlap_samples == 0 and composition_overlap_samples == 0 and chemical_system_overlap_samples == 0:
        return "strict external test"
    if auditable and structure_overlap_samples == 0 and composition_overlap_samples == 0:
        return "composition OOD"
    if auditable and structure_overlap_samples == 0:
        return "structure OOD"
    if {"materials project", "mp", "mp2018", "mptrj"} & datasets_lower:
        return "in-domain adaptation"
    return "cross-dataset adaptation"


def scope_rows_from_hits(
    *,
    model: str,
    target_root: Path,
    target_scopes: dict[str, list[TargetEntry]],
    pretraining_datasets: list[str],
    training_target: str,
    checkpoint_path: Path,
    auditable: bool,
    exact_id_overlap_ids: dict[str, set[str]],
    composition_overlap_ids: dict[str, set[str]],
    chemical_system_overlap_ids: dict[str, set[str]],
    structure_overlap_ids: dict[str, set[str]],
    notes: str,
    source_paths: list[str],
    source_paths_present: list[bool],
    source_rows: int | None,
    source_namespace_note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, entries in target_scopes.items():
        target_ids = {entry.sample_id for entry in entries}
        target_formulas = {entry.reduced_formula for entry in entries}
        target_chemsys = {entry.chemical_system for entry in entries}
        exact_ids = target_ids & exact_id_overlap_ids.get(scope, set())
        comp_ids = target_ids & composition_overlap_ids.get(scope, set())
        chem_ids = target_ids & chemical_system_overlap_ids.get(scope, set())
        struct_ids = target_ids & structure_overlap_ids.get(scope, set())
        struct_count = len(struct_ids) if auditable else None
        comp_count = len(comp_ids) if auditable else None
        chem_count = len(chem_ids) if auditable else None
        label = overlap_label(
            pretraining_datasets=pretraining_datasets,
            auditable=auditable,
            structure_overlap_samples=struct_count,
            composition_overlap_samples=comp_count,
            chemical_system_overlap_samples=chem_count,
        )
        rows.append(
            {
                "generated_date": CURRENT_DATE,
                "model": model,
                "split_scope": scope,
                "exact_checkpoint": str(checkpoint_path.resolve()),
                "pretraining_datasets": "; ".join(pretraining_datasets),
                "training_target": training_target,
                "current_target_dataset": str(target_root.resolve()),
                "target_rows": len(entries),
                "target_unique_ids": len(target_ids),
                "target_unique_reduced_formulas": len(target_formulas),
                "target_unique_chemical_systems": len(target_chemsys),
                "source_rows": source_rows,
                "source_paths": " | ".join(source_paths),
                "source_paths_present": " | ".join("yes" if flag else "no" for flag in source_paths_present),
                "exact_id_overlap": len(exact_ids) if auditable else None,
                "structure_overlap": struct_count,
                "composition_overlap": comp_count,
                "chemical_system_overlap": chem_count,
                "label": label,
                "auditable_from_local_assets": auditable,
                "source_namespace_note": source_namespace_note,
                "notes": notes,
            }
        )
    return rows


def audit_megnet(
    *,
    target_root: Path,
    target_entries: list[TargetEntry],
    source_json: Path,
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scopes = split_scopes(target_entries)
    scope_target_ids = {scope: {entry.sample_id for entry in rows} for scope, rows in scopes.items()}
    scope_target_formula_groups = {
        scope: group_by_formula(rows)
        for scope, rows in scopes.items()
    }
    scope_target_chemsys_groups = {
        scope: group_by_chemsys(rows)
        for scope, rows in scopes.items()
    }

    exact_id_hits = {scope: set() for scope in scopes}
    composition_hits = {scope: set() for scope in scopes}
    chemsys_hits = {scope: set() for scope in scopes}
    structure_hits = {scope: set() for scope in scopes}
    detailed_hits: list[dict[str, Any]] = []
    source_rows = 0

    for record in iter_json_array(source_json):
        if not isinstance(record, dict):
            continue
        source_rows += 1
        if source_rows % 5000 == 0:
            print(
                f"[megnet_source] scanned {source_rows} rows | "
                f"structure_hits={len(structure_hits['all'])} composition_hits={len(composition_hits['all'])}",
                flush=True,
            )
        source_id = str(record.get("material_id", ""))
        structure_text = record.get("structure")
        if not isinstance(structure_text, str) or not structure_text.strip():
            continue
        try:
            source_structure = Structure.from_str(structure_text, fmt="cif")
        except Exception:
            continue
        source_formula = source_structure.composition.formula
        source_reduced_formula = str(source_structure.composition.reduced_formula)
        source_chemsys = "-".join(sorted(str(el) for el in source_structure.composition.elements))
        source_natoms = int(source_structure.num_sites)

        for scope, rows in scopes.items():
            if source_id in scope_target_ids[scope]:
                exact_id_hits[scope].add(source_id)

        candidate_targets = scope_target_formula_groups["all"].get(source_reduced_formula, [])
        for entry in candidate_targets:
            composition_hits["all"].add(entry.sample_id)
            chemsys_hits["all"].add(entry.sample_id)
            if entry.split in composition_hits:
                composition_hits[entry.split].add(entry.sample_id)
        chemsys_targets = scope_target_chemsys_groups["all"].get(source_chemsys, [])
        for entry in chemsys_targets:
            chemsys_hits["all"].add(entry.sample_id)
            if entry.split in chemsys_hits:
                chemsys_hits[entry.split].add(entry.sample_id)

        if not candidate_targets:
            continue

        remaining_candidates = [entry for entry in candidate_targets if entry.sample_id not in structure_hits["all"]]
        for entry in remaining_candidates:
            if not MATCHER.fit(source_structure, entry.structure):
                continue
            structure_hits["all"].add(entry.sample_id)
            structure_hits[entry.split].add(entry.sample_id)
            hit = OverlapHit(
                model="MEGNet",
                split_scope=entry.split,
                target_id=entry.sample_id,
                target_formula=entry.formula,
                target_reduced_formula=entry.reduced_formula,
                target_chemical_system=entry.chemical_system,
                target_natoms=entry.natoms,
                source_id=source_id,
                source_formula=source_formula,
                source_reduced_formula=source_reduced_formula,
                source_chemical_system=source_chemsys,
                source_natoms=source_natoms,
                overlap_kind="structure_matcher",
                source_note="source JSON material_id namespace differs from matbench target IDs",
            )
            detailed_hits.append(hit.__dict__)

    summary_rows = scope_rows_from_hits(
        model="MEGNet",
        target_root=target_root,
        target_scopes=scopes,
        pretraining_datasets=["MP2018"],
        training_target="formation_energy_per_atom",
        checkpoint_path=checkpoint_path,
        auditable=True,
        exact_id_overlap_ids=exact_id_hits,
        composition_overlap_ids=composition_hits,
        chemical_system_overlap_ids=chemsys_hits,
        structure_overlap_ids=structure_hits,
        notes="Source asset is local mp.2018.6.1.json with formation_energy_per_atom labels. Exact-ID overlap is namespace-limited because source IDs are mvc-* while target IDs are mb-mp-e-form-*.",
        source_paths=[str(source_json.resolve())],
        source_paths_present=[source_json.exists()],
        source_rows=source_rows,
        source_namespace_note="source IDs are mvc-*; target IDs are mb-mp-e-form-*; exact-ID overlap is therefore only a namespace check",
    )
    manifest = {
        "model": "MEGNet",
        "checkpoint": str(checkpoint_path.resolve()),
        "source_json": str(source_json.resolve()),
        "source_rows": source_rows,
        "matcher": matcher_manifest(),
    }
    return summary_rows, detailed_hits, manifest


def matcher_manifest() -> dict[str, Any]:
    return dict(MATCHER_CONFIG)


def group_by_formula(entries: Iterable[TargetEntry]) -> dict[str, list[TargetEntry]]:
    grouped: dict[str, list[TargetEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.reduced_formula, []).append(entry)
    return grouped


def group_by_chemsys(entries: Iterable[TargetEntry]) -> dict[str, list[TargetEntry]]:
    grouped: dict[str, list[TargetEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.chemical_system, []).append(entry)
    return grouped


def eqv3_provenance(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    dataset = config.get("dataset", {}) if isinstance(config, dict) else {}
    val_dataset = config.get("val_dataset", {}) if isinstance(config, dict) else {}
    src_paths = list(dataset.get("src", [])) if isinstance(dataset.get("src", []), list) else []
    val_paths = list(val_dataset.get("src", [])) if isinstance(val_dataset.get("src", []), list) else []
    all_paths = [str(item) for item in src_paths + val_paths]
    unique_paths: list[str] = []
    seen: set[str] = set()
    for item in all_paths:
        if item not in seen:
            seen.add(item)
            unique_paths.append(item)
    return {
        "config": config,
        "source_paths": unique_paths,
        "source_paths_present": [Path(path).exists() for path in unique_paths],
    }


def audit_eqv3(
    *,
    target_root: Path,
    target_entries: list[TargetEntry],
    checkpoint_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scopes = split_scopes(target_entries)
    provenance = eqv3_provenance(checkpoint_path)
    source_paths = provenance["source_paths"]
    source_present = provenance["source_paths_present"]
    auditable = bool(source_paths) and all(source_present)
    notes = (
        "Checkpoint provenance shows OMat24 -> MPtrj + sAlex gradient fine-tuning with uncorrected total-energy labels. "
        "The source dataset paths embedded in the checkpoint are not available locally, so exact ID / structure overlap is not directly countable from local assets."
    )
    summary_rows = scope_rows_from_hits(
        model="EqV3",
        target_root=target_root,
        target_scopes=scopes,
        pretraining_datasets=["OMat24", "MPtrj", "sAlex"],
        training_target="uncorrected total energy (+forces,+stress), not formation energy",
        checkpoint_path=checkpoint_path,
        auditable=auditable,
        exact_id_overlap_ids={scope: set() for scope in scopes},
        composition_overlap_ids={scope: set() for scope in scopes},
        chemical_system_overlap_ids={scope: set() for scope in scopes},
        structure_overlap_ids={scope: set() for scope in scopes},
        notes=notes,
        source_paths=source_paths,
        source_paths_present=source_present,
        source_rows=None,
        source_namespace_note="source IDs not available locally because checkpoint references missing ASE DB paths",
    )
    manifest = {
        "model": "EqV3",
        "checkpoint": str(checkpoint_path.resolve()),
        "matcher": matcher_manifest(),
        "source_paths": source_paths,
        "source_paths_present": source_present,
        "dataset_config_excerpt": {
            "dataset": provenance["config"].get("dataset", {}),
            "val_dataset": provenance["config"].get("val_dataset", {}),
        },
    }
    return summary_rows, [], manifest


def write_markdown(
    path: Path,
    *,
    target_root: Path,
    target_entries: list[TargetEntry],
    summary_rows: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> None:
    target_counts = {split: sum(1 for entry in target_entries if entry.split == split) for split in ("train", "val", "test")}
    lines = [
        "# Checkpoint-Target Overlap Audit",
        "",
        f"Generated: {CURRENT_DATE}",
        "",
        f"Target dataset: `{target_root.resolve()}`",
        "",
        f"Target rows: train `{target_counts['train']}`, val `{target_counts['val']}`, test `{target_counts['test']}`.",
        "",
        "Fixed StructureMatcher parameters:",
        "",
        f"- ltol = `{MATCHER_CONFIG['ltol']}`",
        f"- stol = `{MATCHER_CONFIG['stol']}`",
        f"- angle_tol = `{MATCHER_CONFIG['angle_tol']}`",
        f"- primitive_cell = `{MATCHER_CONFIG['primitive_cell']}`",
        f"- scale = `{MATCHER_CONFIG['scale']}`",
        f"- attempt_supercell = `{MATCHER_CONFIG['attempt_supercell']}`",
        f"- allow_subset = `{MATCHER_CONFIG['allow_subset']}`",
        "",
        "## Supplementary Table",
        "",
        "| model | scope | pretraining datasets | training target | exact-ID | structure | composition | chemical-system | label | local source audit |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        exact_id = "" if pd.isna(row["exact_id_overlap"]) else int(row["exact_id_overlap"])
        struct = "" if pd.isna(row["structure_overlap"]) else int(row["structure_overlap"])
        comp = "" if pd.isna(row["composition_overlap"]) else int(row["composition_overlap"])
        chem = "" if pd.isna(row["chemical_system_overlap"]) else int(row["chemical_system_overlap"])
        audit_flag = "direct-counted" if bool(row["auditable_from_local_assets"]) else "provenance-only"
        lines.append(
            f"| {row['model']} | {row['split_scope']} | {row['pretraining_datasets']} | {row['training_target']} | {exact_id} | {struct} | {comp} | {chem} | {row['label']} | {audit_flag} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MEGNet uses a local MP2018 formation-energy JSON source and is directly auditable against the current target split.",
            "- EqV3 provenance is recoverable from the checkpoint config, but the embedded MPtrj and sAlex source paths are missing locally on 2026-09-04, so direct overlap counts are intentionally left blank rather than guessed.",
            "",
            "## Source Manifests",
            "",
        ]
    )
    for manifest in manifests:
        lines.append(f"- `{manifest['model']}` checkpoint: `{manifest['checkpoint']}`")
        if manifest["model"] == "EqV3":
            lines.append(f"  - source paths present: `{manifest['source_paths_present']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_warnings()

    target_root = args.target_root.resolve()
    target_entries = load_target_entries(target_root)
    target_manifest = {
        "target_root": str(target_root),
        "rows": len(target_entries),
        "split_counts": {
            "train": sum(1 for entry in target_entries if entry.split == "train"),
            "val": sum(1 for entry in target_entries if entry.split == "val"),
            "test": sum(1 for entry in target_entries if entry.split == "test"),
        },
        "matcher": matcher_manifest(),
    }

    megnet_summary, megnet_hits, megnet_manifest = audit_megnet(
        target_root=target_root,
        target_entries=target_entries,
        source_json=args.megnet_source_json.resolve(),
        checkpoint_path=args.megnet_checkpoint.resolve(),
    )
    summary_rows = list(megnet_summary)
    detailed_hits = list(megnet_hits)
    manifests = [megnet_manifest]
    if not args.skip_eqv3 and args.eqv3_checkpoint.exists():
        eqv3_summary, eqv3_hits, eqv3_manifest = audit_eqv3(
            target_root=target_root,
            target_entries=target_entries,
            checkpoint_path=args.eqv3_checkpoint.resolve(),
        )
        summary_rows.extend(eqv3_summary)
        detailed_hits.extend(eqv3_hits)
        manifests.append(eqv3_manifest)
    elif not args.skip_eqv3:
        print(
            f"[checkpoint_overlap] EqV3 checkpoint not found; skipping EqV3 provenance: {args.eqv3_checkpoint}",
            flush=True,
        )
    write_csv(out_dir / "checkpoint_overlap_audit.csv", summary_rows)
    write_csv(out_dir / "checkpoint_overlap_hits.csv", detailed_hits)
    write_markdown(
        out_dir / "checkpoint_overlap_supplementary.md",
        target_root=target_root,
        target_entries=target_entries,
        summary_rows=summary_rows,
        manifests=manifests,
    )
    (out_dir / "checkpoint_overlap_manifest.json").write_text(
        json.dumps(
            {
                "generated_date": CURRENT_DATE,
                "target_manifest": target_manifest,
                "source_manifests": manifests,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_csv": str((out_dir / "checkpoint_overlap_audit.csv").resolve()),
                "hits_csv": str((out_dir / "checkpoint_overlap_hits.csv").resolve()),
                "supplementary_md": str((out_dir / "checkpoint_overlap_supplementary.md").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
