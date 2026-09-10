#!/usr/bin/env python3
"""Materialize a paper-wide number registry from revision-v2 source packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from build_revision_v2_headline_registry import compare_with_existing, build_registry_rows, validate_registry
from run_paper_closure_331_341_consolidation import ensure_dir, write_csv
from validate_data_leakage import run_validation as run_data_leakage_validation
from validate_prediction_archives import run_validation as run_prediction_archive_validation
from validate_units_and_scaling import run_validation as run_unit_scaling_validation

ROOT = Path(os.environ.get("FORMDELTA_ROOT", Path(__file__).resolve().parents[1])).resolve()
PAPER = ROOT / "paper_revision_v2"
OUT = PAPER / "24_paper_number_pipeline"

HEADLINE = PAPER / "headline_result_registry.csv"
MATCHED = PAPER / "01_matched_parameterization" / "fullrun_5seed_r20260710" / "model_manifest.csv"
STRONG = PAPER / "02_strong_stacking_delta" / "family_best_summary.csv"
BOOTSTRAP = PAPER / "06_statistics" / "paired_bootstrap_multiple_comparison.csv"
NOOVERLAP = PAPER / "05_provider_exposure" / "alexandria_verified_nooverlap_pbesol" / "protocol_scope_metrics.csv"
HULL = PAPER / "10_convex_hull" / "formal_hull_corrected_rerun_metrics.csv"
RISK = PAPER / "14_selective_prediction" / "selective_prediction_curve_extension.csv"

TABLE_FIELDS = [
    "registry_group",
    "artifact_id",
    "row_id",
    "display_label",
    "metric_name",
    "metric_value",
    "units",
    "source_file",
    "source_row_key",
    "sample_level_provenance",
    "notes",
]


def add_metric(
    rows: list[dict[str, Any]],
    *,
    registry_group: str,
    artifact_id: str,
    row_id: str,
    display_label: str,
    metric_name: str,
    metric_value: Any,
    units: str,
    source_file: Path,
    source_row_key: str,
    sample_level_provenance: str,
    notes: str = "",
) -> None:
    rows.append(
        {
            "registry_group": registry_group,
            "artifact_id": artifact_id,
            "row_id": row_id,
            "display_label": display_label,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "units": units,
            "source_file": str(source_file),
            "source_row_key": source_row_key,
            "sample_level_provenance": sample_level_provenance,
            "notes": notes,
        }
    )


def build_rows() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    unit_scaling_report = run_unit_scaling_validation(fail_on_error=True)
    data_leakage_report = run_data_leakage_validation(fail_on_error=True)
    prediction_archive_report = run_prediction_archive_validation(fail_on_error=True)
    rows_by_group: dict[str, list[dict[str, Any]]] = {
        "abstract": [],
        "main_table": [],
        "supplement": [],
        "bootstrap": [],
        "hull": [],
        "risk_curve": [],
    }
    source_files = [HEADLINE, MATCHED, STRONG, BOOTSTRAP, NOOVERLAP, HULL, RISK]
    missing_sources = [str(path) for path in source_files if not path.exists()]
    if missing_sources:
        raise FileNotFoundError("Missing required sources: " + "; ".join(missing_sources))

    headline_rows = pd.read_csv(HEADLINE)
    headline_exp_lookup = {str(row["experiment_id"]): row for _, row in headline_rows.iterrows()}

    narrative = headline_rows.copy()
    narrative["narrative_chain_step"] = pd.to_numeric(narrative["narrative_chain_step"], errors="coerce")
    narrative = narrative[narrative["narrative_chain_step"].notna()].copy()
    narrative["narrative_chain_step"] = narrative["narrative_chain_step"].astype(int)
    narrative = narrative.sort_values("narrative_chain_step")
    for row in narrative.itertuples(index=False):
        add_metric(
            rows_by_group["abstract"],
            registry_group="abstract",
            artifact_id="headline_chain",
            row_id=str(row.experiment_id),
            display_label=f"headline step {row.narrative_chain_step}",
            metric_name="mae_mev_atom",
            metric_value=float(row.mae_mev_atom),
            units="meV/atom",
            source_file=HEADLINE,
            source_row_key=str(row.experiment_id),
            sample_level_provenance=str(row.prediction_archive_status),
            notes=str(row.what_changed_from_previous),
        )

    matched = pd.read_csv(MATCHED)
    main_variants = ["M1", "M2", "M3", "M4", "M5_noreg", "M9"]
    for variant in main_variants:
        row = matched[matched["variant"].eq(variant)].iloc[0]
        headline_row = headline_exp_lookup[f"MP_FOLD0_M1M9_{variant}"]
        for metric_name, value in [
            ("ensemble_test_mae_mev_atom", float(row["ensemble_test_mae_mev_atom"])),
            ("ensemble_test_p95_mev_atom", float(row["ensemble_test_p95_mev_atom"])),
        ]:
            add_metric(
                rows_by_group["main_table"],
                registry_group="main_table",
                artifact_id="matched_parameterization_core",
                row_id=variant,
                display_label=str(row["formula"]),
                metric_name=metric_name,
                metric_value=value,
                units="meV/atom",
                source_file=MATCHED,
                source_row_key=variant,
                sample_level_provenance=str(headline_row["prediction_archive_status"]),
                notes="Core M1-M9 table row.",
            )

    strong = pd.read_csv(STRONG)
    strong_families = [
        "learned_mixture_stacker",
        "oof_meta_stacker",
        "feature_augmented_direct_stacker",
        "residual_target_stacker",
        "gated_residual_stacker",
    ]
    for family in strong_families:
        row = strong[strong["family"].eq(family)].iloc[0]
        for metric_name, value in [
            ("best_mae_mev_atom", float(row["best_mae_mev_atom"])),
            ("best_p95_mev_atom", float(row["best_p95_mev_atom"])),
            ("delta_vs_fegx_anchor_mev_atom", float(row["delta_vs_fegx_anchor_mev_atom"])),
        ]:
            add_metric(
                rows_by_group["main_table"],
                registry_group="main_table",
                artifact_id="strong_baseline_family_best",
                row_id=str(row["family"]),
                display_label=str(row["best_method"]),
                metric_name=metric_name,
                metric_value=value,
                units="meV/atom",
                source_file=STRONG,
                source_row_key=str(row["family"]),
                sample_level_provenance=str(row["prediction_archive_status"]),
                notes=str(row["subset_alignment"]),
            )

    nooverlap = pd.read_csv(NOOVERLAP)
    exact = nooverlap[nooverlap["evaluation_scope"].eq("exact_no_overlap")].copy()
    formula = nooverlap[nooverlap["evaluation_scope"].eq("formula_heldout_sensitivity")].copy()
    for frame, group_name in [(exact, "main_table"), (formula, "supplement")]:
        artifact_id = "alexandria_exact_no_overlap" if group_name == "main_table" else "alexandria_formula_heldout"
        for row in frame.itertuples(index=False):
            for metric_name, value in [
                ("MAE", float(row.MAE)),
                ("p95", float(row.p95)),
            ]:
                add_metric(
                    rows_by_group[group_name],
                    registry_group=group_name,
                    artifact_id=artifact_id,
                    row_id=str(row.protocol),
                    display_label=f"{row.protocol} {row.evaluation_scope}",
                    metric_name=metric_name,
                    metric_value=value,
                    units="meV/atom",
                    source_file=NOOVERLAP,
                    source_row_key=f"{row.protocol}|{row.evaluation_scope}",
                    sample_level_provenance="package_derived_from_sample_level_predictions",
                    notes=str(row.coverage),
                )

    supplementary_variants = ["M5_reg", "M6", "M7", "M8"]
    for variant in supplementary_variants:
        row = matched[matched["variant"].eq(variant)].iloc[0]
        headline_row = headline_exp_lookup[f"MP_FOLD0_M1M9_{variant}"]
        add_metric(
            rows_by_group["supplement"],
            registry_group="supplement",
            artifact_id="matched_parameterization_extended",
            row_id=variant,
            display_label=str(row["formula"]),
            metric_name="ensemble_test_mae_mev_atom",
            metric_value=float(row["ensemble_test_mae_mev_atom"]),
            units="meV/atom",
            source_file=MATCHED,
            source_row_key=variant,
            sample_level_provenance=str(headline_row["prediction_archive_status"]),
            notes="Extended M1-M9 variant.",
        )

    bootstrap = pd.read_csv(BOOTSTRAP)
    for row in bootstrap.itertuples(index=False):
        row_id = f"{row.variant_a}_vs_{row.variant_b}"
        for metric_name in [
            "delta_mae_a_minus_b_mev_atom",
            "ci95_low_mev_atom",
            "ci95_high_mev_atom",
            "bootstrap_two_sided_p",
            "holm_bonferroni_p",
        ]:
            units = "meV/atom" if "p" not in metric_name else "probability"
            add_metric(
                rows_by_group["bootstrap"],
                registry_group="bootstrap",
                artifact_id="paired_bootstrap_multiple_comparison",
                row_id=row_id,
                display_label=row_id,
                metric_name=metric_name,
                metric_value=float(getattr(row, metric_name)),
                units=units,
                source_file=BOOTSTRAP,
                source_row_key=row_id,
                sample_level_provenance="package_derived_from_sample_level_predictions",
                notes="Holm-adjusted paired bootstrap summary.",
            )

    hull = pd.read_csv(HULL)
    for row in hull.itertuples(index=False):
        row_id = f"{row.subset}|{row.truth_source}|{row.threshold_mev_atom}"
        for metric_name in ["precision", "recall", "accuracy", "f1"]:
            add_metric(
                rows_by_group["hull"],
                registry_group="hull",
                artifact_id="formal_hull_corrected_rerun",
                row_id=row_id,
                display_label=f"{row.subset} {row.truth_source} {row.threshold_mev_atom} meV",
                metric_name=metric_name,
                metric_value=float(getattr(row, metric_name)),
                units="fraction",
                source_file=HULL,
                source_row_key=row_id,
                sample_level_provenance="package_derived_from_sample_level_predictions",
                notes="Corrected formal-hull rerun metric.",
            )

    risk = pd.read_csv(RISK)
    for row in risk.itertuples(index=False):
        row_id = f"{row.protocol}|{row.reference_pair}|rej={row.rejection_pct}"
        for metric_name in ["coverage", "remaining_mae_mev_atom"]:
            units = "fraction" if metric_name == "coverage" else "meV/atom"
            add_metric(
                rows_by_group["risk_curve"],
                registry_group="risk_curve",
                artifact_id="selective_prediction_curve_extension",
                row_id=row_id,
                display_label=f"{row.protocol} {row.reference_pair}",
                metric_name=metric_name,
                metric_value=float(getattr(row, metric_name)),
                units=units,
                source_file=RISK,
                source_row_key=row_id,
                sample_level_provenance="package_derived_from_sample_level_predictions",
                notes=str(row.curve_source),
            )

    rows_by_group["paper_number_registry"] = [
        row
        for group_name, rows in rows_by_group.items()
        if group_name != "paper_number_registry"
        for row in rows
    ]

    headline_rows, headline_metadata = build_registry_rows()
    validate_registry(headline_rows, headline_metadata)
    compare_with_existing(headline_rows, headline_metadata)
    abstract_statuses = {
        str(row["prediction_archive_status"])
        for row in headline_rows
        if str(row["narrative_chain_step"]) not in {"", "nan"}
    }
    validation_report = {
        "headline_validator": "passed",
        "unit_scaling_validator": unit_scaling_report["status"],
        "unit_scaling_checks": int(unit_scaling_report["n_checks"]),
        "unit_scaling_failed_checks": list(unit_scaling_report["failed_check_ids"]),
        "data_leakage_validator": data_leakage_report["status"],
        "data_leakage_checks": int(data_leakage_report["n_checks"]),
        "data_leakage_failed_checks": list(data_leakage_report["failed_check_ids"]),
        "data_leakage_bounded_checks": list(data_leakage_report["bounded_check_ids"]),
        "prediction_archive_validator": prediction_archive_report["status"],
        "prediction_archive_archives": int(prediction_archive_report["n_archives"]),
        "prediction_archive_pass_count": int(prediction_archive_report["pass_count"]),
        "prediction_archive_bounded_count": int(prediction_archive_report["bounded_count"]),
        "prediction_archive_failed_count": int(prediction_archive_report["fail_count"]),
        "prediction_archive_failed_ids": list(prediction_archive_report["failing_archive_ids"]),
        "prediction_archive_bounded_ids": list(prediction_archive_report["bounded_archive_ids"]),
        "missing_source_files": missing_sources,
        "registry_counts": {
            group_name: len(rows)
            for group_name, rows in rows_by_group.items()
            if group_name != "paper_number_registry"
        },
        "headline_archive_statuses": sorted(abstract_statuses),
        "abstract_rows_all_sample_level": abstract_statuses <= {"available_single_file", "available_multi_file"},
        "note": "Top-level paper-number registry now resolves abstract, main-table, supplement, bootstrap, hull, and risk-curve rows through one package entry point.",
    }
    return rows_by_group, validation_report


def write_summary(path: Path, report: dict[str, Any], rows_by_group: dict[str, list[dict[str, Any]]]) -> None:
    counts = report["registry_counts"]
    summary = "\n".join(
        [
            "# 24 Paper Number Pipeline",
            "",
            "This package is the paper-wide number back-calculation and figure-data entry point.",
            "",
            "## Readout",
            "",
            f"1. Unit/scaling validation status: `{report['unit_scaling_validator']}` across `{report['unit_scaling_checks']}` checks.",
            f"2. Data-leakage validation status: `{report['data_leakage_validator']}` across `{report['data_leakage_checks']}` checks.",
            f"3. Prediction-archive validation status: `{report['prediction_archive_validator']}` across `{report['prediction_archive_archives']}` audited archive units.",
            f"4. Headline validation status: `{report['headline_validator']}`.",
            f"5. Abstract registry rows: `{counts['abstract']}`.",
            f"6. Main-table registry rows: `{counts['main_table']}`.",
            f"7. Supplemental registry rows: `{counts['supplement']}`.",
            f"8. Bootstrap registry rows: `{counts['bootstrap']}`.",
            f"9. Hull metric rows: `{counts['hull']}`.",
            f"10. Risk-curve rows: `{counts['risk_curve']}`.",
            f"11. Abstract headline steps all resolve to sample-level archives: `{report['abstract_rows_all_sample_level']}`.",
            "",
            "## Limitation",
            "",
            "- this top-level pipeline consolidates already-validated revision packages rather than rerunning every upstream experiment from raw training checkpoints;",
            "- therefore the reproducibility gate is now paper-wide, but it still inherits the scope limits already recorded inside each source package.",
        ]
    )
    path.write_text(summary + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(OUT)
    rows_by_group, report = build_rows()
    for group_name, rows in rows_by_group.items():
        write_csv(OUT / f"{group_name}.csv", rows, TABLE_FIELDS)
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_summary(OUT / "summary.md", report, rows_by_group)
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# 24 Paper Number Pipeline",
                "",
                "Files:",
                "",
                "- `abstract.csv`: abstract-facing headline numbers.",
                "- `main_table.csv`: main-table number registry.",
                "- `supplement.csv`: supplemental number registry.",
                "- `bootstrap.csv`: paired-bootstrap and multiplicity rows.",
                "- `hull.csv`: corrected hull metrics.",
                "- `risk_curve.csv`: selective-prediction curve rows.",
                "- `paper_number_registry.csv`: combined paper-wide number registry.",
                "- `validation_report.json`: top-level validation status.",
                "- `summary.md`: reviewer-facing summary.",
                "",
                "Boundary:",
                "",
                "- this package is a paper-wide consolidation and validation layer over existing revision-v2 source packages;",
                "- it is intended to be paired with `make_all_paper_figures.py` for figure-ready data extraction.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "package_manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "headline_registry": str(HEADLINE),
                    "matched_parameterization_manifest": str(MATCHED),
                    "strong_baseline_family_best": str(STRONG),
                    "bootstrap_multiple_comparison": str(BOOTSTRAP),
                    "verified_nooverlap_metrics": str(NOOVERLAP),
                    "formal_hull_metrics": str(HULL),
                    "risk_curve_extension": str(RISK),
                },
                "normalized_outputs": [
                    "abstract.csv",
                    "main_table.csv",
                    "supplement.csv",
                    "bootstrap.csv",
                    "hull.csv",
                    "risk_curve.csv",
                    "paper_number_registry.csv",
                    "validation_report.json",
                    "summary.md",
                    "README.md",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "written",
                "out_dir": str(OUT),
                "paper_number_rows": len(rows_by_group["paper_number_registry"]),
                "validation": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
