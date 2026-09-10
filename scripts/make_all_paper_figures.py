#!/usr/bin/env python3
"""Materialize figure-ready data extracts for the paper-wide revision package."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from run_paper_closure_331_341_consolidation import ensure_dir, write_csv

ROOT = Path(os.environ.get("FORMDELTA_ROOT", Path(__file__).resolve().parents[1])).resolve()
PAPER = ROOT / "paper_revision_v2"
OUT = PAPER / "24_paper_number_pipeline"
FIG = OUT / "figure_data"

HEADLINE = PAPER / "headline_result_registry.csv"
MATCHED = PAPER / "01_matched_parameterization" / "fullrun_5seed_r20260710" / "model_manifest.csv"
LADDER = PAPER / "02_strong_stacking_delta" / "comparison_ladder_same_subset.csv"
BOOTSTRAP = PAPER / "06_statistics" / "paired_bootstrap_multiple_comparison.csv"
NOOVERLAP = PAPER / "05_provider_exposure" / "alexandria_verified_nooverlap_pbesol" / "protocol_scope_metrics.csv"
HULL = PAPER / "10_convex_hull" / "formal_hull_corrected_rerun_metrics.csv"
RISK = PAPER / "14_selective_prediction" / "selective_prediction_curve_extension.csv"

REGISTRY_FIELDS = ["figure_id", "generated_file", "source_file", "x_field", "y_field", "purpose", "notes"]


def main() -> None:
    ensure_dir(FIG)
    registry_rows: list[dict[str, Any]] = []

    headline = pd.read_csv(HEADLINE)
    f1 = headline.copy()
    f1["narrative_chain_step"] = pd.to_numeric(f1["narrative_chain_step"], errors="coerce")
    f1 = f1[f1["narrative_chain_step"].notna()].copy()
    f1["narrative_chain_step"] = f1["narrative_chain_step"].astype(int)
    f1 = f1.sort_values("narrative_chain_step")[
        ["narrative_chain_step", "experiment_id", "mae_mev_atom", "comparability_family", "prediction_archive_status"]
    ]
    f1_path = FIG / "F01_headline_chain.csv"
    f1.to_csv(f1_path, index=False)
    registry_rows.append(
        {
            "figure_id": "F01",
            "generated_file": str(f1_path),
            "source_file": str(HEADLINE),
            "x_field": "narrative_chain_step",
            "y_field": "mae_mev_atom",
            "purpose": "headline mixed-chain figure",
            "notes": "Shows the 94.517 -> 5.875 paper headline path with comparability labels.",
        }
    )

    matched = pd.read_csv(MATCHED)
    f2 = matched[
        [
            "variant",
            "formula",
            "ensemble_test_mae_mev_atom",
            "ensemble_test_p95_mev_atom",
            "prediction_mode",
            "feature_mode",
        ]
    ].copy()
    f2_path = FIG / "F02_m1m9_matrix.csv"
    f2.to_csv(f2_path, index=False)
    registry_rows.append(
        {
            "figure_id": "F02",
            "generated_file": str(f2_path),
            "source_file": str(MATCHED),
            "x_field": "variant",
            "y_field": "ensemble_test_mae_mev_atom",
            "purpose": "matched-parameterization matrix figure",
            "notes": "All M1-M9 rows for the matched fairness figure.",
        }
    )

    ladder = pd.read_csv(LADDER).head(15).copy()
    f3_path = FIG / "F03_strong_baseline_ladder.csv"
    ladder.to_csv(f3_path, index=False)
    registry_rows.append(
        {
            "figure_id": "F03",
            "generated_file": str(f3_path),
            "source_file": str(LADDER),
            "x_field": "rank_by_mae",
            "y_field": "mae_mev_atom",
            "purpose": "strong-baseline ladder figure",
            "notes": "Top 15 same-subset ladder rows.",
        }
    )

    nooverlap = pd.read_csv(NOOVERLAP)
    f4_path = FIG / "F04_verified_nooverlap_protocols.csv"
    nooverlap.to_csv(f4_path, index=False)
    registry_rows.append(
        {
            "figure_id": "F04",
            "generated_file": str(f4_path),
            "source_file": str(NOOVERLAP),
            "x_field": "protocol",
            "y_field": "MAE",
            "purpose": "verified no-overlap external figure",
            "notes": "Exact-clean and formula-heldout protocol rows.",
        }
    )

    bootstrap = pd.read_csv(BOOTSTRAP)
    f5_path = FIG / "F05_bootstrap_registry.csv"
    bootstrap.to_csv(f5_path, index=False)
    registry_rows.append(
        {
            "figure_id": "F05",
            "generated_file": str(f5_path),
            "source_file": str(BOOTSTRAP),
            "x_field": "variant_a",
            "y_field": "delta_mae_a_minus_b_mev_atom",
            "purpose": "bootstrap significance figure",
            "notes": "Full paired-bootstrap registry for forest/heatmap style rendering.",
        }
    )

    hull = pd.read_csv(HULL)
    f6_path = FIG / "F06_formal_hull_metrics.csv"
    hull.to_csv(f6_path, index=False)
    registry_rows.append(
        {
            "figure_id": "F06",
            "generated_file": str(f6_path),
            "source_file": str(HULL),
            "x_field": "threshold_mev_atom",
            "y_field": "f1",
            "purpose": "corrected formal-hull figure",
            "notes": "Threshold-swept hull metrics.",
        }
    )

    risk = pd.read_csv(RISK)
    f7_path = FIG / "F07_selective_risk_curves.csv"
    risk.to_csv(f7_path, index=False)
    registry_rows.append(
        {
            "figure_id": "F07",
            "generated_file": str(f7_path),
            "source_file": str(RISK),
            "x_field": "coverage",
            "y_field": "remaining_mae_mev_atom",
            "purpose": "selective-prediction risk curve figure",
            "notes": "Coverage-vs-risk curve data.",
        }
    )

    write_csv(OUT / "figure_source_registry.csv", registry_rows, REGISTRY_FIELDS)
    (OUT / "figure_summary.md").write_text(
        "\n".join(
            [
                "# Figure Data Registry",
                "",
                "This file records the figure-ready CSV extracts generated by `make_all_paper_figures.py`.",
                "",
                f"- figure datasets written: `{len(registry_rows)}`",
                f"- output directory: `{FIG}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "figure_manifest.json").write_text(
        json.dumps(
            {
                "figure_sources": registry_rows,
                "outputs": [
                    "figure_source_registry.csv",
                    "figure_summary.md",
                    "figure_manifest.json",
                    "figure_data/",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    package_manifest = OUT / "package_manifest.json"
    if package_manifest.exists():
        payload = json.loads(package_manifest.read_text(encoding="utf-8"))
        outputs = list(payload.get("outputs", []))
        for item in ["figure_source_registry.csv", "figure_summary.md", "figure_manifest.json", "figure_data/"]:
            if item not in outputs:
                outputs.append(item)
        payload["outputs"] = outputs
        payload["figure_source_registry"] = str(OUT / "figure_source_registry.csv")
        package_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "written",
                "figure_count": len(registry_rows),
                "out_dir": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
