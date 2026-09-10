# Data Layout

`splits/` contains exact train/val/test manifests and deterministic fraction selections. `foundation_predictions/` contains the MEGNet frozen prediction cache copied from the prior reproducibility package; larger foundation prediction caches are listed in `manifests/large_artifacts_manifest.csv`.

The large local CSV caches `cache/composition_features.csv`, `cache/structure_metadata.csv`, and `manifests/historical_dataset_manifest_20260626.csv` are committed as `.csv.gz` files for GitHub transport. Run `bash scripts/materialize_large_csvs.sh` from the repository root to restore the original CSV filenames when needed for retraining or local audit work.

`coefficients/` contains train-only calibration coefficients and input manifests for cross-backbone, r2SCAN, and WBM experiments. `checksums/` contains original package checksums plus the generated snapshot SHA256 manifest.

Third-party raw databases are not automatically relicensed by this repository. Keep provider citations and access routes in the manuscript and repository release metadata.
