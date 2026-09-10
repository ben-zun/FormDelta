# Data Layout

`splits/` contains exact train/val/test manifests and deterministic fraction selections. `foundation_predictions/` contains the MEGNet frozen prediction cache copied from the prior reproducibility package; larger foundation prediction caches are listed in `manifests/large_artifacts_manifest.csv`.

`coefficients/` contains train-only calibration coefficients and input manifests for cross-backbone, r2SCAN, and WBM experiments. `checksums/` contains original package checksums plus the generated snapshot SHA256 manifest.

Third-party raw databases are not automatically relicensed by this repository. Keep provider citations and access routes in the manuscript and repository release metadata.
