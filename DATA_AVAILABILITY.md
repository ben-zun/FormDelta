# Data Availability

The fixed target splits, MEGNet CIF subset, frozen MEGNet predictions, derived feature caches, calibration coefficients, per-sample MEGNet results, audit tables, and figure source data supporting this study are provided in this repository snapshot, subject to the rights checks below. Larger cross-backbone, r2SCAN, WBM, and MP-JARVIS prediction caches are listed in `data/manifests/large_artifacts_manifest.csv` and should be deposited as Git LFS assets, a GitHub release asset, Zenodo, Figshare, or another DOI-backed repository before manuscript submission. Insert the final repository URL/DOI here: `AUTHOR_INPUT_NEEDED`.

Public datasets reused in the analysis include Materials Project/Matbench-derived formation-energy structures, JARVIS-derived labels, WBM/Matbench Discovery data, and pretrained-model outputs. The original third-party datasets are not claimed as newly generated data by this project and must be cited according to their provider terms. Confirm redistribution rights for CIFs, cached features, MEGNet checkpoint files, and any Materials Project-derived content before public release.

Latent-feature delta-learning for r2SCAN was not evaluated because a split-aligned r2SCAN latent representation cache was unavailable; this is recorded in `results/r2scan/latent_feature_availability.csv`.

## Repository and citation actions

- Replace `AUTHOR_INPUT_NEEDED` placeholders with the final GitHub URL, release tag, and archive DOI.
- Add dataset DOI/accession records for large prediction caches and checkpoints if they are kept outside Git.
- Confirm separate code and data licenses.
- Keep third-party source IDs, exact split manifests, derived predictions, derived statistics, and checksums even if raw third-party databases are not redistributed.

## Missing information / risk flags

- Final repository DOI is pending.
- Final software license is pending.
- Third-party data and model-weight redistribution rights must be confirmed.
- Official backbone checkpoint versions, URLs, licenses, and SHA256 hashes need author confirmation in `data/manifests/backbone_registry.csv`.
