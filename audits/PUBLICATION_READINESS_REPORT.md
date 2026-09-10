# Publication Readiness Report

Generated: 2026-09-10

Tracked files in snapshot: 12043

Tracked file payload size: 771.2 MiB

## Complete or directly present

Most implementation, configuration, MEGNet main predictions, exact split manifests, residual checkpoints, audit summaries, result tables, and paper result mappings are present.

For GitHub transport reliability, the three largest local CSV artifacts are committed as `.csv.gz`: `data/cache/composition_features.csv.gz`, `data/cache/structure_metadata.csv.gz`, and `data/manifests/historical_dataset_manifest_20260626.csv.gz`. Run `bash scripts/materialize_large_csvs.sh` to restore the original `.csv` filenames after cloning.

## Validation performed on this snapshot

- `bash reproduce_from_predictions.sh`: passed; MEGNet test rows = 2048, base MAE = 83.372 meV/atom, simple residual MAE = 75.271 meV/atom, FormDelta MAE = 62.090 meV/atom; canonical/r2SCAN/ranking/WBM config audits passed; central leakage failures = 0.
- `PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile $(find formdelta scripts figures/scripts -name '*.py')`: passed.
- `/tmp/formdelta_pytest_venv/bin/python -m pytest tests -q`: 4 passed, 1 skipped.
- `find . -type f -size +90M -print`: no files reported.
- Secret/private-path scan for local paths, host IPs, API keys, passwords, GitHub PATs, and OpenAI-style keys: no relevant matches.

## Pending author/release actions

- backbone checkpoint/version registry: Family rows present; official version/hash/license pending.
- LICENSE: License placeholder, not invented.
- CITATION.cff: Citation scaffold, DOI/authors pending.
- GitHub release v1.0.0-paper: Release cannot be created locally without repository owner action.
- Zenodo DOI: DOI pending final archive.

## Partial or external-deposit items

- MEGNet scaling fraction manifests: Available sample manifests copied; deterministic residual fractions documented.
- cross-backbone foundation prediction caches: Large CSVs listed for LFS/Zenodo.
- classical delta and FormDelta predictions: Full prediction CSVs are large; summary tables included.
- environment lock files: Final pip freeze pending.

## Audit commands

```bash
bash reproduce_from_predictions.sh
python -m py_compile $(find formdelta scripts figures/scripts -name '*.py')
python -m pytest tests -q
```
