# FormDelta

Residual adaptation of pretrained materials models for formation-energy prediction.

## What is FormDelta?

FormDelta keeps a pretrained or train-only calibrated materials model fixed and learns only a residual head:

```text
y_hat(x) = R(x) + r_hat(x)
```

`R(x)` is the frozen foundation-model prediction or calibrated reference. `r_hat(x)` is the target-data residual learned by a small adapter. The canonical FormDelta protocol in this snapshot is a unified three-layer MLP with `hidden_dim=256`, `dropout=0.05`, `SiLU`, `LayerNorm`, `SmoothL1`, and `AdamW`; foundation model weights are not retrained.

## Locked paper-facing results

| Experiment | Base | FormDelta |
|---|---:|---:|
| MEGNet | 83.372 | 62.090 meV/atom |
| MatRIS | 112.23 | 29.27 |
| EqV3 | 112.97 | 30.06 |
| MACE | 112.54 | 33.01 |
| SevenNet | 86.23 | 61.43 |
| CHGNet | 33.76 | 28.15 |
| EqV3-LoRA | 13.70 | 11.99 |
| ALIGNN | 69.06 | 68.18 |
| PBE/GGA to r2SCAN | -- | 17.81 |

Classical delta-learning reaches 16.42 meV/atom under the same full-label r2SCAN setting; FormDelta is not the strongest r2SCAN baseline in that comparison.

## Reproduce from saved artifacts

```bash
bash reproduce_from_predictions.sh
```

This read-only path checks the included MEGNet per-sample prediction table, config audits, leakage audits, and locked result tables. Full foundation-model inference is intentionally not rerun.

## Retrain residual adapters

```bash
bash retrain_adapters.sh
```

Use a GPU environment with the dependencies in `environment.yml` and the exact inputs listed under `data/`. Large frozen prediction caches should be supplied through Git LFS or a DOI-backed repository when not committed directly.

## Repository guide

- `formdelta/`: canonical residual adapter, reference construction, feature contracts, training and evaluation utilities.
- `configs/`: locked canonical and experiment-specific configuration files.
- `data/`: exact split manifests, foundation prediction inputs, calibration coefficients, checksums, and large-artifact manifest.
- `results/`: MEGNet, cross-backbone, r2SCAN, WBM, ranking, and MP-JARVIS result tables.
- `audits/`: config, leakage, bootstrap, and cost-accounting audits.
- `figures/`: source data and rendered MEGNet diagnostic figures.
- `paper/result_mapping.md`: manuscript result to file/script mapping.

## Historical naming

Earlier internal runs used the label FE-GX. In this repository, FormDelta is the manuscript-facing method name. Historical filenames are retained only where necessary for provenance.
