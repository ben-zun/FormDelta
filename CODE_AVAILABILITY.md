# Code Availability

The code used to build references, train the simple residual adapter and canonical FormDelta residual head, evaluate MEGNet/cross-backbone/r2SCAN/WBM/ranking results, and run leakage/config/bootstrap audits is included in this repository snapshot. The final public release should be tagged as `v1.0.0-paper` and archived with a persistent DOI. Final URL/DOI: `AUTHOR_INPUT_NEEDED`.

The scripts in `scripts/` provide two levels of reuse:

- `reproduce_from_predictions.sh` verifies locked results from saved predictions and summary tables.
- `retrain_adapters.sh` reruns residual-adapter training when full prediction caches, feature caches, structure caches, and checkpoints are available.

Foundation-model weights from third-party projects are not rehosted here unless redistribution rights are confirmed. Use `MODEL_REGISTRY.md` and `data/manifests/backbone_registry.csv` for exact external model identity.
