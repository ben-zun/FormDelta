# Model Registry

Fill the official version, URL, license, and SHA256 fields before public release. Do not upload third-party foundation weights unless their licenses permit redistribution.

Primary registry table: `data/manifests/backbone_registry.csv`.

Custom residual checkpoints included in this snapshot:

- `checkpoints/megnet/`
- `checkpoints/cross_backbone/`
- `checkpoints/r2scan/`
- `checkpoints/wbm/`
- `checkpoints/ranking/`

The MEGNet foundation checkpoint is copied under `checkpoints/foundation/` for local continuity from the prior package, but its public redistribution route must be confirmed.
