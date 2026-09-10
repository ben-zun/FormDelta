# WBM Before/After Results

This directory contains formation-energy metrics, fixed-hull metrics, top-k enrichment, prediction matrices, bootstrap tests, and audit files for the WBM 50k experiment.

The fixed-hull evaluation uses:

```text
E_hull_pred = E_hull_DFT + E_f_pred - E_f_DFT
```

Formation-energy MAE improves strongly, but published-hull classification and top-k behavior do not uniformly improve.
