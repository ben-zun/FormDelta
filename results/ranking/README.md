# Same-Formula Ranking

`data/manifests/same_formula_pairs.csv` lists exact MEGNet same-formula pairs reconstructed from the locked test predictions. `lambda_sweep.csv` records the r2SCAN ranking-loss sweep for lambda = 0, 0.05, 0.10, and 0.20.

The lambda sweep is a negative result: increasing the ranking weight worsened MAE and did not reliably improve pairwise ordering.
