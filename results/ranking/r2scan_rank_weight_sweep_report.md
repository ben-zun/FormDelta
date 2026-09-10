# r2SCAN Ranking-Aware Lambda Sweep

Generated: 2026-09-10T01:05:38+09:00

Protocol: canonical fixed-anchor FormDelta, 3-layer MLP, hidden 256, dropout 0.05, no MoE/env/dual/missing branches.
Backbone/foundation predictions, composition features, and structure cache were reused; no large-model inference was rerun.

## Files

- Summary by run: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/summary_by_run.csv`
- Aggregate by rank weight: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/aggregate_by_rank_weight_split.csv`
- Ranking summary: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/ranking_summary_by_run.csv`
- Ranking aggregate: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/aggregate_ranking_by_method_family.csv`
- MAE bootstrap: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/paired_bootstrap_mae_vs_lambda0.csv`
- Ranking bootstrap: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/paired_bootstrap_ranking_vs_lambda0.csv`
- Config audit: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/config_audit.csv`
- Leakage audit: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/leakage_audit.csv`
- Cost accounting: `<RUNS_ROOT>/r2scan_rank_weight_sweep_20260909/tables/cost_accounting.csv`

## Test MAE Pareto

|   rank_weight |   n_runs |   mae_mev_atom_mean |   mae_mev_atom_std |   rmse_mev_atom_mean |   p95_abs_mev_atom_mean |   best_epoch_mean |   runtime_s_mean |
|--------------:|---------:|--------------------:|-------------------:|---------------------:|------------------------:|------------------:|-----------------:|
|          0    |        3 |             17.8055 |           0.379058 |              40.93   |                 57.1339 |           196.333 |            0     |
|          0.05 |        3 |             18.1186 |           0.312789 |              41.3925 |                 58.5994 |           196     |          433.833 |
|          0.1  |        3 |             18.6792 |           0.372358 |              42.6702 |                 60.1214 |           198     |          435.267 |
|          0.2  |        3 |             19.6625 |           0.323972 |              44.4798 |                 61.923  |           196.333 |          323.971 |

## Ranking Metrics

| method_family             |   rank_weight |   n_runs |   mae_mev_atom_mean |   pairwise_order_accuracy_mean |   top1_accuracy_mean |   near_degenerate_pair_accuracy_mean |   spearman_pair_weighted_mean_mean |
|:--------------------------|--------------:|---------:|--------------------:|-------------------------------:|---------------------:|-------------------------------------:|-----------------------------------:|
| rankw000                  |          0    |        3 |             17.8055 |                       0.755    |             0.696581 |                             0.719921 |                           0.569588 |
| rankw005                  |          0.05 |        3 |             18.1186 |                       0.736667 |             0.705128 |                             0.698225 |                           0.548571 |
| rankw010                  |          0.1  |        3 |             18.6792 |                       0.728333 |             0.713675 |                             0.682446 |                           0.522333 |
| rankw020                  |          0.2  |        3 |             19.6625 |                       0.721667 |             0.705128 |                             0.686391 |                           0.519952 |
| affine_energy_composition |        nan    |        3 |            104.158  |                       0.805    |             0.705128 |                             0.781065 |                           0.661143 |
| composition_only_linear   |        nan    |        3 |            374.516  |                       0        |             0.461538 |                             0        |                         nan        |
| element_reference         |        nan    |        3 |            103.989  |                       0.805    |             0.705128 |                             0.781065 |                           0.661143 |
| raw                       |        nan    |        3 |           4511.15   |                       0.805    |             0.705128 |                             0.781065 |                           0.661143 |

## Audit

- Config audit pass: 12/12

- Leakage central-key overlaps: 0

- Accounted adapter GPU-hours: 0.9942

## Interpretation Guardrails

- Formation-energy MAE is bootstrapped over structure rows.
- Ranking metrics are audited over formula groups because pairwise comparisons are nested within formula.
- This sweep is a B-level Pareto experiment: it can support a ranking tradeoff claim only if the CI and MAE tradeoff agree.
