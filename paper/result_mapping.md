# Paper Result Mapping

| Paper item | Source data | Script | Expected result |
|---|---|---|---|
| MEGNet main controlled split | `results/megnet/predictions/megnet_main_test_predictions.csv` | `scripts/evaluate.py` | 83.372 to 62.090 meV/atom |
| MEGNet scaling | `results/megnet/scaling_summary.csv` | `scripts/reproduce_megnet.py` | 1/5/10/25/50/100 percent label fractions |
| MEGNet overlap audit | `data/manifests/megnet_pretrain_overlap.csv` | `scripts/build_pretraining_overlap.py` | structure/formula/chemical-system overlap flags |
| Canonical cross-backbone table | `results/cross_backbone/aggregate_by_family.csv` | `scripts/reproduce_cross_backbone.py` | MatRIS 29.27, EqV3 30.06, MACE 33.01, SevenNet 61.43, CHGNet 28.15, EqV3-LoRA 11.99, ALIGNN 68.18 |
| r2SCAN full-label baseline comparison | `results/r2scan/aggregate_by_method_fraction.csv` | `scripts/reproduce_r2scan.py` | classical delta 16.42, FormDelta 17.81 |
| WBM before/after formation energy | `results/wbm/formation_metrics.csv` | `scripts/reproduce_wbm.py` | MatRIS affine, simple residual, and FormDelta before/after metrics |
| WBM fixed-hull analysis | `results/wbm/formal_hull_metrics.csv` and `results/wbm/predictions/formal_hull_predictions.csv` | `scripts/reproduce_wbm.py` | fixed-hull classification/top-k caveat |
| Same-formula ranking | `data/manifests/same_formula_pairs.csv` and `results/ranking/megnet_same_formula_ranking_metrics.csv` | `scripts/build_same_formula_ranking.py` | 39 formula groups, 98 pairs, 46 near-degenerate pairs |
| r2SCAN ranking lambda sweep | `results/ranking/lambda_sweep.csv` | `scripts/reproduce_ranking.py` | lambda sweep negative result |
| MP to JARVIS supplement | `data/manifests/mp_jarvis_matching.csv` and `results/mp_jarvis/` | `scripts/reproduce_ranking.py` | auxiliary cross-source evidence only |
| Config/leakage audits | `audits/` | `scripts/run_all_audits.sh` | canonical 23/23, r2SCAN 72/72, WBM 6/6, ranking 12/12 audits pass |
| GPU/cost accounting | `audits/cost_accounting.csv` | `scripts/evaluate.py` | no foundation inference rerun in canonical residual protocols |
