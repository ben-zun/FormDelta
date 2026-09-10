# WBM Canonical Before/After 20260909

## Corrected Formation Energy

| method                           | pred_col                              |    n |   mae_mev_atom |   rmse_mev_atom |   p95_abs_mev_atom |   bias_mev_atom |
|:---------------------------------|:--------------------------------------|-----:|---------------:|----------------:|-------------------:|----------------:|
| simple_residual                  | pred_simple_residual_ensemble         | 4981 |        4.6382  |         22.4491 |            12.1973 |        -2.00924 |
| simple_residual_seed20260530     | pred_simple_residual_seed20260530     | 4981 |        4.86437 |         15.7296 |            13.7178 |        -1.84448 |
| simple_residual_seed20260529     | pred_simple_residual_seed20260529     | 4981 |        5.04488 |         20.1312 |            13.4903 |        -2.48379 |
| simple_residual_seed20260528     | pred_simple_residual_seed20260528     | 4981 |        5.20279 |         33.9357 |            13.1334 |        -1.69946 |
| canonical_formdelta              | pred_canonical_formdelta_ensemble     | 4981 |        7.43964 |         13.9534 |            22.8824 |        -1.94769 |
| canonical_formdelta_seed20260528 | pred_canonical_formdelta_seed20260528 | 4981 |        7.69317 |         14.5788 |            24.0371 |        -2.25527 |
| canonical_formdelta_seed20260530 | pred_canonical_formdelta_seed20260530 | 4981 |        8.20295 |         14.5534 |            25.1024 |        -1.58243 |
| canonical_formdelta_seed20260529 | pred_canonical_formdelta_seed20260529 | 4981 |        8.64524 |         15.7216 |            27.1409 |        -2.00536 |
| matris_affine_base               | pred_matris_affine_base               | 4981 |       71.5013  |        111.266  |           213.444  |       -23.6184  |
| composition_only_linear          | pred_composition_only_linear          | 4981 |       79.2202  |        120.988  |           232.324  |       -23.9075  |
| matris_element_reference         | pred_matris_element_reference         | 4981 |      302.763   |        415.525  |           862.89   |       -19.0168  |
| matris_raw_as_formation          | pred_matris_raw_as_formation          | 4981 |     5548.28    |       5900.52   |          9082.49   |     -5548.28    |

## Formal Hull Classification

| method                   |    n |   base_rate |   precision |     recall |        f1 |   discovery_acceleration_factor |
|:-------------------------|-----:|------------:|------------:|-----------:|----------:|--------------------------------:|
| matris_raw_as_formation  | 4981 |    0.328247 |   0.328247  | 1          | 0.494256  |                        1        |
| matris_element_reference | 4981 |    0.328247 |   0.269811  | 0.0874618  | 0.132102  |                        0.821976 |
| composition_only_linear  | 4981 |    0.328247 |   0.109375  | 0.0299694  | 0.0470475 |                        0.333209 |
| matris_affine_base       | 4981 |    0.328247 |   0.109005  | 0.0281346  | 0.0447253 |                        0.332081 |
| simple_residual          | 4981 |    0.328247 |   0.0977444 | 0.00795107 | 0.0147059 |                        0.297777 |
| canonical_formdelta      | 4981 |    0.328247 |   0.0833333 | 0.00733945 | 0.0134907 |                        0.253874 |

## EF@50

| method                   |   k |   precision |     recall |   enrichment_factor |
|:-------------------------|----:|------------:|-----------:|--------------------:|
| simple_residual          |  50 |        0.5  | 0.0116117  |            1.15676  |
| matris_element_reference |  50 |        0.5  | 0.0116117  |            1.15676  |
| canonical_formdelta      |  50 |        0.34 | 0.00789596 |            0.786595 |
| matris_affine_base       |  50 |        0.32 | 0.00743149 |            0.740325 |
| composition_only_linear  |  50 |        0.28 | 0.00650255 |            0.647784 |
| matris_raw_as_formation  |  50 |        0.28 | 0.00650255 |            0.647784 |

## Paired Bootstrap MAE

| comparison                                             | method                           |   base_mae_mev_atom |   method_mae_mev_atom |   delta_base_minus_method_mev_atom |   ci95_low_mev_atom |   ci95_high_mev_atom | interpretation            |
|:-------------------------------------------------------|:---------------------------------|--------------------:|----------------------:|-----------------------------------:|--------------------:|---------------------:|:--------------------------|
| matris_affine_base_vs_matris_raw_as_formation          | matris_raw_as_formation          |             87.6527 |             5536.54   |                        -5448.89    |          -5504.84   |          -5392.4     | worse_ci_excludes_0       |
| matris_affine_base_vs_matris_element_reference         | matris_element_reference         |             87.6527 |              319.117  |                         -231.464   |           -239.452  |           -223.68    | worse_ci_excludes_0       |
| matris_affine_base_vs_composition_only_linear          | composition_only_linear          |             87.6527 |               95.2744 |                           -7.62166 |             -8.311  |             -6.93764 | worse_ci_excludes_0       |
| matris_affine_base_vs_simple_residual_seed20260528     | simple_residual_seed20260528     |             87.6527 |               10.8751 |                           76.7776  |             71.2796 |             82.7351  | improvement_ci_excludes_0 |
| matris_affine_base_vs_simple_residual_seed20260529     | simple_residual_seed20260529     |             87.6527 |               10.6781 |                           76.9746  |             71.4847 |             83.2298  | improvement_ci_excludes_0 |
| matris_affine_base_vs_simple_residual_seed20260530     | simple_residual_seed20260530     |             87.6527 |               10.621  |                           77.0317  |             71.5668 |             83.2926  | improvement_ci_excludes_0 |
| matris_affine_base_vs_simple_residual                  | simple_residual                  |             87.6527 |               10.3271 |                           77.3256  |             71.6163 |             83.4561  | improvement_ci_excludes_0 |
| matris_affine_base_vs_canonical_formdelta_seed20260528 | canonical_formdelta_seed20260528 |             87.6527 |               12.1532 |                           75.4996  |             69.7306 |             82.1279  | improvement_ci_excludes_0 |
| matris_affine_base_vs_canonical_formdelta_seed20260529 | canonical_formdelta_seed20260529 |             87.6527 |               13.3908 |                           74.262   |             68.3836 |             80.9175  | improvement_ci_excludes_0 |
| matris_affine_base_vs_canonical_formdelta_seed20260530 | canonical_formdelta_seed20260530 |             87.6527 |               13.3827 |                           74.27    |             68.5798 |             80.5532  | improvement_ci_excludes_0 |
| matris_affine_base_vs_canonical_formdelta              | canonical_formdelta              |             87.6527 |               12.1762 |                           75.4766  |             69.7719 |             81.9119  | improvement_ci_excludes_0 |

## Protocol Audit

- Passed: 6/6.

## Boundary

- This WBM rerun uses corrected published per-atom formation-energy labels from the WBM manifest.
- Foundation predictions and caches are reused; no MatRIS or EqV3 foundation inference is rerun.
- The canonical FormDelta row is a 3-layer MLP fixed-anchor residual on a train-only affine MatRIS base.
