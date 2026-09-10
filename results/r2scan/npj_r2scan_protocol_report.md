# npj r2SCAN Protocol 20260909

## Scope

- Task: MP-PBE/GGA provider prediction to MP-r2SCAN formation-energy transfer.
- Foundation predictions are frozen and reused from the provider bakeoff cache.
- Calibration coefficients are refit on only the selected train subset for each fraction/seed.
- Independent unit for bootstrap and leakage audit: structure row (`id`, `cif_path`).

## Test MAE Aggregate

| method                     |   fraction |   n_runs |   mae_mev_atom_mean |   mae_mev_atom_std |   rmse_mev_atom_mean |   p95_abs_mev_atom_mean |
|:---------------------------|-----------:|---------:|--------------------:|-------------------:|---------------------:|------------------------:|
| classical_delta            |       0.01 |        3 |            118.314  |        6.97599     |             265.742  |                422.491  |
| formdelta                  |       0.01 |        3 |            147.061  |        3.35435     |             280.156  |                468.373  |
| element_reference          |       0.01 |        3 |            149.926  |        3.66694     |             280.3    |                458.678  |
| element_reference_internal |       0.01 |        9 |            149.926  |        3.17566     |             280.3    |                458.678  |
| affine_energy_composition  |       0.01 |        3 |            150.265  |        3.93407     |             280.031  |                459.599  |
| feature_stacking           |       0.01 |        3 |            193.39   |        3.6616      |             295.785  |                572.575  |
| composition_only_linear    |       0.01 |        3 |            466.923  |        6.44421     |             668.485  |               1317.27   |
| target_only                |       0.01 |        3 |            484.558  |       13.7313      |             679.642  |               1452.23   |
| raw                        |       0.01 |        3 |           4511.15   |        0           |            4891.87   |               8016.21   |
| classical_delta            |       0.05 |        3 |             47.9384 |        3.62436     |             116.176  |                157.864  |
| formdelta                  |       0.05 |        3 |             81.9045 |        0.322124    |             160.656  |                306.119  |
| element_reference_internal |       0.05 |        9 |            109.843  |        1.62814     |             183.635  |                367.456  |
| element_reference          |       0.05 |        3 |            109.843  |        1.88001     |             183.635  |                367.456  |
| affine_energy_composition  |       0.05 |        3 |            110.143  |        2.2131      |             183.807  |                367.573  |
| feature_stacking           |       0.05 |        3 |            111.789  |        3.66086     |             187.082  |                356.638  |
| target_only                |       0.05 |        3 |            350.359  |        3.34983     |             513.428  |               1116.46   |
| composition_only_linear    |       0.05 |        3 |            394.069  |        3.35628     |             570.352  |               1161.29   |
| raw                        |       0.05 |        3 |           4511.15   |        0           |            4891.87   |               8016.21   |
| classical_delta            |       0.1  |        3 |             33.1219 |        0.469021    |              88.9646 |                104.926  |
| formdelta                  |       0.1  |        3 |             58.5639 |        0.272383    |             123.434  |                208.178  |
| feature_stacking           |       0.1  |        3 |             85.4831 |        1.26903     |             154.765  |                290.434  |
| element_reference_internal |       0.1  |        9 |            107.757  |        0.975555    |             179.462  |                366.335  |
| element_reference          |       0.1  |        3 |            107.757  |        1.12647     |             179.462  |                366.334  |
| affine_energy_composition  |       0.1  |        3 |            107.93   |        1.41797     |             179.52   |                368.346  |
| target_only                |       0.1  |        3 |            271.567  |        8.25996     |             411.254  |                868.504  |
| composition_only_linear    |       0.1  |        3 |            383.497  |        3.10612     |             559.367  |               1149.82   |
| raw                        |       0.1  |        3 |           4511.15   |        0           |            4891.87   |               8016.21   |
| classical_delta            |       0.25 |        3 |             24.199  |        0.353259    |              56.9142 |                 77.1352 |
| formdelta                  |       0.25 |        3 |             35.5026 |        0.412399    |              71.4495 |                122.575  |
| feature_stacking           |       0.25 |        3 |             58.4651 |        0.59666     |             116.568  |                196.451  |
| element_reference_internal |       0.25 |        9 |            105.237  |        0.831307    |             167.717  |                359.62   |
| element_reference          |       0.25 |        3 |            105.237  |        0.959913    |             167.717  |                359.62   |
| affine_energy_composition  |       0.25 |        3 |            105.359  |        1.11165     |             167.738  |                360.146  |
| target_only                |       0.25 |        3 |            170.666  |        6.18461     |             274.368  |                528.336  |
| composition_only_linear    |       0.25 |        3 |            375.869  |        0.699024    |             538.668  |               1153.24   |
| raw                        |       0.25 |        3 |           4511.15   |        0           |            4891.87   |               8016.21   |
| classical_delta            |       0.5  |        3 |             20.2829 |        0.827403    |              52.1794 |                 60.4736 |
| formdelta                  |       0.5  |        3 |             25.24   |        0.494339    |              56.6929 |                 84.1207 |
| feature_stacking           |       0.5  |        3 |             39.7272 |        0.522436    |              88.9459 |                124.125  |
| target_only                |       0.5  |        3 |            104.47   |        2.558       |             182.121  |                304.064  |
| element_reference_internal |       0.5  |        9 |            104.751  |        0.432217    |             167.001  |                362.787  |
| element_reference          |       0.5  |        3 |            104.751  |        0.499083    |             167.001  |                362.787  |
| affine_energy_composition  |       0.5  |        3 |            104.901  |        0.737695    |             167.032  |                362.965  |
| composition_only_linear    |       0.5  |        3 |            374.687  |        0.319506    |             538.102  |               1152.95   |
| raw                        |       0.5  |        3 |           4511.15   |        0           |            4891.87   |               8016.21   |
| classical_delta            |       1    |        3 |             16.4248 |        0.08481     |              41.3429 |                 48.805  |
| formdelta                  |       1    |        3 |             17.8055 |        0.379058    |              40.93   |                 57.1339 |
| feature_stacking           |       1    |        3 |             30.2892 |        0.123869    |              76.2329 |                 91.139  |
| target_only                |       1    |        3 |             65.6979 |        0.449452    |             132.243  |                187.405  |
| element_reference          |       1    |        3 |            103.989  |        0           |             166.802  |                364.127  |
| element_reference_internal |       1    |        9 |            103.989  |        0           |             166.802  |                364.127  |
| affine_energy_composition  |       1    |        3 |            104.158  |        0           |             166.829  |                363.328  |
| composition_only_linear    |       1    |        3 |            374.516  |        6.96187e-14 |             537.905  |               1161.01   |
| raw                        |       1    |        3 |           4511.15   |        0           |            4891.87   |               8016.21   |

## Protocol Audit

- Passed: 72/72.

## Paired Bootstrap

|   fraction |     seed |   delta_base_minus_method_mev_atom |   ci95_low_mev_atom |   ci95_high_mev_atom | interpretation            |
|-----------:|---------:|-----------------------------------:|--------------------:|---------------------:|:--------------------------|
|       0.01 | 20260528 |                            4.76554 |             3.16108 |              6.37189 | improvement_ci_excludes_0 |
|       0.01 | 20260529 |                            1.94688 |             0.6897  |              3.19517 | improvement_ci_excludes_0 |
|       0.01 | 20260530 |                            1.88332 |             1.24179 |              2.51906 | improvement_ci_excludes_0 |
|       0.05 | 20260528 |                           29.7642  |            27.0195  |             32.4682  | improvement_ci_excludes_0 |
|       0.05 | 20260529 |                           27.2763  |            24.8741  |             29.6791  | improvement_ci_excludes_0 |
|       0.05 | 20260530 |                           26.7758  |            24.4031  |             29.2499  | improvement_ci_excludes_0 |
|       0.1  | 20260528 |                           50.3003  |            47.1917  |             53.4452  | improvement_ci_excludes_0 |
|       0.1  | 20260529 |                           49.3901  |            46.2221  |             52.5736  | improvement_ci_excludes_0 |
|       0.1  | 20260530 |                           47.8893  |            44.7137  |             51.1088  | improvement_ci_excludes_0 |
|       0.25 | 20260528 |                           70.7618  |            67.1201  |             74.4735  | improvement_ci_excludes_0 |
|       0.25 | 20260529 |                           70.1871  |            66.658   |             73.8327  | improvement_ci_excludes_0 |
|       0.25 | 20260530 |                           68.2538  |            64.7081  |             71.8476  | improvement_ci_excludes_0 |
|       0.5  | 20260528 |                           79.6056  |            75.8405  |             83.4417  | improvement_ci_excludes_0 |
|       0.5  | 20260529 |                           79.6171  |            75.8525  |             83.3933  | improvement_ci_excludes_0 |
|       0.5  | 20260530 |                           79.3105  |            75.5274  |             83.1379  | improvement_ci_excludes_0 |
|       1    | 20260528 |                           86.558   |            82.6388  |             90.4665  | improvement_ci_excludes_0 |
|       1    | 20260529 |                           85.8001  |            81.9924  |             89.8624  | improvement_ci_excludes_0 |
|       1    | 20260530 |                           86.1927  |            82.2993  |             90.1777  | improvement_ci_excludes_0 |

## Latent-Feature Baseline Availability

| baseline                      | availability                          | path   | decision                                                                       |
|:------------------------------|:--------------------------------------|:-------|:-------------------------------------------------------------------------------|
| latent_feature_delta_learning | not_available_from_local_r2scan_cache |        | not run; no r2SCAN split-aligned latent representation cache was found locally |
