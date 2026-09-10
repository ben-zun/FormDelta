# Direct-Reference Single-Model FE-GX

Generated: 2026-09-03

Pipeline:

1. frozen single-model prediction z(x);
2. fixed direct reference R(x)=z(x);
3. simple residual adapter on target data;
4. single-model FE-GX residual training with the reference path fixed.

## Test Summary

| stage | variant | test MAE (meV/atom) | delta vs direct |
|---|---|---:|---:|
| single_model_fegx | direct_reference_fegx_single | 62.090 | -21.282 |
| simple_residual_adapter | direct_reference_simple_residual | 75.271 | -8.101 |
| adapter_baseline | direct_reference_simple_residual_baseline | 83.372 | 0.000 |
| direct_raw | raw_model_energy_as_formation | 83.372 | 0.000 |
| direct_reference_contract | direct_reference | 83.372 | 0.000 |
| fegx_baseline | direct_reference_fegx_single_baseline | 83.372 | 0.000 |
