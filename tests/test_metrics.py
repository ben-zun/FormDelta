from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]


def test_megnet_main_mae_values():
    with (ROOT / 'results/megnet/predictions/megnet_main_test_predictions.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    base = sum(float(r['abs_error_base_mev_atom']) for r in rows) / len(rows)
    formdelta = sum(float(r['abs_error_formdelta_mev_atom']) for r in rows) / len(rows)
    assert abs(base - 83.3721048213711) < 1e-6
    assert abs(formdelta - 62.0899488065882) < 1e-6
