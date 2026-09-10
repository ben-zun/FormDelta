from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]


def count_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return sum(1 for _ in csv.DictReader(handle))


def test_megnet_split_counts():
    assert count_rows(ROOT / 'data/splits/megnet_medium_same_split_269/train/train.csv') == 8192
    assert count_rows(ROOT / 'data/splits/megnet_medium_same_split_269/val/val.csv') == 1024
    assert count_rows(ROOT / 'data/splits/megnet_medium_same_split_269/test/test.csv') == 2048


def test_r2scan_fraction_manifest_exists():
    assert (ROOT / 'data/splits/r2scan/fraction_split_manifest.csv').exists()
