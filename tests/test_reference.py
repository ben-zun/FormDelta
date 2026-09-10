from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_docs_exist():
    for rel in ['README.md', 'DATA_AVAILABILITY.md', 'CODE_AVAILABILITY.md', 'paper/result_mapping.md']:
        assert (ROOT / rel).exists()
