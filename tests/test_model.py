import pytest


def test_residual_adapter_forward_shape():
    torch = pytest.importorskip('torch')
    from formdelta.adapter import ResidualAdapter

    model = ResidualAdapter(in_dim=4, hidden_dim=8, num_layers=3, dropout=0.0)
    out = model(torch.zeros(2, 4))
    assert tuple(out.shape) == (2,)
