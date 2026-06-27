import torch

from pinn.models import (
    AttentionPINN,
    EnsemblePINN,
    FourierFeaturePINN,
    MultiScalePINN,
    MLP,
)


def test_multiscale_forward():
    model = MultiScalePINN(
        in_features=2,
        scales=(1.0, 2.0),
        base_network_factory=lambda: MLP(in_dim=2, hidden_layers=1, width=4, out_dim=1),
    )
    x = torch.randn(5, 2)
    out = model(x)
    assert out.shape == (5, 1)


def test_fourier_forward():
    model = FourierFeaturePINN(
        in_features=2,
        mapping_size=4,
        base_network=lambda **kw: MLP(hidden_layers=1, width=8, out_dim=1, **kw),
    )
    x = torch.randn(3, 2)
    out = model(x)
    assert out.shape == (3, 1)


def test_attention_forward():
    model = AttentionPINN(
        in_features=2,
        embed_dim=8,
        num_heads=2,
        base_network=lambda **kw: MLP(hidden_layers=1, width=8, out_dim=1, **kw),
    )
    x = torch.randn(2, 2)
    out = model(x)
    assert out.shape == (2, 1)


def test_ensemble_predict_with_uncertainty():
    nets = [MLP(in_dim=2, hidden_layers=1, width=4, out_dim=1) for _ in range(3)]
    model = EnsemblePINN(nets)
    x = torch.randn(4, 2)
    mean, std = model.predict_with_uncertainty(x)
    assert mean.shape == std.shape == (4, 1)
