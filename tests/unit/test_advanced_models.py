"""Tests for advanced model architecture helpers."""

import torch

from pinnkit.models import (
    AttentionPINN,
    EnsemblePINN,
    FourierFeaturePINN,
    MultiScalePINN,
    MLP,
    SineLayer,
    SirenPINN,
)


def test_multiscale_forward():
    """Multi-scale models preserve batch and output shape."""
    model = MultiScalePINN(
        in_features=2,
        scales=(1.0, 2.0),
        base_network_factory=lambda: MLP(in_dim=2, hidden_layers=1, width=4, out_dim=1),
    )
    x = torch.randn(5, 2)
    out = model(x)
    assert out.shape == (5, 1)


def test_fourier_forward():
    """Fourier-feature models map embedded coordinates to predictions."""
    model = FourierFeaturePINN(
        in_features=2,
        mapping_size=4,
        base_network=lambda **kw: MLP(hidden_layers=1, width=8, out_dim=1, **kw),
    )
    x = torch.randn(3, 2)
    out = model(x)
    assert out.shape == (3, 1)


def test_attention_forward():
    """Attention models preserve the expected scalar-output shape."""
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
    """Ensemble prediction returns mean and standard deviation tensors."""
    nets = [MLP(in_dim=2, hidden_layers=1, width=4, out_dim=1) for _ in range(3)]
    model = EnsemblePINN(nets)
    x = torch.randn(4, 2)
    mean, std = model.predict_with_uncertainty(x)
    assert mean.shape == std.shape == (4, 1)


def test_siren_forward_and_coordinate_derivative():
    """SIREN models are differentiable with respect to input coordinates."""
    model = SirenPINN(in_dim=2, hidden_layers=2, width=16, out_dim=1)
    x = torch.randn(5, 2, requires_grad=True)

    out = model(x)
    grad = torch.autograd.grad(out.sum(), x, create_graph=True)[0]

    assert out.shape == (5, 1)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()


def test_siren_initialization_respects_layer_bounds():
    """SIREN layer initialization uses first and hidden frequency bounds."""
    first = SineLayer(4, 8, omega_0=30.0, is_first=True)
    hidden = SineLayer(8, 8, omega_0=10.0)

    assert first.linear.weight.abs().max().item() <= 1.0 / 4.0
    assert hidden.linear.weight.abs().max().item() <= (6.0 / 8.0) ** 0.5 / 10.0


def test_siren_rejects_invalid_frequency():
    """SIREN frequency factors must be positive."""
    try:
        SirenPINN(omega_0=0.0)
    except ValueError as exc:
        assert "omega_0 must be positive" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("SirenPINN accepted an invalid frequency")
