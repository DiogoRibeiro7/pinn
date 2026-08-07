import torch

from pinn.models import (
    FourierMultiScalePINN,
    HierarchicalPINN,
    MLP,
    MultiScalePINN,
    MultiScaleTrainer,
    MultiScaleTrainingConfig,
    WaveletPINN,
    multiscale_loss,
    scale_adaptive_sampling,
)


def test_multiscale_forward_and_uncertainty():
    model = MultiScalePINN(
        in_features=2,
        scales=(1.0, 2.0),
        base_network_factory=lambda: MLP(in_dim=2, hidden_layers=1, width=8, out_dim=1),
        combination_method="learned",
    )
    x = torch.randn(10, 2)
    prediction = model(x)
    assert prediction.shape == (10, 1)
    mean, std = model.predict_with_uncertainty(x)
    assert mean.shape == std.shape == (10, 1)
    stats = model.scale_statistics()
    assert set(stats) == {"scales", "usage"}


def test_multiscale_attention_weights_sum_to_one():
    model = MultiScalePINN(
        in_features=1,
        scales=(1.0, 3.0, 5.0),
        combination_method="attention",
        base_network_factory=lambda: MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1),
    )
    x = torch.randn(4, 1)
    _, _, weights = model.forward_with_details(x)
    assert torch.allclose(
        weights.sum(dim=1), torch.ones_like(weights.sum(dim=1)), atol=1e-5
    )


def test_multiscale_trainer_progressive_updates():
    torch.manual_seed(0)
    model = MultiScalePINN(
        in_features=1,
        scales=(1.0, 2.0),
        combination_method="weighted_average",
        base_network_factory=lambda: MLP(in_dim=1, hidden_layers=1, width=8, out_dim=1),
    )
    coords = torch.linspace(0, 1, 16).unsqueeze(1)
    targets = torch.sin(2 * torch.pi * coords) + 0.5 * torch.sin(6 * torch.pi * coords)
    trainer = MultiScaleTrainer(
        model,
        MultiScaleTrainingConfig(epochs_per_scale=1, lr=1e-2, consistency_weight=0.0),
    )
    state = trainer.train_progressive({"coordinates": coords, "targets": targets})
    # One loss entry per epoch/scale
    assert len(state.losses) == 2
    assert len(state.consistency) == 2


def test_multiscale_loss_reports_components():
    outputs = [torch.ones(3, 1), torch.zeros(3, 1)]
    pred = torch.ones(3, 1) * 0.5
    target = torch.zeros(3, 1)
    total, metrics = multiscale_loss(pred, target, outputs, consistency_weight=0.2)
    assert total.item() >= 0
    assert set(metrics) == {"mse", "consistency"}


def test_scale_adaptive_sampling_prefers_high_weights():
    base_points = torch.linspace(0, 1, 10).unsqueeze(1)

    def sampler(n: int) -> torch.Tensor:
        repeat = base_points.repeat((n // base_points.shape[0] + 1, 1))
        return repeat[:n]

    importance = torch.tensor([0.1, 0.9])
    samples = scale_adaptive_sampling(sampler, importance, 5)
    assert samples.shape == (5, 1)


def test_hierarchical_and_wavelet_models_forward():
    x = torch.randn(6, 2)
    hierarchical = HierarchicalPINN(in_features=2, levels=2, base_width=8)
    out = hierarchical(x)
    assert out.shape == (6, 1)
    all_outputs, final_out = hierarchical(x, return_all=True)
    assert isinstance(all_outputs, list)
    assert final_out.shape == (6, 1)

    wavelet = WaveletPINN(in_features=2, wavelet_levels=2)
    wavelet_out = wavelet(x)
    assert wavelet_out.shape == (6, 1)


def test_fourier_multiscale_inherits_behaviour():
    model = FourierMultiScalePINN(
        in_features=1, scales=(1.0, 2.0), combination_method="uniform"
    )
    x = torch.randn(3, 1)
    out = model(x)
    assert out.shape == (3, 1)


def test_adapt_scales_updates_parameters():
    model = MultiScalePINN(
        in_features=1,
        scales=(1.0, 2.0),
        learnable_scales=True,
        base_network_factory=lambda: MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1),
    )
    original = model.get_scales().clone()
    model.adapt_scales(torch.tensor([1.1, 0.9]))
    updated = model.get_scales()
    assert not torch.allclose(original, updated)
