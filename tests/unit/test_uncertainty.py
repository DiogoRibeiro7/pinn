import torch

from pinn.uncertainty import (
    BayesianPINN,
    deep_ensemble_predict,
    gp_prior_predict,
    mc_dropout_predict,
)
from pinn.models import MLP


def test_mc_dropout_predict():
    model = MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1)
    # add dropout
    model.net.insert(1, torch.nn.Dropout(p=0.5))
    x = torch.randn(3, 1)
    mean, std = mc_dropout_predict(model, x, passes=5)
    assert mean.shape == std.shape == (3, 1)


def test_deep_ensemble_predict():
    models = [MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1) for _ in range(3)]
    x = torch.randn(2, 1)
    mean, std = deep_ensemble_predict(models, x)
    assert mean.shape == std.shape == (2, 1)


def test_gp_prior_predict():
    def kernel(a, b):
        sqdist = (a - b.T) ** 2
        return torch.exp(-0.5 * sqdist)

    x_train = torch.tensor([[0.0], [1.0]])
    y_train = torch.tensor([[0.0], [1.0]])
    x_test = torch.tensor([[0.5]])
    mean, cov = gp_prior_predict(kernel, x_train, y_train, x_test)
    assert mean.shape == (1, 1)
    assert cov.shape == (1, 1)


def test_bayesian_pinn_forward():
    model = BayesianPINN(in_features=1, hidden=4, out_features=1)
    x = torch.randn(2, 1)
    out = model(x)
    assert out.shape == (2, 1)
