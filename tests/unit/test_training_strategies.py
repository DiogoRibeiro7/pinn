import torch

from pinnlab.training import (
    adversarial_training_step,
    adaptive_loss_weights,
    architecture_search,
    curriculum_train,
    gradient_hyperparam_search,
    meta_learning_step,
    multi_fidelity_train,
    second_order_step,
)
from pinnlab.models import MLP


def test_curriculum_train():
    calls = []

    def train_fn(data):
        calls.append(len(data))

    curriculum_train(train_fn, [[1], [1, 2]])
    assert calls == [1, 2]


def test_adversarial_training_step():
    model = MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1)
    loss_fn = torch.nn.MSELoss()
    x = torch.randn(2, 1)
    y = torch.randn(2, 1)
    adv = adversarial_training_step(model, loss_fn, x, y)
    assert adv.shape == (2, 1)


def test_meta_learning_step():
    model = MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1)
    tasks = [(torch.randn(1, 1), torch.randn(1, 1)) for _ in range(2)]
    meta_learning_step(model, tasks, inner_steps=1)


def test_multi_fidelity_train():
    model = MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1)
    high = torch.randn(3, 2)
    low = torch.randn(3, 2)
    loss_fn = torch.nn.MSELoss()
    loss = multi_fidelity_train(model, high, low, loss_fn)
    assert loss > 0


def test_second_order_step_and_optimisation():
    model = MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1)
    x = torch.randn(3, 1)
    y = torch.randn(3, 1)
    loss_fn = torch.nn.MSELoss()
    loss = second_order_step(model, loss_fn, x, y)
    assert loss > 0

    losses = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    weights = adaptive_loss_weights(losses)
    assert set(weights.keys()) == {"a", "b"}

    param = torch.nn.Parameter(torch.tensor(1.0))
    gradient_hyperparam_search(param, lambda p: (p - 1) ** 2)

    arch = architecture_search(
        [lambda: MLP(in_dim=1, hidden_layers=1, width=4, out_dim=1)], lambda m: 0.0
    )
    assert isinstance(arch, MLP)
