import torch
from torch import nn

from pinnlab.utils.checkpointing import CheckpointManager


def test_checkpoint_save_load(tmp_path, device):
    model = nn.Linear(2, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CheckpointManager(tmp_path, save_frequency=1, keep_best_n=2)
    mgr.save(model, opt, step=5, metrics={"loss": 0.5}, config={"foo": "bar"})

    model2 = nn.Linear(2, 1).to(device)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    _, _, step, meta = mgr.load_latest(model2, opt2)

    assert step == 5
    assert meta.metrics["loss"] == 0.5
    assert meta.config["foo"] == "bar"


def test_checkpoint_prune(tmp_path, device):
    model = nn.Linear(1, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CheckpointManager(
        tmp_path, save_frequency=1, keep_best_n=2, metric_key="loss"
    )
    for i in range(5):
        mgr.save(model, opt, step=i, metrics={"loss": float(5 - i)})
    json_files = list(tmp_path.glob("checkpoint_*.json"))
    assert len(json_files) == 2


def test_early_stopping(tmp_path, device):
    model = nn.Linear(1, 1).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    mgr = CheckpointManager(tmp_path, save_frequency=1, keep_best_n=1, patience=2)
    for step in range(1, 5):
        mgr.maybe_save(model, opt, step, {"loss": 1.0})
        if mgr.should_stop():
            break
    assert mgr.should_stop()
