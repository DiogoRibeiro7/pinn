import torch
from torch import nn

from pinnkit.distributed import (
    AsyncGradientBuffer,
    DistributedPINNTrainer,
    DistributedTrainer,
    DynamicLoadBalancer,
    GradientCompression,
    PipelineParallelEngine,
)
from pinnkit.utils.checkpointing import CheckpointManager


def test_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = nn.Linear(1, 1)
    trainer = DistributedTrainer(model)
    assert trainer.primary_device.type == "cpu"
    assert trainer.devices == ["cpu"]


def test_fit_passes_distributed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class Dummy:
        def __init__(self):
            self.called = False
            self.model = nn.Linear(1, 1)

        def train(self, *args, **kwargs):
            self.called = True
            assert "distributed" in kwargs

    solver = Dummy()
    trainer = DistributedTrainer(solver)
    trainer.fit(None)
    assert solver.called


def test_gradient_compression_roundtrip():
    grads = {"w": torch.randn(4, 4)}
    compressor = GradientCompression(compression_ratio=0.0)
    compressed = compressor.compress(grads)
    restored = compressor.decompress(compressed)
    assert torch.allclose(restored["w"], grads["w"])


def test_async_buffer_roundtrip():
    buffer = AsyncGradientBuffer()
    payload = {"w": torch.ones(3)}
    buffer.push(payload)
    popped = buffer.pop_all()
    assert len(popped) == 1
    assert torch.equal(popped[0]["w"], payload["w"])  # type: ignore[index]


def test_dynamic_load_balancer_prefers_fast_worker():
    balancer = DynamicLoadBalancer(["w0", "w1"])
    balancer.record_batch("w0", batch_size=16, duration=1.0)
    balancer.record_batch("w1", batch_size=16, duration=2.0)
    allocation = balancer.allocate(10)
    assert allocation["w0"] >= allocation["w1"]


def test_pipeline_engine_handles_microbatches():
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    engine = PipelineParallelEngine(model, devices=["cpu", "cpu"], microbatches=2)
    wrapped = engine.prepare()
    x = torch.randn(4, 2)
    out = wrapped(x)
    assert out.shape[0] == x.shape[0]


def test_distributed_pinn_trainer_scaling(tmp_path):
    class DummySolver:
        def __init__(self) -> None:
            self.model = nn.Linear(2, 1)
            self.train_called = False

        def train(self, *args, distributed=None, **kwargs):
            self.train_called = True
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            optimizer.zero_grad()
            loss = self.model(torch.zeros(4, 2)).sum()
            loss.backward()
            if distributed is not None:
                distributed.synchronize_gradients(self.model)
            optimizer.step()
            return loss

    solver = DummySolver()
    trainer = DistributedPINNTrainer(solver, devices="auto")
    trainer.fit(None)
    assert solver.train_called
    summary = trainer.scaling_summary(max(trainer.average_step_time(), 1e-6))
    assert "efficiency" in summary
    trainer.record_batch(8, 0.5)
    plan = trainer.plan_batches(8)
    assert sum(plan.values()) == 8

    optimizer = torch.optim.SGD(solver.model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path.as_posix(), save_frequency=1)
    manager.save(solver.model, optimizer, step=5, metrics={"loss": 1.0})
    trainer.failure_tracker.threshold = 1
    step = trainer.recover_from_failure(trainer.worker_id, manager)
    assert step == 5
