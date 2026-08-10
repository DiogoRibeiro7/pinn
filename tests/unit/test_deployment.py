import torch
import pytest

try:  # pragma: no cover - optional dependency
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None  # type: ignore
    pytest.skip("fastapi not installed", allow_module_level=True)

from pinnlab.deployment import PINNModelServer, create_app, save_model, load_model


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.fill_(0.0)

    def forward(self, x):
        return self.linear(x)


def test_save_load_and_predict(tmp_path):
    path = tmp_path / "model"
    model = TinyModel()
    save_model(model, str(path))
    loaded = load_model(TinyModel, str(path))
    server = PINNModelServer(model=loaded)
    app = create_app(server)
    client = TestClient(app)
    resp = client.post("/predict", json={"coordinates": [[1.0, 2.0]]})
    assert resp.status_code == 200
    assert resp.json()["predictions"] == [[3.0]]
    health = client.get("/health").json()
    assert health["model_loaded"]
