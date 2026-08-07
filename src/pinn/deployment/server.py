"""FastAPI-based serving utilities for trained PINN models."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Sequence

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class PINNModelServer:
    """In-memory model server with basic caching and batching support."""

    def __init__(
        self,
        model: Optional[torch.jit.ScriptModule] = None,
        model_path: Optional[str] = None,
    ) -> None:
        if model is None and model_path is None:
            raise ValueError("Either model or model_path must be provided")
        if model is None:
            model = torch.jit.load(model_path)
        self.model = model.eval()
        self._device = next(self.model.parameters()).device

    @property
    def is_loaded(self) -> bool:  # pragma: no cover - simple property
        return self.model is not None

    @lru_cache(maxsize=128)
    def _cached_predict(self, key: tuple) -> torch.Tensor:
        arr = torch.tensor(key, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            out = self.model(arr)
        return out.cpu()

    def predict(self, coordinates: Sequence[Sequence[float]]) -> torch.Tensor:
        key = tuple(tuple(map(float, row)) for row in coordinates)
        return self._cached_predict(key)


class _PredictRequest(BaseModel):
    coordinates: List[List[float]]


def create_app(model_server: Optional[PINNModelServer] = None) -> FastAPI:
    """Create a FastAPI application serving ``model_server``."""
    if model_server is None:
        model_path = os.getenv("PINN_MODEL_PATH")
        if model_path is None:
            raise RuntimeError("No model server or PINN_MODEL_PATH provided")
        model_server = PINNModelServer(model_path=model_path)

    app = FastAPI()

    @app.post("/predict")
    async def predict(req: _PredictRequest):
        try:
            preds = model_server.predict(req.coordinates).tolist()
            return {"predictions": preds}
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy", "model_loaded": model_server.is_loaded}

    return app


# Application used by ``uvicorn pinn.deployment.server:app``
try:  # pragma: no cover - executed in deployment scenarios
    app = create_app()
except Exception:  # pragma: no cover - during import without model
    app = FastAPI()


def main() -> None:  # pragma: no cover - thin wrapper for uvicorn
    """Launch a Uvicorn server using the model path from ``PINN_MODEL_PATH``."""
    import uvicorn

    # A container has to bind every interface to be reachable from outside it.
    # Override with PINN_HOST (e.g. 127.0.0.1) when running directly on a host.
    host = os.getenv("PINN_HOST", "0.0.0.0")  # nosec B104
    uvicorn.run(app, host=host, port=int(os.getenv("PORT", "8000")))
