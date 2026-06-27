"""Deployment helpers for serving PINN models.

FastAPI and gRPC dependencies are optional to keep the core package lightweight.
The objects are imported lazily so that unit tests can run without the optional
dependencies being installed.
"""
from .model_io import load_model, save_model

try:  # pragma: no cover - optional dependency
    from .server import PINNModelServer, create_app
except Exception:  # pragma: no cover - dependency missing
    PINNModelServer = None  # type: ignore
    create_app = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .grpc_server import serve_grpc
except Exception:  # pragma: no cover - dependency missing
    serve_grpc = None  # type: ignore

__all__ = [
    "load_model",
    "save_model",
    "PINNModelServer",
    "create_app",
    "serve_grpc",
]
