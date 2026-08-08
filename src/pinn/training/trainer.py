"""Equation-agnostic trainer for composable PINN problems."""

from __future__ import annotations

import random
import time
from typing import Dict, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from ..problems import PDEProblem
from ..residuals import AutogradDerivativeBackend, DerivativeBackend, StrongFormResidual
from ..scaling import ScaledModel
from .causal_weighting import causal_residual_loss
from .config import TrainerConfig, resolve_dtype
from .state import TrainingResult, TrainingState


class Trainer:
    """Generic trainer for ``PDEProblem`` instances.

    The trainer knows how to combine residual and constraint losses, but it does
    not know which equation is being solved. Equation-specific work belongs to
    the problem, residual form and constraints.
    """

    def __init__(
        self,
        config: TrainerConfig | None = None,
        *,
        derivative_backend: DerivativeBackend | None = None,
        residual_form: StrongFormResidual | None = None,
    ) -> None:
        """Create a trainer with optional custom residual and derivative logic."""
        self.config = config or TrainerConfig()
        self.config.validate()
        self.derivative_backend = derivative_backend or AutogradDerivativeBackend()
        self.residual_form = residual_form or StrongFormResidual()

    def _seed(self, device: torch.device) -> torch.Generator:
        """Seed random number generators and return a torch generator."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.config.seed)
        return generator

    def _weight(self, name: str) -> float:
        """Return a configured loss weight, defaulting to one."""
        return float(self.config.loss_weights.get(name, 1.0))

    def _loss_components(
        self,
        problem: PDEProblem,
        model: nn.Module,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """Evaluate residual and constraint losses."""
        coordinates = problem.geometry.sample_interior(
            self.config.collocation_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        coordinates = coordinates.detach().requires_grad_(True)
        residual = self.residual_form.evaluate(
            problem, model, coordinates, self.derivative_backend
        )
        if self.config.causal is not None and self.config.causal.enabled:
            tmin = float(problem.geometry.lower_bounds[0])
            tmax = float(problem.geometry.upper_bounds[0])
            pde_loss, weights = causal_residual_loss(
                residual, coordinates[:, [0]], self.config.causal, tmin, tmax
            )
            components: Dict[str, Tensor] = {
                "pde": pde_loss,
                "min_causal_weight": weights.min().to(device=device, dtype=dtype),
            }
        else:
            components = {"pde": torch.mean(residual**2)}

        for constraint in problem.constraints():
            count = self.config.constraint_sample_counts.get(
                constraint.name, constraint.default_sample_count
            )
            loss = constraint.loss(
                model,
                problem.geometry,
                self.derivative_backend,
                n_samples=count,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            components[loss.name] = loss.value
        return components

    def _total_loss(self, components: Mapping[str, Tensor]) -> Tensor:
        """Combine named loss components with configured weights."""
        total: Tensor | None = None
        for name, value in components.items():
            if name.startswith("min_"):
                continue
            weighted = self._weight(name) * value
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("no trainable loss components were produced")
        return total

    def _record_state(
        self,
        step: int,
        start_time: float,
        total: Tensor,
        components: Mapping[str, Tensor],
    ) -> TrainingState:
        """Convert tensor losses to an immutable state record."""
        named_losses = {
            name: float(value.detach())
            for name, value in components.items()
            if not name.startswith("min_")
        }
        diagnostics = {
            name: float(value.detach())
            for name, value in components.items()
            if name.startswith("min_")
        }
        weights = {name: self._weight(name) for name in named_losses}
        return TrainingState(
            step=step,
            total_loss=float(total.detach()),
            named_losses=named_losses,
            weights=weights,
            elapsed_time=time.perf_counter() - start_time,
            diagnostics=diagnostics,
        )

    def train(self, problem: PDEProblem, model: nn.Module) -> TrainingResult:
        """Train ``model`` on ``problem`` and return recorded state history."""
        device = torch.device(self.config.device)
        dtype = resolve_dtype(self.config.dtype)
        if problem.input_dim < 1 or problem.output_dim < 1:
            raise ValueError("problem input_dim and output_dim must be positive")
        model = model.to(device=device, dtype=dtype)
        loss_model: nn.Module
        if self.config.scaling is not None:
            loss_model = ScaledModel(model, self.config.scaling)
        else:
            loss_model = model
        generator = self._seed(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.optimizer.lr)
        history: list[TrainingState] = []
        start_time = time.perf_counter()

        for step in range(1, self.config.optimizer.adam_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            components = self._loss_components(
                problem, loss_model, generator, device, dtype
            )
            total = self._total_loss(components)
            total.backward()
            optimizer.step()
            if (
                step == 1
                or step % self.config.log_every == 0
                or step == self.config.optimizer.adam_steps
            ):
                history.append(self._record_state(step, start_time, total, components))

        if self.config.optimizer.lbfgs_max_iter > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                max_iter=self.config.optimizer.lbfgs_max_iter,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                lbfgs.zero_grad(set_to_none=True)
                closure_components = self._loss_components(
                    problem, loss_model, generator, device, dtype
                )
                closure_total = self._total_loss(closure_components)
                closure_total.backward()
                return closure_total

            lbfgs.step(closure)
            components = self._loss_components(
                problem, loss_model, generator, device, dtype
            )
            total = self._total_loss(components)
            final_step = (
                self.config.optimizer.adam_steps + self.config.optimizer.lbfgs_max_iter
            )
            history.append(
                self._record_state(final_step, start_time, total, components)
            )

        if not history:
            components = self._loss_components(
                problem, loss_model, generator, device, dtype
            )
            total = self._total_loss(components)
            history.append(self._record_state(0, start_time, total, components))

        return TrainingResult(history=tuple(history), final_state=history[-1])
