"""Equation-agnostic trainer for composable PINN problems."""

from __future__ import annotations

import random
import time
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..problems import PDEProblem
from ..residuals import AutogradDerivativeBackend, DerivativeBackend, StrongFormResidual
from ..scaling import ScaledModel
from .adaptive_refinement import ResidualAdaptiveRefiner
from .callbacks import TrainerCallback, _notify
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
        extra_collocation_points: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """Evaluate residual and constraint losses."""
        coordinates = self.config.sampler.sample(
            problem.geometry,
            self.config.collocation_count,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        if extra_collocation_points is not None:
            coordinates = torch.cat(
                [coordinates, extra_collocation_points.to(device=device, dtype=dtype)],
                dim=0,
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

    def _evaluate(
        self,
        problem: PDEProblem,
        loss_model: nn.Module,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
        refiner: ResidualAdaptiveRefiner | None,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        """Sample, evaluate every loss component and combine them.

        Both optimizers and the no-step fallback need exactly this sequence, so
        it lives in one place rather than being repeated at each call site.
        """
        extra_points = (
            refiner.current_points(device=device, dtype=dtype)
            if refiner is not None
            else None
        )
        components = self._loss_components(
            problem,
            loss_model,
            generator,
            device,
            dtype,
            extra_collocation_points=extra_points,
        )
        return self._total_loss(components), components

    def _record_state(
        self,
        step: int,
        start_time: float,
        total: Tensor,
        components: Mapping[str, Tensor],
        extra_diagnostics: Mapping[str, float] | None = None,
        phase: str = "adam",
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
        if extra_diagnostics is not None:
            diagnostics.update(extra_diagnostics)
        weights = {name: self._weight(name) for name in named_losses}
        return TrainingState(
            step=step,
            total_loss=float(total.detach()),
            named_losses=named_losses,
            weights=weights,
            elapsed_time=time.perf_counter() - start_time,
            diagnostics=diagnostics,
            phase=phase,
        )

    def train(
        self,
        problem: PDEProblem,
        model: nn.Module,
        *,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> TrainingResult:
        """Train ``model`` on ``problem`` and return recorded state history.

        Args:
            problem: The PDE problem supplying geometry, residual and constraints.
            model: Network mapping coordinates to the solution field.
            callbacks: Optional observers notified as training progresses. They
                see each recorded state, each phase transition and the final
                result. Exceptions raised by a callback propagate.

        Returns:
            The recorded history and the final state.

        Raises:
            ValueError: If the problem's input or output dimension is not positive.
        """
        callbacks = list(callbacks or ())
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
        refiner = (
            ResidualAdaptiveRefiner(self.config.adaptive_refinement)
            if self.config.adaptive_refinement is not None
            else None
        )
        generator = self._seed(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.optimizer.lr)
        history: list[TrainingState] = []
        start_time = time.perf_counter()

        _notify(callbacks, "on_train_begin", problem, model)

        def record(
            step: int, total: Tensor, components: Mapping[str, Tensor], phase: str
        ) -> TrainingState:
            """Append a state to the history and notify callbacks."""
            state = self._record_state(
                step,
                start_time,
                total,
                components,
                refiner.diagnostics() if refiner is not None else None,
                phase=phase,
            )
            history.append(state)
            _notify(callbacks, "on_state_recorded", state)
            return state

        for step in range(1, self.config.optimizer.adam_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if refiner is not None and refiner.should_refine(step):
                refiner.refine(
                    problem,
                    loss_model,
                    self.residual_form,
                    self.derivative_backend,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            total, components = self._evaluate(
                problem, loss_model, generator, device, dtype, refiner
            )
            total.backward()
            optimizer.step()
            if (
                step == 1
                or step % self.config.log_every == 0
                or step == self.config.optimizer.adam_steps
            ):
                record(step, total, components, "adam")

        if history:
            _notify(callbacks, "on_phase_end", "adam", history[-1])

        if self.config.optimizer.lbfgs_max_iter > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                max_iter=self.config.optimizer.lbfgs_max_iter,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                lbfgs.zero_grad(set_to_none=True)
                closure_total, _ = self._evaluate(
                    problem, loss_model, generator, device, dtype, refiner
                )
                closure_total.backward()
                return closure_total

            lbfgs.step(closure)
            total, components = self._evaluate(
                problem, loss_model, generator, device, dtype, refiner
            )
            final_step = (
                self.config.optimizer.adam_steps + self.config.optimizer.lbfgs_max_iter
            )
            lbfgs_state = record(final_step, total, components, "lbfgs")
            _notify(callbacks, "on_phase_end", "lbfgs", lbfgs_state)

        if not history:
            # No optimizer steps were requested; still report where the model
            # starts so callers always get a usable state.
            total, components = self._evaluate(
                problem, loss_model, generator, device, dtype, refiner
            )
            record(0, total, components, "initial")

        result = TrainingResult(history=tuple(history), final_state=history[-1])
        _notify(callbacks, "on_train_end", result)
        return result
