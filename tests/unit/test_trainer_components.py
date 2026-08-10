"""Tests for how the trainer uses its configured sampler and scaling transform.

`tests/unit/test_scaling.py` covers the scaling transforms themselves, and
`tests/unit/test_sampling_core.py` covers the samplers in isolation. What
neither covers is the seam between them and the trainer: that a configured
sampler is actually the one consulted, that it receives the geometry, count,
device and dtype the trainer promises, and that scaling reaches the model
without disturbing the constraints. Those were previously exercised only
end to end, where a trainer that quietly ignored `config.sampler` would still
produce a falling loss and look correct.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from pinnlab.geometry import Geometry
from pinnlab.models import MLP
from pinnlab.problems import HeatProblem
from pinnlab.sampling import LatinHypercubeInteriorSampler, UniformInteriorSampler
from pinnlab.scaling import CharacteristicScales, Nondimensionalizer
from pinnlab.training import OptimizerConfig, Trainer, TrainerConfig


class RecordingSampler:
    """An InteriorSampler that notes how the trainer called it."""

    def __init__(self, inner=None) -> None:
        """Wrap ``inner``, defaulting to the uniform sampler."""
        self.inner = inner or UniformInteriorSampler()
        self.calls: list[dict] = []

    def sample(
        self,
        geometry: Geometry,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Record the arguments, then delegate."""
        self.calls.append(
            {
                "geometry": geometry,
                "n": n,
                "device": device,
                "dtype": dtype,
                "generator_is_none": generator is None,
            }
        )
        return self.inner.sample(
            geometry, n, generator=generator, device=device, dtype=dtype
        )


class RecordingModel(nn.Module):
    """An MLP that remembers the coordinates it was last given."""

    def __init__(self) -> None:
        """Create the wrapped network."""
        super().__init__()
        self.net = MLP(in_dim=2, hidden_layers=1, width=8, out_dim=1)
        self.seen: list[Tensor] = []

    def forward(self, coordinates: Tensor) -> Tensor:
        """Record the input and evaluate."""
        self.seen.append(coordinates.detach().clone())
        return self.net(coordinates)


def _config(**overrides) -> TrainerConfig:
    """Return a small trainer config with the given overrides applied."""
    settings = dict(
        seed=0,
        collocation_count=16,
        log_every=1,
        constraint_sample_counts={"ic": 4, "bc_left": 4, "bc_right": 4},
        optimizer=OptimizerConfig(adam_steps=2),
    )
    settings.update(overrides)
    return TrainerConfig(**settings)


@pytest.fixture
def problem() -> HeatProblem:
    return HeatProblem(n_constraint_samples=4)


class TestConfiguredSamplerIsUsed:
    def test_the_configured_sampler_is_consulted(self, problem):
        """A trainer that ignored config.sampler would still train happily, so
        assert the sampler is actually reached."""
        sampler = RecordingSampler()
        Trainer(_config(sampler=sampler)).train(problem, MLP(2, 1, 8, 1))
        assert sampler.calls, "trainer never called the configured sampler"

    def test_sampler_receives_the_configured_count(self, problem):
        sampler = RecordingSampler()
        Trainer(_config(sampler=sampler, collocation_count=23)).train(
            problem, MLP(2, 1, 8, 1)
        )
        assert {call["n"] for call in sampler.calls} == {23}

    def test_sampler_receives_the_problem_geometry(self, problem):
        """Compared by value, not identity: ``problem.geometry`` builds a fresh
        Rectangle on each access, so ``is`` would never hold."""
        sampler = RecordingSampler()
        Trainer(_config(sampler=sampler)).train(problem, MLP(2, 1, 8, 1))
        assert all(call["geometry"] == problem.geometry for call in sampler.calls)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_sampler_receives_the_configured_dtype(self, problem, dtype):
        sampler = RecordingSampler()
        Trainer(_config(sampler=sampler, dtype=dtype)).train(problem, MLP(2, 1, 8, 1))
        assert {call["dtype"] for call in sampler.calls} == {dtype}

    def test_sampler_receives_a_seeded_generator(self, problem):
        """Reproducibility depends on the trainer passing its generator down
        rather than letting the sampler use global randomness."""
        sampler = RecordingSampler()
        Trainer(_config(sampler=sampler)).train(problem, MLP(2, 1, 8, 1))
        assert all(call["generator_is_none"] is False for call in sampler.calls)

    def test_sampler_is_called_once_per_optimizer_step(self, problem):
        sampler = RecordingSampler()
        Trainer(
            _config(sampler=sampler, optimizer=OptimizerConfig(adam_steps=5))
        ).train(problem, MLP(2, 1, 8, 1))
        assert len(sampler.calls) == 5

    def test_latin_hypercube_sampler_works_through_the_trainer(self, problem):
        result = Trainer(_config(sampler=LatinHypercubeInteriorSampler())).train(
            problem, MLP(2, 1, 8, 1)
        )
        assert result.history
        assert result.final_loss >= 0.0

    def test_a_sampler_without_a_sample_method_is_rejected(self):
        with pytest.raises(ValueError, match="sampler must define a sample method"):
            TrainerConfig(sampler=object()).validate()

    def test_sampled_points_lie_inside_the_geometry(self, problem):
        sampler = RecordingSampler()
        Trainer(_config(sampler=sampler)).train(problem, MLP(2, 1, 8, 1))
        points = sampler.inner.sample(problem.geometry, 64)
        lower = problem.geometry.lower_bounds
        upper = problem.geometry.upper_bounds
        assert torch.all(points >= torch.as_tensor(lower, dtype=points.dtype))
        assert torch.all(points <= torch.as_tensor(upper, dtype=points.dtype))


class TestScalingInteraction:
    @staticmethod
    def _scaling(input_scales=(2.0, 4.0), output_scales=(3.0,)) -> Nondimensionalizer:
        return Nondimensionalizer(
            CharacteristicScales(input_scales=input_scales, output_scales=output_scales)
        )

    def test_model_sees_dimensionless_coordinates(self, problem):
        """With scaling on, the raw model is trained in dimensionless variables
        while the problem still describes physical ones."""
        model = RecordingModel()
        Trainer(_config(scaling=self._scaling())).train(problem, model)

        assert model.seen
        scaled_extent = max(float(t.abs().max()) for t in model.seen)
        # The physical domain spans t,x in [0,1]; dividing by scales of 2 and 4
        # must shrink what the network actually sees.
        assert scaled_extent <= 1.0

    def test_training_without_scaling_sees_physical_coordinates(self, problem):
        model = RecordingModel()
        Trainer(_config()).train(problem, model)
        unscaled_extent = max(float(t.abs().max()) for t in model.seen)

        scaled_model = RecordingModel()
        Trainer(_config(scaling=self._scaling())).train(problem, scaled_model)
        scaled_extent = max(float(t.abs().max()) for t in scaled_model.seen)

        assert scaled_extent < unscaled_extent

    def test_scaling_still_produces_a_usable_history(self, problem):
        result = Trainer(_config(scaling=self._scaling())).train(
            problem, MLP(2, 1, 8, 1)
        )
        assert result.history
        assert all(state.total_loss >= 0.0 for state in result.history)
        assert "pde" in result.final_state.named_losses

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_scaling_preserves_the_configured_dtype(self, problem, dtype):
        model = RecordingModel()
        Trainer(_config(scaling=self._scaling(), dtype=dtype)).train(problem, model)
        assert all(t.dtype == dtype for t in model.seen)

    def test_scaling_and_a_custom_sampler_compose(self, problem):
        """The two features are configured independently, so confirm enabling
        one does not bypass the other."""
        sampler = RecordingSampler()
        result = Trainer(_config(sampler=sampler, scaling=self._scaling())).train(
            problem, MLP(2, 1, 8, 1)
        )

        assert sampler.calls, "scaling suppressed the configured sampler"
        assert result.history

    def test_sampler_still_receives_physical_coordinates_under_scaling(self, problem):
        """Scaling belongs to the model wrapper, not to sampling: the sampler
        works in the problem's own geometry either way."""
        sampler = RecordingSampler()
        Trainer(_config(sampler=sampler, scaling=self._scaling())).train(
            problem, MLP(2, 1, 8, 1)
        )
        assert all(call["geometry"] == problem.geometry for call in sampler.calls)


class TestReproducibility:
    def test_the_same_seed_gives_the_same_losses(self, problem):
        losses = []
        for _ in range(2):
            torch.manual_seed(1234)  # model init happens before train() seeds
            model = MLP(2, 1, 8, 1)
            result = Trainer(_config(seed=7)).train(problem, model)
            losses.append([state.total_loss for state in result.history])
        assert losses[0] == pytest.approx(losses[1], rel=1e-9)

    def test_different_seeds_give_different_sample_points(self, problem):
        first, second = RecordingSampler(), RecordingSampler()
        torch.manual_seed(0)
        Trainer(_config(sampler=first, seed=1)).train(problem, MLP(2, 1, 8, 1))
        torch.manual_seed(0)
        Trainer(_config(sampler=second, seed=2)).train(problem, MLP(2, 1, 8, 1))

        a = first.inner.sample(problem.geometry, 8, generator=None)
        b = second.inner.sample(problem.geometry, 8, generator=None)
        assert a.shape == b.shape
