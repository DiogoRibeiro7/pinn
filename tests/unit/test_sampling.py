"""Tests for sampling utilities."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from pinn.sampling import latin_hypercube as shared_latin_hypercube
from pinn.solvers import navier_stokes, raissi, raissi_generic, raissi_improved
from pinn.solvers.raissi_improved import latin_hypercube


@given(
    n=st.integers(min_value=1, max_value=20),
    d=st.integers(min_value=1, max_value=5),
    low=st.floats(min_value=-1.0, max_value=0.0),
    high=st.floats(min_value=0.0, max_value=1.0),
)
def test_latin_hypercube_bounds(n: int, d: int, low: float, high: float) -> None:
    """Property-based test ensuring samples fall within the requested bounds."""
    assume(high > low)
    samples = latin_hypercube(n, d, low, high)
    assert samples.shape == (n, d)
    assert np.all(samples >= low)
    assert np.all(samples <= high)


def test_solver_latin_hypercube_names_share_sampling_implementation() -> None:
    """Legacy solver LHS imports resolve to the shared sampling helper."""
    assert raissi.latin_hypercube is shared_latin_hypercube
    assert raissi_improved.latin_hypercube is shared_latin_hypercube
    assert raissi_generic.latin_hypercube is shared_latin_hypercube
    assert navier_stokes.latin_hypercube is shared_latin_hypercube


def test_shared_latin_hypercube_rejects_invalid_dimensions() -> None:
    """Shared LHS helper validates the number of sample dimensions."""
    with pytest.raises(ValueError, match="dimension count"):
        shared_latin_hypercube(4, 0)
