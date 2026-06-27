"""Tests for sampling utilities."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, strategies as st

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

