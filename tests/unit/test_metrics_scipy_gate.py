"""Tests for the SciPy-gated statistics in :mod:`pinnlab.utils.metrics`.

``metrics.py`` imported ``wasserstein_distance`` from ``scipy.spatial.distance``,
where it has never lived -- it is in ``scipy.stats``. The import therefore raised
``ImportError`` on every SciPy version, the ``except ImportError`` branch set
``SCIPY_AVAILABLE = False``, and the module warned that SciPy was unavailable
while it was installed and importable. Every metric behind that flag returned
``nan``.

Nothing caught it because the failure mode was indistinguishable from the
supported one: a user without SciPy sees exactly the same warning and the same
``nan``. Only someone who had installed SciPy *and* noticed the warning
contradicting that would look.

These tests are skipped when SciPy genuinely is absent, so they assert the
distinction the old code could not make.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinnlab.utils.metrics import SCIPY_AVAILABLE, StatisticalAnalysis

scipy = pytest.importorskip("scipy", reason="these tests are about SciPy being present")


class TestScipyIsDetectedWhenInstalled:
    def test_availability_flag_matches_reality(self):
        """The regression: importable SciPy must mean ``SCIPY_AVAILABLE``.

        This fails if the probing imports at the top of ``metrics`` drift back
        to a name that does not exist, whatever the reason.
        """
        assert (
            SCIPY_AVAILABLE
        ), "SciPy imports fine here, so metrics must not report it as missing"

    def test_wasserstein_distance_imports_from_the_module_that_has_it(self):
        """Pin the location, since importing it from the wrong one is silent."""
        from scipy.stats import wasserstein_distance

        assert callable(wasserstein_distance)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="requires SciPy")
class TestDistributionDistancesReturnNumbers:
    """Each method returned ``nan`` for years because of the bad import."""

    @staticmethod
    def _samples():
        rng = np.random.RandomState(0)
        return rng.normal(size=400), np.random.RandomState(1).normal(loc=0.5, size=400)

    @pytest.mark.parametrize("method", ["wasserstein", "ks", "anderson"])
    def test_distance_is_finite_and_non_negative(self, method: str):
        a, b = self._samples()
        distance = StatisticalAnalysis().compute_distribution_distance(
            a, b, method=method
        )
        assert np.isfinite(distance), f"{method} returned {distance}"
        assert distance >= 0.0

    def test_identical_samples_are_closer_than_shifted_ones(self):
        """A negative control: the metric has to respond to the data.

        Returning a finite constant would satisfy the test above.
        """
        a, b = self._samples()
        analysis = StatisticalAnalysis()
        same = analysis.compute_distribution_distance(a, a, method="wasserstein")
        shifted = analysis.compute_distribution_distance(a, b, method="wasserstein")
        assert same == pytest.approx(0.0, abs=1e-12)
        assert shifted > same

    def test_unknown_method_is_rejected(self):
        a, b = self._samples()
        with pytest.raises(ValueError, match="Unknown method"):
            StatisticalAnalysis().compute_distribution_distance(a, b, method="nope")
