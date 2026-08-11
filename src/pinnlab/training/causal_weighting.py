"""Causal (temporal) weighting for time-dependent PINNs.

A standard PINN minimises the PDE residual over the whole space-time domain at
once, which lets it fit late times before it has fitted early ones. The result
is a network that satisfies the residual everywhere while representing a
solution that never followed from the initial condition -- a failure mode that
looks like convergence.

Causal weighting, from Wang, Sankaran and Perdikaris (2022), "Respecting
causality is all you need for training physics-informed neural networks"
(arXiv:2203.07404), restores the ordering. The time interval is split into
``n_chunks`` consecutive windows; window *i* is weighted by

    w_i = exp(-eps * sum_{k < i} L_k)

where ``L_k`` is the mean residual on window *k*. A window is therefore only
weighted appreciably once every earlier window is already well satisfied, so the
network is pushed to solve the problem forward in time. The weights are treated
as constants -- gradients do not flow through them -- so this reweights the loss
without changing what a converged solution is.

``eps`` controls the strictness: as ``eps -> 0`` the scheme degenerates to the
unweighted loss, and larger values enforce a sharper front.

A note on what has actually been measured here
----------------------------------------------
The weighting itself is verified against the formula above in
``tests/unit/test_exact_solvers.py``, including the closed-form weights for a
uniform residual. What is *not* established is an accuracy benefit on the
problems currently in this package.

On the 1-D wave equation over a single period, one run with weighting reached
4.3e-2 against 4.0e-2 without. Do not read that as a result: run-to-run spread
on these problems has since been measured at 2.2x to 2.4x, so a 7% difference
between single runs carries no information at all. Over three periods every
configuration tried landed between 0.85 and 0.88, which is failure across the
board for all of them -- the network was under-trained at that horizon rather
than mis-ordered in time, and that comparison says nothing about the weighting
either.

Heat and wave were the wrong problems to look for the effect on. Demonstrating
it needs one where the causality violation actually bites.

Allen-Cahn is that problem, and the effect is now measured
------------------------------------------------------------
:class:`~pinnlab.problems.AllenCahnProblem` has a trivial solution: ``u = 0``
satisfies the PDE and both periodic constraints exactly, so only the initial
condition objects to it. Plain Adam falls into it -- it fits the initial
condition and decays to zero for ``t > 0``. That is precisely a causality
failure, since the network is free to ignore what the early times imply about
the late ones.

At ``nu = 1e-2``, three seeds each, identical budgets (1500 Adam steps plus 200
L-BFGS, 2000 collocation points, 4x64 MLP), scored as relative L2 against the
spectral reference:

===========  ======================  ===============
Setting      Median relative L2      Range
===========  ======================  ===============
unweighted   0.917                   0.913 - 0.922
``eps=1``,   0.615                   0.600 - 0.688
16 chunks
===========  ======================  ===============

The ranges are disjoint across seeds, which is the bar the wave-equation
comparison above failed to clear. The mechanism is visible in the solution
magnitude rather than only in the score: mean ``|u|`` over the domain was
``0.07`` unweighted against ``0.32`` to ``0.44`` weighted, where the reference
is ``0.578``. The weighting is escaping the trivial solution, which is what it
is supposed to do.

Read that as "the method works and this is where it bites", not as "Allen-Cahn
is solved". At 0.6 relative error the weighted runs are still far from
accurate; the budget is small and the architecture untuned. Treat ``eps`` and
``n_chunks`` as parameters to measure on your own problem rather than a setting
that improves things by being switched on. See
``examples/advanced/causal_weighting.py``, which reports both errors rather than
assuming the weighted one wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CausalWeightConfig",
    "temporal_chunk_index",
    "causal_weights",
    "causal_residual_loss",
]


@dataclass(frozen=True)
class CausalWeightConfig:
    """Settings for causal residual weighting.

    Attributes:
        enabled: Whether to apply causal weighting at all.
        n_chunks: Number of consecutive time windows the interval is split into.
        eps: Causality parameter. Larger values enforce the temporal ordering
            more strictly; as it approaches zero the loss becomes the ordinary
            unweighted mean.
        tol: Convergence criterion. Training on the whole interval can be
            considered complete once the smallest weight exceeds this value,
            meaning even the last window is being optimised.
    """

    enabled: bool = True
    n_chunks: int = 16
    eps: float = 1.0
    tol: float = 0.99

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is unusable."""
        if self.n_chunks < 1:
            raise ValueError(f"n_chunks must be >= 1, got {self.n_chunks}")
        if self.eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {self.eps}")
        if not 0.0 < self.tol <= 1.0:
            raise ValueError(f"tol must lie in (0, 1], got {self.tol}")


def temporal_chunk_index(t: Tensor, n_chunks: int, tmin: float, tmax: float) -> Tensor:
    """Map each time coordinate to the index of its time window.

    Args:
        t: Time coordinates, shape ``(N,)`` or ``(N, 1)``.
        n_chunks: Number of windows spanning ``[tmin, tmax]``.
        tmin: Start of the time interval.
        tmax: End of the time interval.

    Returns:
        Long tensor of shape ``(N,)`` with values in ``[0, n_chunks)``.

    Raises:
        ValueError: If ``n_chunks`` is not positive or the interval is empty.
    """
    if n_chunks < 1:
        raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
    if not tmax > tmin:
        raise ValueError(f"tmax must exceed tmin, got tmin={tmin}, tmax={tmax}")

    flat = t.reshape(-1)
    normalised = (flat - tmin) / (tmax - tmin)
    idx = torch.floor(normalised * n_chunks).long()
    # t == tmax lands on n_chunks; fold it back into the final window.
    return idx.clamp_(0, n_chunks - 1)


def causal_weights(chunk_losses: Tensor, eps: float) -> Tensor:
    """Compute the causal weight of each time window.

    Args:
        chunk_losses: Mean residual per window, shape ``(M,)``, ordered by time.
        eps: Causality parameter.

    Returns:
        Weights of shape ``(M,)``, detached from the graph. The first window
        always has weight 1, and each later one decays with the accumulated
        residual of everything before it.
    """
    detached = chunk_losses.detach()
    # Exclusive cumulative sum: window i sees the losses of windows < i only.
    exclusive_cumsum = torch.cumsum(detached, dim=0) - detached
    return torch.exp(-eps * exclusive_cumsum)


def causal_residual_loss(
    residual: Tensor,
    t: Tensor,
    config: CausalWeightConfig,
    tmin: float,
    tmax: float,
) -> Tuple[Tensor, Tensor]:
    """Reduce a pointwise PDE residual to a causally weighted scalar loss.

    Args:
        residual: Pointwise residual, shape ``(N,)`` or ``(N, 1)``.
        t: Matching time coordinates, shape ``(N,)`` or ``(N, 1)``.
        config: Causal weighting settings.
        tmin: Start of the time interval.
        tmax: End of the time interval.

    Returns:
        A tuple of the scalar loss and the per-window weights. When weighting is
        disabled the loss is the plain mean squared residual and the returned
        weights are all ones.

    Raises:
        ValueError: If the configuration is invalid or the shapes disagree.
    """
    config.validate()

    squared = residual.reshape(-1) ** 2
    times = t.reshape(-1)
    if squared.shape != times.shape:
        raise ValueError(
            f"residual and t must describe the same points, got {tuple(squared.shape)} "
            f"and {tuple(times.shape)}"
        )

    if not config.enabled:
        return squared.mean(), torch.ones(1, device=residual.device)

    idx = temporal_chunk_index(times, config.n_chunks, tmin, tmax)

    # Mean residual per window, computed without a Python loop so it stays cheap.
    totals = torch.zeros(config.n_chunks, device=squared.device, dtype=squared.dtype)
    counts = torch.zeros_like(totals)
    totals = totals.index_add(0, idx, squared)
    counts = counts.index_add(0, idx, torch.ones_like(squared))
    # An empty window contributes nothing and must not produce a NaN.
    occupied = counts > 0
    chunk_losses = torch.where(occupied, totals / counts.clamp(min=1.0), totals)

    weights = causal_weights(chunk_losses, config.eps)
    # Average over occupied windows only, so the loss does not shrink simply
    # because some windows drew no collocation points.
    n_occupied = occupied.sum().clamp(min=1)
    loss = (weights * chunk_losses).sum() / n_occupied
    return loss, weights
