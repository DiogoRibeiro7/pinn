"""Curriculum learning utilities."""

from __future__ import annotations

from typing import Callable, Iterable


def curriculum_train(
    train_fn: Callable[[Iterable], None], curricula: Iterable[Iterable]
) -> None:
    """Train a model over an increasing sequence of curricula.

    Parameters
    ----------
    train_fn:
        Callable that performs training given a dataset-like iterable.
    curricula:
        Sequence of datasets of increasing difficulty.
    """

    for data in curricula:
        train_fn(data)
