import numpy as np
import torch
import pytest
from dataclasses import dataclass

from pinn.utils.validation import (
    ValidationError,
    validate_arrays,
    validate_config,
    validate_device,
)


@dataclass
class DummyConfig:
    a: int


@validate_arrays(["x"], shapes={"x": (-1, 2)})
def uses_array(x):
    return x


def test_validate_arrays_fail():
    with pytest.raises(ValidationError):
        uses_array(np.zeros((3, 3)))


def test_validate_arrays_pass():
    arr = np.zeros((4, 2))
    assert uses_array(arr).shape == (4, 2)


@validate_config(DummyConfig)
def uses_config(config):
    return config.a


def test_validate_config():
    cfg = DummyConfig(1)
    assert uses_config(cfg) == 1
    with pytest.raises(ValidationError):
        uses_config(object())


@validate_device(["t"], device_arg="device")
def uses_device(t, *, device="cpu"):
    return t


def test_validate_device():
    tensor = torch.zeros(2)
    uses_device(tensor, device="cpu")
    with pytest.raises(ValidationError):
        uses_device(tensor, device="cuda")
