from examples.invariant_mnist import invariant_mnist
from examples.cifar100 import cifar100
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
import pytest


@pytest.mark.parametrize("p_refine", [True, False])
def test_invariant_mnist(p_refine: bool):
    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "gpus": 0,
            "n": 2,
            "batch_size": 64,
            "segments": 2,
            "layer_type": "continuous",
            "train_fraction": 0.0001,
            "val_fraction": 0.0001,
            "periodicity": 2.0,
            "p_refine": p_refine,
            "target_n": 3,
            "gradient_clip_val": 0,
            "optimizer": "adam",  # adam,adahessian
            "linear_part": 1.0,
        }
    )
    result = invariant_mnist(cfg=cfg)
    assert result[0]["test_acc"] is not None
    assert result[0]["test_loss"] is not None


@pytest.mark.parametrize("p_refine", [True, False])
def test_cifar100(p_refine: bool):
    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "gpus": 0,
            "n": 5,
            "batch_size": 128,
            "segments": 2,
            "layer_type": "continuous2d",
            "train_fraction": 0.001,
            "rescale_output": False,
            "linear_output": True,
            "periodicity": 2.0,
            "lr": 0.001,
            "nonlinearity": False,
        }
    )
    result = cifar100(cfg=cfg)
    assert result[0]["test_acc"] is not None
    assert result[0]["test_loss"] is not None
