import pytest
from omegaconf import DictConfig, OmegaConf

from examples.cifar10 import run_cifar10
from examples.cifar100 import cifar100
from examples.function_example import plot_results
from examples.invariant_mnist import invariant_mnist
from examples.mnist import mnist
from examples.xor import model_set_d, plot_approximation
from examples.autoencoder import autoencoder
import torch

torch.set_default_device(device="cpu")


@pytest.mark.parametrize("p_refine", [True, False])
def test_invariant_mnist(p_refine: bool):
    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "accelerator": "cpu",
            "batch_size": 64,
            "train_fraction": 0.0001,
            "val_fraction": 0.0001,
            "periodicity": 2.0,
            "gradient_clip_val": 0,
            "optimizer": {
                "name": "adam",
                "lr": 1e-4,
                "patience": 5,
                "factor": 0.1,
                "gamma": 0.9,
                "scheduler": "plateau",
            },
            "mlp": {
                "dropout": 0,
                "normalize": "max_abs",
                "segments": 2,
                "layer_type": "continuous",
                "p_refine": p_refine,
                "target_n": 3,
                "n": 2,
                "input": {"width": 784, "segments": 2},
                "hidden": {
                    "width": 10,
                    "segments": 2,
                    "layers": 1,
                },
                "output": {"width": 10, "segments": 2},
            },
        }
    )
    result = invariant_mnist(cfg=cfg)
    assert result[0]["test_acc"] is not None
    assert result[0]["test_loss"] is not None


"""
TODO: Fix this!
def test_cifar100():
    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "accelerator": "cpu",
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
"""


def test_mnist():
    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "accelerator": "cpu",
            "n": 3,
            "batch_size": 16,
            "segments": 2,
            "layer_type": "continuous2d",
            "train_fraction": 0.01,
            "add_pos": False,
            "output_layer_type": "linear",
            "channels": [1, 1],
            "double": False,
            "kernel_size": 5,
            "normalization": "max_abs",
            "optimizer": {
                "name": "sophia",
                "lr": 1e-4,
                "rho": 0.035,
            },
        }
    )
    result = mnist(cfg=cfg)
    assert result[0]["test_acc"] is not None
    assert result[0]["test_loss"] is not None


def test_xor():
    result = plot_approximation(model_set=model_set_d, segments=2, epochs=1, plot=False)
    print("result", result)
    assert len(result) > 0


def test_function_approximation():
    result = plot_results(epochs=1, segments=5, plot=False)


"""
def test_autoencoder() :

    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "accelerator": "cpu",
            "n": 3,
            "batch_size": 16,
            "segments": 2,
            "layer_type": "continuous2d",
            "train_fraction": 0.01,
            "add_pos": False,
        }
    )

    result = autoencoder(cfg=cfg)
"""

"""
def test_cifar10():
    run_cifar10(max_epochs=1, gpus=0, n=2, segments=2, layer_type="continuous")
"""
