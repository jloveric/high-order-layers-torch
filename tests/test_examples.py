from examples.invariant_mnist import invariant_mnist
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
import hydra


def test_invariant_mnist():
    cfg = DictConfig(
        content={
            "max_epochs": 1,
            "gpus": 0,
            "n": 2,
            "batch_size": 64,
            "segments": 2,
            "layer_type": "polynomial",
            "train_fraction": 0.001,
            "val_fraction": 0.0001,
            "periodicity": 2.0,
            "p_refine": False,
            "target_n": 5,
            "gradient_clip_val": 0,
            "optimizer": "adam",  # adam,adahessian
            "linear_part": 1.0,
        }
    )
    result = invariant_mnist(cfg=cfg)
    assert result[0]["test_acc"] is not None
    assert result[0]["test_loss"] is not None
