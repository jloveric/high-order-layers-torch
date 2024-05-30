import os

import hydra
import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, Trainer

from high_order_layers_torch.layers import *
from high_order_layers_torch.modules import ClassificationNet
from lion_pytorch import Lion

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class Cifar100DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        train_fraction: float = 0.9,
        num_workers: int = 1,
    ):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._train_fraction = train_fraction
        self._num_workers = num_workers

    def setup(self, stage):
        num_train = int(self._train_fraction * 50000)
        num_val = 50000 - num_train

        train = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=True, download=True, transform=transform
        )

        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train,
            [num_train, num_val, 0],
            generator=torch.Generator().manual_seed(1),
        )
        self._testset = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=False, download=True, transform=transform
        )

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self._train_subset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=10,
        )
        return trainloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_subset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self):
        testloader = torch.utils.data.DataLoader(
            self._testset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
        return testloader


def cifar100(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))

    try:
        data_dir = f"{hydra.utils.get_original_cwd()}/data"
    except:
        data_dir = "./data"

    print(f"Orig working directory    : {data_dir}")

    datamodule = Cifar100DataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
    )

    trainer = Trainer(max_epochs=cfg.max_epochs, accelerator=cfg.accelerator)
    model = ClassificationNet(cfg)
    trainer.fit(model, datamodule=datamodule)
    result = trainer.test(model)
    return result


@hydra.main(config_path="../config", config_name="cifar100_config", version_base="1.3")
def run(cfg: DictConfig):
    cifar100(cfg=cfg)


if __name__ == "__main__":
    run()
