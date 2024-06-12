import os

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lion_pytorch import Lion
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torchmetrics.functional import accuracy
from Sophia import SophiaG

from high_order_layers_torch.layers import *

transformStandard = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
transformPoly = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
)
# transformPoly = transformStandard

classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

normalization = {
    "max_abs": MaxAbsNormalizationND,
    "max_center": MaxCenterNormalizationND,
}


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)

        self._cfg = cfg
        try:
            self._data_dir = f"{hydra.utils.get_original_cwd()}/data"
        except:
            self._data_dir = "../data"
        n = cfg.n
        self._batch_size = cfg.batch_size
        self._layer_type = cfg.layer_type
        self._train_fraction = cfg.train_fraction
        segments = cfg.segments

        self._transform = transformPoly

        in_channels = cfg.channels[0]
        out_channels = cfg.channels[1]

        if self._layer_type == "standard":
            self.conv1 = torch.nn.Conv2d(
                in_channels=1,
                out_channels=in_channels,
                kernel_size=5,
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=5
            )
        else:
            self.conv1 = high_order_convolution_layers(
                layer_type=self._layer_type,
                n=n,
                in_channels=1,
                out_channels=in_channels,
                kernel_size=cfg.kernel_size,
                segments=cfg.segments,
            )
            if self._cfg.double is True:
                self.conv15 = high_order_convolution_layers(
                    layer_type=self._layer_type,
                    n=n,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=cfg.kernel_size,
                    segments=cfg.segments,
                )

            self.conv2 = high_order_convolution_layers(
                layer_type=self._layer_type,
                n=n,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=cfg.kernel_size,
                segments=cfg.segments,
            )
            if self._cfg.double is True:
                self.conv25 = high_order_convolution_layers(
                    layer_type=self._layer_type,
                    n=n,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=cfg.kernel_size,
                    segments=cfg.segments,
                )

        self.normalize = normalization[cfg.normalization]()
        # self.normalize = MaxAbsNormalizationND()

        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)

        if self._cfg.double is True:
            last_layer_size = (
                4 * 4 * out_channels if cfg.kernel_size == 3 else 2 * 2 * out_channels
            )
        else:
            last_layer_size = (
                5 * 5 * out_channels if cfg.kernel_size == 3 else 4 * 4 * out_channels
            )

        if cfg.output_layer_type == "linear":
            self.fc1 = nn.Linear(last_layer_size, 10)
        else:

            output_layer_type = (
                cfg.output_layer_type
                if cfg.output_layer_type != "auto"
                else cfg.layer_type[:-2]
            )
            print("output_layer_type", output_layer_type)
            self.fc1 = high_order_fc_layers(
                layer_type=output_layer_type,
                n=n,
                in_features=last_layer_size,
                out_features=10,
                segments=segments,
                length=2.0,
                periodicity=None,
            )

    def forward(self, xin):

        x = xin

        if self._cfg.double is False:
            if self._layer_type == "standard":
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.reshape(x.shape[0], -1)
                x = self.fc1(x)
            else:
                x = self.pool(self.conv1(x))
                x = self.normalize(x)
                x = self.pool(self.conv2(x))
                x = self.normalize(x)
                x = x.reshape(x.shape[0], -1)
                x = self.fc1(x)
            return x
        else:
            x = self.conv1(x)
            x = self.normalize(x)
            x = self.pool(self.conv15(x))
            x = self.normalize(x)
            x = self.conv2(x)
            x = self.normalize(x)
            x = self.pool(self.conv25(x))
            x = self.normalize(x)
            x = x.reshape(x.shape[0], -1)
            x = self.fc1(x)
            return x

    def setup(self, stage):
        num_train = int(self._train_fraction * 50000)
        num_val = 10000
        num_extra = 50000 - num_train

        train = torchvision.datasets.MNIST(
            root=self._data_dir, train=True, download=True, transform=self._transform
        )
        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train,
            [num_train, 10000, num_extra],
            generator=torch.Generator().manual_seed(1),
        )

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "train")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_subset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=10,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_subset, batch_size=self._batch_size, shuffle=False, num_workers=10
        )

    def test_dataloader(self):
        testset = torchvision.datasets.MNIST(
            root=self._data_dir, train=False, download=True, transform=self._transform
        )
        return torch.utils.data.DataLoader(
            testset, batch_size=self._batch_size, shuffle=False, num_workers=10
        )

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def eval_step(self, batch, batch_idx, name):
        x, y = batch

        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self._cfg.optimizer.name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self._cfg.optimizer.lr)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self._cfg.optimizer.patience,
                factor=self._cfg.optimizer.factor,
                verbose=True,
            )
            return [optimizer], [
                {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        elif self._cfg.optimizer.name == "lion":
            optimizer = Lion(self.parameters(), lr=self._cfg.optimizer.lr)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self._cfg.optimizer.patience,
                factor=self._cfg.optimizer.factor,
                verbose=True,
            )
            return [optimizer], [
                {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            ]
        elif self._cfg.optimizer.name == "sophia":
            optimizer = SophiaG(
                self.parameters(),
                lr=self._cfg.optimizer.lr,
                rho=self._cfg.optimizer.rho,
            )
            return optimizer


def mnist(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))

    try:
        print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    except:
        pass

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=20, verbose=False, mode="min"
    )

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        callbacks=[lr_monitor],
    )
    model = Net(cfg)
    trainer.fit(model)

    print("testing")
    results = trainer.test(model)

    print("finished testing")
    return results


@hydra.main(config_path="../config", config_name="mnist_config")
def run(cfg: DictConfig):
    mnist(cfg)


if __name__ == "__main__":
    run()
