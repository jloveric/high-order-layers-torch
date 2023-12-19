import os

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.functional import accuracy

from high_order_layers_torch.layers import *

transformStandard = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
transformPoly = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
)
# transformPoly = transformStandard

classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


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
        layer_type = cfg.layer_type
        segments = cfg.segments

        self._transform = transformPoly

        in_channels = 1
        if self._cfg.add_pos == True:
            in_channels = 3

        if self._layer_type == "standard":
            self.conv1 = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=6 * ((n - 1) * segments + 1),
                kernel_size=5,
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=6 * ((n - 1) * segments + 1), out_channels=16, kernel_size=5
            )
        else:
            self.conv1 = high_order_convolution_layers(
                layer_type=self._layer_type,
                n=n,
                in_channels=in_channels,
                out_channels=6,
                kernel_size=5,
                segments=cfg.segments,
            )
            self.conv2 = high_order_convolution_layers(
                layer_type=self._layer_type,
                n=n,
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                segments=cfg.segments,
            )

        w1 = 28 - 4
        w2 = (w1 // 2) - 4
        c1 = 6
        c2 = 16

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 10)

        # Create xy objects
        if self._cfg.add_pos == True:
            xm = torch.linspace(-1, 1, 28, device=self.device)
            ym = torch.linspace(-1, 1, 28, device=self.device)
            xv, yv = torch.meshgrid(xm, ym)
            xv = torch.stack(self._batch_size * [xv], dim=0)
            yv = torch.stack(self._batch_size * [yv], dim=0)
            # This is a hack.  Apparently self.device is not on cuda.
            self._pos = torch.stack([xv, yv], dim=1).cuda()

    def forward(self, xin):

        if self._cfg.add_pos == True:
            x = torch.cat([xin, self._pos], dim=1)
        else:
            x = xin

        if self._layer_type == "standard":
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.reshape(-1, 16 * 4 * 4)
            x = self.fc1(x)
        else:
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = x.reshape(-1, 16 * 4 * 4)
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
        acc = accuracy(preds, y, task='multiclass',num_classes=10)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


def mnist(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))

    try:
        print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    except:
        pass

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min"
    )

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        callbacks=[early_stop_callback],
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
