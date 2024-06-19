"""
Using a polynomial 3d to solve this problem
"""

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

normalization = {
    "max_abs": MaxAbsNormalizationND,
    "max_center": MaxCenterNormalizationND,
}

grid_x, grid_y = torch.meshgrid(
    (torch.arange(28) - 14) // 14, (torch.arange(28) - 14) // 14, indexing="ij"
)
grid = torch.stack([grid_x, grid_y])


def collate_fn(batch):
    
    input = []
    classification = []
    for element in batch:
        color_and_xy = torch.cat([element[0], grid]).permute(1, 2, 0).view(-1, 3)
        input.append(color_and_xy)

        classification.append(element[1])

    batch_input = torch.stack(input)
    batch_output = torch.tensor(classification)

    return batch_input, batch_output


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

        self._transform = transformPoly

        layer1 = high_order_fc_layers(
            layer_type=cfg.layer_type,
            n=n,
            in_features=1,
            out_features=10,
            intialization="constant_random",
            device=cfg.accelerator,
        )
        self.model = nn.Sequential(*[layer1])

    def forward(self, x):
        #print("x.shape", x.shape)
        batch_size, inputs = x.shape[:2]
        xin = x.view(-1, 1, 3)
        #print("xin.shape", xin.shape)
        res = self.model(xin)
        res = res.reshape(batch_size, inputs, -1)
        output = torch.sum(res,dim=1)
        #print("res.shape", output.shape)
        # xout = res.view(batch_size, )
        return output

    def setup(self, stage):
        num_train = int(self._train_fraction * 50000)
        num_val = 10000

        # extra only exist if we aren't training on the full dataset
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
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_subset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=10,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        testset = torchvision.datasets.MNIST(
            root=self._data_dir,
            train=False,
            download=True,
            transform=self._transform,
        )
        return torch.utils.data.DataLoader(
            testset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=10,
            collate_fn=collate_fn,
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
                    "reduce_on_plateau": True,
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
                    "reduce_on_plateau": True,
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


@hydra.main(config_path="../config", config_name="block_mnist")
def run(cfg: DictConfig):
    mnist(cfg)


if __name__ == "__main__":
    run()
