# copied straight from the PyTorch examples.
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from high_order_layers_torch.FunctionalConvolution import PolynomialConvolution2d as PolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewisePolynomialConvolution2d as PiecewisePolyConv2d
from high_order_layers_torch.FunctionalConvolution import PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2d
from high_order_layers_torch.layers import *
from pytorch_lightning.metrics.functional import accuracy
from high_order_layers_torch.PolynomialLayers import PiecewiseDiscontinuousPolynomial, PiecewisePolynomial, Polynomial
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pytorch_lightning.metrics import Metric


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class AccuracyTopK(Metric):
    """
    This will eventually be in pytorch-lightning, not yet merged so here it is.
    """

    def __init__(self, top_k=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = top_k
        self.add_state("correct", default=torch.tensor(
            0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(
            0.0), dist_reduce_fx="sum")

    def update(self, logits, y):
        _, pred = logits.topk(self.k, dim=1)
        pred = pred.t()
        corr = pred.eq(y.view(1, -1).expand_as(pred))
        self.correct += corr[:self.k].sum()
        self.total += y.numel()

    def compute(self):
        return self.correct.float() / self.total


class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._cfg = cfg
        self._data_dir = f"{hydra.utils.get_original_cwd()}/data"

        n = cfg.n
        self.n = cfg.n
        self._batch_size = cfg.batch_size
        self._layer_type = cfg.layer_type
        self._train_fraction = cfg.train_fraction
        segments = cfg.segments
        self._topk_metric = AccuracyTopK(top_k=5)

        if self._layer_type == "standard":
            self.conv1 = torch.nn.Conv2d(
                in_channels=3, out_channels=6*((n-1)*segments+1), kernel_size=5)
            self.conv2 = torch.nn.Conv2d(
                in_channels=6*((n-1)*segments+1), out_channels=16, kernel_size=5)

        else:
            self.conv1 = high_order_convolution_layers(
                layer_type=self._layer_type, n=n, in_channels=3, out_channels=6, kernel_size=5, segments=cfg.segments, rescale_output=cfg.rescale_output)
            self.conv2 = high_order_convolution_layers(
                layer_type=self._layer_type, n=n, in_channels=6, out_channels=16, kernel_size=5, segments=cfg.segments, rescale_output=cfg.rescale_output)

        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 100)

    def forward(self, x):
        if self._layer_type == "standard":
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
        else:
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
        return x

    def setup(self, stage):
        num_train = int(self._train_fraction*40000)
        num_val = 10000
        num_extra = 40000-num_train

        train = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=True, download=True, transform=transform)

        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train, [num_train, 10000, num_extra], generator=torch.Generator().manual_seed(1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)

        acc = accuracy(preds, y)
        val = self._topk_metric(y_hat, y)
        val = self._topk_metric.compute()
        
        self.log(f'train_loss', loss, prog_bar=True)
        self.log(f'train_acc', acc, prog_bar=True)
        self.log(f'train_acc5', val, prog_bar=True)

        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self._train_subset, batch_size=self._batch_size, shuffle=True, num_workers=10)
        return trainloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._val_subset, batch_size=self._batch_size, shuffle=False, num_workers=10)

    def test_dataloader(self):
        testset = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=10)
        return testloader

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        val = self._topk_metric(logits, y)
        val = self._topk_metric.compute()

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f'{name}_loss', loss, prog_bar=True)
        self.log(f'{name}_acc', acc, prog_bar=True)
        self.log(f'{name}_acc5', val, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


@hydra.main(config_name="./config/cifar100_config")
def run_cifar100(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")

    trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
    model = Net(cfg)
    trainer.fit(model)
    print('testing')
    trainer.test(model)
    print('finished testing')


if __name__ == "__main__":
    run_cifar100()
