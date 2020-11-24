import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from functional_layers.FunctionalConvolution import PolynomialConvolution2d as PolyConv2d
from functional_layers.FunctionalConvolution import PiecewisePolynomialConvolution2d as PiecewisePolyConv2d
from functional_layers.FunctionalConvolution import PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2d
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from functional_layers.PolynomialLayers import PiecewiseDiscontinuousPolynomial, PiecewisePolynomial, Polynomial

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class Net(LightningModule):
    def __init__(self, n, batch_size, segments=1, layer_type="continuous"):
        super().__init__()
        self.n = n
        self._batch_size = batch_size
        self._layer_type = layer_type

        if layer_type == "continuous":
            self.conv1 = PolyConv2d(
                n, in_channels=1, out_channels=6, kernel_size=5)
            self.conv2 = PolyConv2d(
                n, in_channels=6, out_channels=16, kernel_size=5)
        elif layer_type == "piecewise":
            self.conv1 = PiecewisePolyConv2d(
                n, segments=segments, in_channels=1, out_channels=6, kernel_size=5)
            self.conv2 = PiecewisePolyConv2d(
                n, segments=segments, in_channels=6, out_channels=16, kernel_size=5)
        elif layer_type == "discontinuous":
            self.conv1 = PiecewiseDiscontinuousPolyConv2d(
                n, segments=segments, in_channels=1, out_channels=6, kernel_size=5)
            self.conv2 = PiecewiseDiscontinuousPolyConv2d(
                n, segments=segments, in_channels=6, out_channels=16, kernel_size=5)
        elif layer_type == "standard":
            self.conv1 = torch.nn.Conv2d(
                in_channels=1, out_channels=6*((n-1)*segments+1), kernel_size=5)
            self.conv2 = torch.nn.Conv2d(
                in_channels=6*((n-1)*segments+1), out_channels=16, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x):
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
        print('calling setup')
        train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        self._train_subset, self._val_subset = torch.utils.data.random_split(
            train, [50000, 10000], generator=torch.Generator().manual_seed(1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self._train_subset, batch_size=self._batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self._val_subset, batch_size=self._batch_size, shuffle=False, num_workers=10)

    def test_dataloader(self):
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        return torch.utils.data.DataLoader(testset, batch_size=self._batch_size, shuffle=False, num_workers=10)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f'{name}_loss', loss, prog_bar=True)
        self.log(f'{name}_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


def run_mnist(max_epochs: int = 100, gpus: int = 1, n: int = 3, batch_size: int = 16, segments: int = 2, layer_type: str = "piecewise"):
    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.00, patience=3, verbose=False, mode='min')

    trainer = Trainer(max_epochs=max_epochs, gpus=gpus,
                      callbacks=[early_stop_callback])
    model = Net(n=n, batch_size=batch_size,
                segments=segments, layer_type=layer_type)
    trainer.fit(model)
    print('testing')
    trainer.test(model)
    print('finished testing')


if __name__ == "__main__":
    run_mnist()
