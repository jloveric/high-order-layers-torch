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
from functional_layers.FunctionalConvolution import FourierConvolution2d as FourierConv2d

from pytorch_lightning.metrics.functional import accuracy
from functional_layers.PolynomialLayers import PiecewiseDiscontinuousPolynomial, PiecewisePolynomial, Polynomial

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class Net(LightningModule):
    def __init__(self, n, batch_size, out_channels=6, segments=1, convolution=PolyConv2d):
        super().__init__()
        self.n = n
        self._batch_size = batch_size

        self.conv1 = convolution(
            n, in_channels=1, out_channels=out_channels, kernel_size=5)
        self.conv2 = convolution(
            n, in_channels=1, out_channels=out_channels*2, kernel_size=10)
        self.conv3 = convolution(
            n, in_channels=1, out_channels=out_channels*4, kernel_size=20)

        w1 = 24*24*out_channels
        w2 = 19*19*out_channels*2
        w3 = 9*9*out_channels*4
        in_features = w1+w2+w3

        self.fc1 = nn.Linear(in_features, 10)
        #self.fc1 = Polynomial(n, in_features=in_features, out_features=10)

    def forward(self, x):
        x1 = self.conv1(x).flatten(start_dim=1)
        x2 = self.conv2(x).flatten(start_dim=1)
        x3 = self.conv3(x).flatten(start_dim=1)

        x4 = torch.cat((x1, x2, x3), dim=1)

        out = self.fc1(x4)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

    def train_dataloader(self):
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(trainset, batch_size=self._batch_size, shuffle=True, num_workers=10)

    def test_dataloader(self):
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        return torch.utils.data.DataLoader(testset, batch_size=self._batch_size, shuffle=True, num_workers=10)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f'{name}_loss', loss, prog_bar=True)
        self.log(f'{name}_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.001)


trainer = Trainer(max_epochs=2, gpus=1)
#model = Net(n=40, batch_size=64, out_channels=10, convolution=PolyConv2d)
model = Net(n=40, batch_size=64, out_channels=10, convolution=FourierConv2d)

trainer.fit(model)
print('testing')
trainer.test(model)
print('finished testing')
