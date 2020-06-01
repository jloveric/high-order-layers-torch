'''
This example is meant to demonstrate how you can map complex
functions using a single input and single output with polynomial
synaptic weights
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import high_order_layers_torch.PolynomialLayers as poly
from torchvision.datasets import MNIST
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torch.utils.data import random_split
from high_order_layers_torch.PolynomialLayers import *
import math
import os


class simple_func():
    def __init__(self):
        self.factor = 1.5 * 3.14159
        self.offset = 0.25

    def __call__(self, x):
        return 0.5 * torch.cos(self.factor * 1.0/(abs(x) + self.offset))


xTest = np.arange(10000)/5000.0-1.0
xTest = torch.stack([torch.tensor(val) for val in xTest])

xTest = xTest.view(-1, 1)
yTest = simple_func()(xTest)
yTest = yTest.view(-1, 1)

# Loader for reading in a local dataset


class FunctionDataset(Dataset):
    def __init__(self, transform=None):
        self.x = (2.0*torch.rand(1000)-1.0).view(-1, 1)
        self.y = simple_func()(self.x)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.x.clone().detach()[idx], self.y.clone().detach()[idx]

# Simple network consisting of on input and one output
# and no hidden layers.


class PolynomialFunctionApproximation(LightningModule):
    def __init__(self, poly_order, segments=2):
        super().__init__()
        #self.layer = poly.Polynomial(poly_order+1, 1, 1)
        self.layer = poly.PiecewiseDiscontinuousPolynomial(
            poly_order+1, 1, 1, segments)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(FunctionDataset())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


modelSetD = [
    {'name': 'Discontinuous 1'},
    {'name': 'Discontinuous 2'},
    {'name': 'Discontinuous 3'},
    {'name': 'Discontinuous 4'},
    {'name': 'Discontinuous 5'}
]

modelSetC = [
    {'name': 'Continuous 1'},
    {'name': 'Continuous 2'},
    {'name': 'Continuous 3'},
    {'name': 'Continuous 4'},
    {'name': 'Continuous 5'}
]

colorIndex = ['red', 'green', 'blue', 'purple', 'black']
symbol = ['+', 'x', 'o', 'v', '.']

thisModelSet = modelSetC

start_at = 5

for i in range(0, len(thisModelSet)):

    trainer = Trainer(max_epochs=1)
    model = PolynomialFunctionApproximation(poly_order=i+1,segments=3)
    trainer.fit(model)
    predictions = model(xTest)
    plt.scatter(
        xTest.data.numpy(),
        predictions.flatten().data.numpy(),
        c=colorIndex[i],
        marker=symbol[i],
        label=thisModelSet[i]['name'])

plt.plot(xTest.data.numpy(), yTest.data.numpy(),
         '-', label='actual', color='black')
plt.title('fourier synapse - no hidden layers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
