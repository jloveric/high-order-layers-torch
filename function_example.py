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
from torchvision.datasets import MNIST
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torch.utils.data import random_split
from functional_layers.PolynomialLayers import PiecewiseDiscontinuousPolynomial, PiecewisePolynomial
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


class FunctionDataset(Dataset):
    """
    Loader for reading in a local dataset
    """

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


class PolynomialFunctionApproximation(LightningModule):
    """
    Simple network consisting of on input and one output
    and no hidden layers.
    """

    def __init__(self, poly_order, segments=2, continuous=True):
        super().__init__()
        if continuous:
            print('Using continuous piecewise polynomial.')
            self.layer = PiecewisePolynomial(
                poly_order+1, 1, 1, segments)
        else:
            print('Using discontinuous piecewise polynomial.')
            self.layer = PiecewiseDiscontinuousPolynomial(
                poly_order+1, 1, 1, segments)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(FunctionDataset(), batch_size=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.1)


modelSetD = [
    {'name': 'Discontinuous 1', 'order' : 1},
    #{'name': 'Discontinuous 2', 'order' : 2},
    {'name': 'Discontinuous 3', 'order' : 3},
    #{'name': 'Discontinuous 4', 'order' : 4},
    {'name': 'Discontinuous 5', 'order' : 5}
]

modelSetC = [
    {'name': 'Continuous 1', 'order' : 1},
    #{'name': 'Continuous 2', 'order' : 2},
    {'name': 'Continuous 3', 'order' : 3},
    #{'name': 'Continuous 4', 'order' : 4},
    {'name': 'Continuous 5', 'order' : 5}
]

colorIndex = ['red', 'green', 'blue', 'purple', 'black']
symbol = ['+', 'x', 'o', 'v', '.']


def plot_approximation(continuous, model_set, segments, epochs):
    for i in range(0, len(model_set)):

        trainer = Trainer(max_epochs=epochs)

        model = PolynomialFunctionApproximation(
            poly_order=model_set[i]['order'], segments=segments, continuous=continuous)

        trainer.fit(model)
        predictions = model(xTest)
        plt.scatter(
            xTest.data.numpy(),
            predictions.flatten().data.numpy(),
            c=colorIndex[i],
            marker=symbol[i],
            label=model_set[i]['name'])

    plt.plot(xTest.data.numpy(), yTest.data.numpy(),
             '-', label='actual', color='black')
    plt.title('Piecewise Polynomial Function Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()


plt.figure(1)
plot_approximation(False, modelSetD, 3, 1)
plt.figure(2)
plot_approximation(True, modelSetC, 3, 1)
plt.show()
