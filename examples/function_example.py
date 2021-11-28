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
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torch.utils.data import random_split
from high_order_layers_torch.layers import *

import math
import os


class simple_func():
    def __init__(self):
        self.factor = 1.5 * 3.14159
        self.offset = 0.25

    def __call__(self, x):
        return 0.5 * torch.cos(self.factor * 1.0/(abs(x) + self.offset))


xTest = np.arange(1000)/500.0-1.0
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

    def __init__(self, n, segments=2, function=True, periodicity=None):
        super().__init__()

        if function == "standard" :
            print('Inside standard')
            alpha = 0.0
            layer1 = nn.Linear(in_features=1, out_features=n)
            layer2 = nn.Linear(in_features=n, out_features=n)
            layer3 = nn.Linear(in_features=n, out_features=1)

            self.layer = nn.Sequential(
                layer1, 
                nn.ReLU(), 
                layer2,
                nn.ReLU(),
                layer3
            ) 

        elif function == "product":
            print('Inside product')
            alpha = 0.0
            layer1 = high_order_fc_layers(
                layer_type=function, in_features=1, out_features=n, alpha=1.0)
            layer2 = high_order_fc_layers(
                layer_type=function, in_features=n, out_features=1, alpha=1.0)
            self.layer = nn.Sequential(
                layer1, 
                #nn.ReLU(), 
                layer2, 
                #nn.ReLU()
            )
        else:
            self.layer = high_order_fc_layers(
                layer_type=function, n=n, in_features=1, out_features=1, segments=segments, length=2.0, periodicity=periodicity)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(FunctionDataset(), batch_size=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

modelSetL = [
    {'name': 'Relu 2', 'n': 2},
    {'name': 'Relu 3', 'n': 8},
    {'name': 'Relu 4', 'n': 16}
]

modelSetProd = [
    {'name': 'Product 2', 'n': 2},
    {'name': 'Product 3', 'n': 8},
    {'name': 'Product 4', 'n': 16}
]

modelSetD = [
    {'name': 'Discontinuous', 'n': 2},
    #{'name': 'Discontinuous 2', 'order' : 2},
    {'name': 'Discontinuous', 'n': 4},
    #{'name': 'Discontinuous 4', 'order' : 4},
    {'name': 'Discontinuous', 'n': 6}
]

modelSetC = [
    {'name': 'Continuous', 'n': 2},
    #{'name': 'Continuous 2', 'order' : 2},
    {'name': 'Continuous', 'n': 4},
    #{'name': 'Continuous 4', 'order' : 4},
    {'name': 'Continuous', 'n': 6}
]

modelSetP = [
    {'name': 'Polynomial', 'n': 10},
    #{'name': 'Continuous 2', 'order' : 2},
    {'name': 'Polynomial', 'n': 20},
    #{'name': 'Continuous 4', 'order' : 4},
    {'name': 'Polynomial', 'n': 30}
]

modelSetF = [
    {'name': 'Fourier', 'n': 10},
    #{'name': 'Continuous 2', 'order' : 2},
    {'name': 'Fourier', 'n': 20},
    #{'name': 'Continuous 4', 'order' : 4},
    {'name': 'Fourier', 'n': 30}
]

colorIndex = ['red', 'green', 'blue', 'purple', 'black']
symbol = ['+', 'x', 'o', 'v', '.']


def plot_approximation(function, model_set, segments, epochs, gpus=0, periodicity=None):
    for i in range(0, len(model_set)):

        trainer = Trainer(max_epochs=epochs, gpus=gpus)

        model = PolynomialFunctionApproximation(
            n=model_set[i]['n'], segments=segments, function=function, periodicity=periodicity)

        trainer.fit(model)
        predictions = model(xTest.float())
        plt.scatter(
            xTest.data.numpy(),
            predictions.flatten().data.numpy(),
            c=colorIndex[i],
            marker=symbol[i],
            label=f"{model_set[i]['name']} {model_set[i]['n']}")

    plt.plot(xTest.data.numpy(), yTest.data.numpy(),
             '-', label='actual', color='black')
    plt.title('Piecewise Polynomial Function Approximation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

epochs=20

'''
plt.figure(0)
plot_approximation("standard", modelSetL, 1, epochs, gpus=0)
plt.title('Relu Function Approximation')
'''
"""
plt.figure(0)
plot_approximation("product", modelSetProd, 1, epochs, gpus=0)
"""

plt.figure(1)
plot_approximation("discontinuous", modelSetD, 5, epochs, gpus=0, periodicity=2)
plt.title('Piecewise Discontinuous Function Approximation')


plt.figure(2)
plot_approximation("continuous", modelSetC, 5, epochs, gpus=0, periodicity=2)
plt.title('Piecewise Continuous Function Approximation')


plt.figure(3)
plot_approximation("polynomial", modelSetP, 5, epochs, gpus=0, periodicity=2)
plt.title('Polynomial Function Approximation')


plt.figure(4)
plot_approximation("fourier", modelSetF, 5, epochs, gpus=0)
plt.title('Fourier Function Approximation')

plt.show()
