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
import functional_layers.PolynomialLayers as poly
from torchvision.datasets import MNIST
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torch.utils.data import random_split
from functional_layers.PolynomialLayers import *
import math
import os


xTest = torch.FloatTensor(100, 2).uniform_(-1, 1)
print(xTest[0])
print('thisTest.shape', xTest.shape)


class XorDataset(Dataset):
    def __init__(self, transform=None):
        x = (2.0*torch.rand(1000)-1.0).view(-1, 1)
        y = (2.0*torch.rand(1000)-1.0).view(-1, 1)
        z = torch.where(x*y > 0, -0.5+0*x, 0.5+0*x)

        self.data = torch.cat([x,y],dim=1)
        self.z = z
        print(self.data.shape)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data.clone().detach()[idx], self.z.clone().detach()[idx]
        #return self.x.clone().detach()[idx], self.y.clone().detach()[idx], self.z.clone().detach()[idx]


class NDFunctionApproximation(LightningModule):
    def __init__(self, poly_order, segments=2):
        """
        Simple network consisting of 2 input and 1 output
        and no hidden layers.
        """
        super().__init__()
        self.layer = poly.PiecewiseDiscontinuousPolynomial(
            poly_order+1, 2, 1, segments)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(XorDataset(), batch_size=4)

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
    model = NDFunctionApproximation(poly_order=i+1, segments=3)
    trainer.fit(model)
    predictions = model(xTest)
    print('predictions.shape', predictions.shape)
    plt.scatter(
        xTest.data.numpy()[:,0],
        xTest.data.numpy()[:,1],
        c=predictions.flatten().data.numpy())

plt.title('fourier synapse - no hidden layers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
