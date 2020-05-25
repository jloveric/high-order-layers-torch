'''
This example is meant to demonstrate how you can map complex
functions using a single input and single output with polynomial
synaptic weights
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import high_order_layers_torch.PolynomialLayers as poly
from torchvision.datasets import MNIST
from pytorch_lightning import LightningModule, Trainer
from torchvision import transforms
from torch.utils.data import random_split

import os

offset = -0.1
factor = 1.5 * 3.14159
xTest = np.arange(100) / 50 - 1.0
yTest = 0.5 * np.cos(factor * (xTest - offset))

xTrain = tf.random.uniform([1000], minval=-1.0, maxval=1, dtype=tf.float32)
yTrain = 0.5 * tf.math.cos(factor * (xTrain - offset))


class FunctionDataset(Dataset):
    def __init__(self, transform=None):
        self.x = torch.random(
            [1000], minval=-1.0, maxval=1, dtype=tf.float32)
        self.y = 0.5 * tf.math.cos(factor * (xTrain - offset))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x': self.x[idx], 'y': self.y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class PolynomialFunctionApproximation(LightningModule):
    def __init__(self, poly_order):
        super().__init__()
        self.layer = poly.Polynomial(poly_order+1, 1, 1)

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=32)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


modelSetD = [
    {'name': 'Discontinuous 1', 'func': poly.b1D},
    {'name': 'Discontinuous 2', 'func': poly.b2D},
    {'name': 'Discontinuous 3', 'func': poly.b3D},
    {'name': 'Discontinuous 4', 'func': poly.b4D},
    {'name': 'Discontinuous 5', 'func': poly.b5D}
]

modelSetC = [
    {'name': 'Continuous 1', 'func': poly.b1C},
    {'name': 'Continuous 2', 'func': poly.b2C},
    {'name': 'Continuous 3', 'func': poly.b3C},
    {'name': 'Continuous 4', 'func': poly.b4C},
    {'name': 'Continuous 5', 'func': poly.b5C}
]

colorIndex = ['red', 'green', 'blue', 'purple', 'black']
symbol = ['+', 'x', 'o', 'v', '.']

thisModelSet = modelSetC

for i in range(0, len(thisModelSet)):

    trainer = Trainer()
    model = PolynomialFunctionApproximation(poly_order=i+1)
    trainer.fit(model)
    predictions = model(xTest)

    plt.scatter(
        xTest,
        predictions.flatten(),
        c=colorIndex[i],
        marker=symbol[i],
        label=thisModelSet[i]['name'])

plt.plot(xTest, yTest, '-', label='actual', color='black')
plt.title('fourier synapse - no hidden layers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
