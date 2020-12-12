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
from high_order_layers_torch.layers import *

elements = 100
a = torch.linspace(-1, 1,  elements)
b = torch.linspace(-1, 1, elements)
xv, yv = torch.meshgrid([a, b])
xTest = torch.stack([xv.flatten(), yv.flatten()], dim=1)


class XorDataset(Dataset):
    def __init__(self, transform=None):
        x = (2.0*torch.rand(1000)-1.0).view(-1, 1)
        y = (2.0*torch.rand(1000)-1.0).view(-1, 1)
        z = torch.where(x*y > 0, -0.5+0*x, 0.5+0*x)

        self.data = torch.cat([x, y], dim=1)
        self.z = z
        print(self.data.shape)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data.clone().detach()[idx], self.z.clone().detach()[idx]


class NDFunctionApproximation(LightningModule):
    def __init__(self, n, segments=2, layer_type="continuous"):
        """
        Simple network consisting of 2 input and 1 output
        and no hidden layers.
        """
        super().__init__()

        self.layer1 = high_order_fc_layers(
            layer_type=layer_type, n=n, in_features=2, out_features=1, segments=segments, alpha=0.0)
        self.layer2 = high_order_fc_layers(
            layer_type=layer_type, n=n, in_features=1, out_features=1, segments=segments, alpha=0.0)
        
    def forward(self, x):
        out1 = self.layer1(x)
        return out1

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(XorDataset(), batch_size=11)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


model_set_p = [{'name': f"Polynomial {i+1}", "order": i+1} for i in range(2,5)]
model_set_c = [{'name': f"Continuous {i+1}", "order": i+1} for i in range(2,5)]
model_set_d = [{'name': f"Discontinuous {i+1}", "order": i+1}
               for i in range(2,5)]


def plot_approximation(continuous, model_set, segments, epochs, fig_start=0):
    for i in range(0, len(model_set)):
        plt.figure(i+fig_start)
        trainer = Trainer(max_epochs=epochs)
        model = NDFunctionApproximation(
            n=model_set[i]['order'], segments=segments, layer_type=continuous)
        trainer.fit(model)
        predictions = model(xTest.view(xTest.size(0), -1))
        plt.scatter(
            xTest.data.numpy()[:, 0],
            xTest.data.numpy()[:, 1],
            c=predictions.flatten().data.numpy())
        plt.colorbar()
        plt.title(f"{model_set[i]['name']} with {segments} segments.")
        plt.xlabel('x')
        plt.ylabel('y')


plot_approximation("continuous", model_set_p, 2, 4, 0)
plt.show()
