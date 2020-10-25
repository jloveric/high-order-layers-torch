import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util
from pytorch_lightning import LightningModule, Trainer
from functional_layers.FunctionalConvolution import PolynomialConvolution2d as PolyConv2d

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

class Net(LightningModule):
    def __init__(self, n, batch_size):
        super().__init__()
        self.n = n
        self._batch_size = batch_size

        self.conv1 = PolyConv2d(n, in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = PolyConv2d(n, in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        print('x,shape', x.shape)
        #x_new = x.view(x.shape[0], -1)
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

    def train_dataloader(self):
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(trainset, batch_size=self._batch_size, shuffle=True, num_workers=10)

    def test_dataloader(self):
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return torch.utils.data.DataLoader(testset, batch_size=self._batch_size, shuffle=True, num_workers=10)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


trainer = Trainer(max_epochs=2,gpus=0)
model = Net(n=2, batch_size=64)
trainer.fit(model)