import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util
from functional_layers.PolynomialLayers import PiecewisePolynomial
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


class Net(LightningModule):
    def __init__(self, poly_order, segments=2, batch_size=32):
        super().__init__()
        self._batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()

        self.layer1 = PiecewisePolynomial(
            poly_order+1, 784, 100, segments)
        self.layer2 = nn.LayerNorm(100)
        self.layer3 = PiecewisePolynomial(
            poly_order+1, 100, 10, segments)
        #self.layer4 = nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        output = F.log_softmax(x,dim=1)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_new = x.view(x.shape[0], -1)
        y_hat = self(x_new)
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


trainer = Trainer(max_epochs=10,gpus=1)
model = Net(poly_order=2, segments=2, batch_size=64)
trainer.fit(model)
