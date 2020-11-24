import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.metrics.functional import accuracy
from high_order_layers_torch.PolynomialLayers import PiecewisePolynomial, PiecewiseDiscontinuousPolynomial, Polynomial
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
    def __init__(self, n, segments=2, batch_size=32, layer_type="continuous"):
        super().__init__()
        self._batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()

        if layer_type == "continuous":
            self.layer1 = PiecewisePolynomial(
                n, 784, 100, segments)
            self.layer2 = nn.LayerNorm(100)
            self.layer3 = PiecewisePolynomial(
                n, 100, 10, segments)
        elif layer_type == "discontinuous":
            self.layer1 = PiecewiseDiscontinuousPolynomial(
                n, 784, 100, segments)
            self.layer2 = nn.LayerNorm(100)
            self.layer3 = PiecewiseDiscontinuousPolynomial(
                n, 100, 10, segments)
        elif layer_type == "polynomial":
            self.layer1 = Polynomial(
                n, 784, 100)
            self.layer2 = nn.LayerNorm(100)
            self.layer3 = Polynomial(
                n, 100, 10)

        self.layer4 = nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_new = x.view(x.shape[0], -1)
        y_hat = self(x_new)
        return F.cross_entropy(y_hat, y)

    def train_dataloader(self):
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=self.cfg.download, transform=transform)
        return torch.utils.data.DataLoader(trainset, batch_size=self._batch_size, shuffle=True, num_workers=10)

    def test_dataloader(self):
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=self.cfg.download, transform=transform)
        return torch.utils.data.DataLoader(testset, batch_size=self._batch_size, shuffle=False, num_workers=10)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'val')

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        x_new = x.view(x.shape[0], -1)
        logits = self(x_new)
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


trainer = Trainer(max_epochs=10, gpus=1)
model = Net(n=3, segments=2, batch_size=64, layer_type="continuous")
trainer.fit(model)
print('testing')
trainer.test(model)
print('finished testing')
