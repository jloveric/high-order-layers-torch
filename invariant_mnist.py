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
from high_order_layers_torch.PolynomialLayers import *

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

'''
class NDFunctionApproximation(LightningModule):
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
'''


class Net(LightningModule):
    def __init__(self, poly_order, segments=2):
        super(Net, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

        self.layer1 = poly.PiecewiseDiscontinuousPolynomial(
            poly_order+1, 784, 100, segments)
        self.layer2 = nn.LayerNorm(100)
        self.layer3 = poly.PiecewiseDiscontinuousPolynomial(
            poly_order+1, 100, 10, segments)
        self.layer4 = nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def traning_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        return torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    def configure_optim(self):
        return optim.Adam(self.parameters(), lr=0.001)


net = Net()

'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

if True:
    util.train(net, epochs=1, trainloader=trainloader,
               criterion=criterion, optimizer=optimizer)


print('Finished Training')

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

util.accuracy(net, testloader=testloader)
util.class_accuracy(net, testloader=testloader, classes=classes)
'''
