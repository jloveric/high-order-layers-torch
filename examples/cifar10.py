import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy

from high_order_layers_torch.FunctionalConvolution import (
    PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2d,
)
from high_order_layers_torch.FunctionalConvolution import (
    PiecewisePolynomialConvolution2d as PiecewisePolyConv2d,
)
from high_order_layers_torch.FunctionalConvolution import (
    PolynomialConvolution2d as PolyConv2d,
)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# TODO: combine mnist, cifar10 and cifar100 to use a single network that uses
# fully convolutional network with different heads to reduce code.
class Net(LightningModule):
    def __init__(self, n, batch_size, segments=1, layer_type="continuous"):
        super().__init__()
        self.n = n
        self._batch_size = batch_size
        self._layer_type = layer_type
        self._accuracy = Accuracy()

        if layer_type == "continuous":
            self.conv1 = PolyConv2d(n, in_channels=3, out_channels=6, kernel_size=5)
            self.conv2 = PolyConv2d(n, in_channels=6, out_channels=16, kernel_size=5)
        elif layer_type == "piecewise":
            self.conv1 = PiecewisePolyConv2d(
                n, segments=segments, in_channels=3, out_channels=6, kernel_size=5
            )
            self.conv2 = PiecewisePolyConv2d(
                n, segments=segments, in_channels=6, out_channels=16, kernel_size=5
            )
        elif layer_type == "discontinuous":
            self.conv1 = PiecewiseDiscontinuousPolyConv2d(
                n, segments=segments, in_channels=3, out_channels=6, kernel_size=5
            )
            self.conv2 = PiecewiseDiscontinuousPolyConv2d(
                n, segments=segments, in_channels=6, out_channels=16, kernel_size=5
            )
        elif layer_type == "standard":
            self.conv1 = torch.nn.Conv2d(
                in_channels=3, out_channels=6 * ((n - 1) * segments + 1), kernel_size=5
            )
            self.conv2 = torch.nn.Conv2d(
                in_channels=6 * ((n - 1) * segments + 1), out_channels=16, kernel_size=5
            )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        if self._layer_type == "standard":
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.reshape(-1, 16 * 5 * 5)
            x = self.fc1(x)
        else:
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = x.reshape(-1, 16 * 5 * 5)
            x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

    def train_dataloader(self):
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=4, shuffle=True, num_workers=10
        )
        return trainloader

    def test_dataloader(self):
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=10
        )
        return testloader

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self._accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


def run_cifar10(
    max_epochs: int = 1,
    gpus: int = 1,
    n: int = 7,
    batch_size: int = 16,
    segments: int = 4,
    layer_type: str = "piecewise",
):
    trainer = Trainer(max_epochs=max_epochs, gpus=gpus)
    model = Net(n=n, batch_size=batch_size, segments=segments, layer_type=layer_type)

    trainer.fit(model)
    print("testing")
    trainer.test(model)
    print("finished testing")


if __name__ == "__main__":
    run_cifar10()
