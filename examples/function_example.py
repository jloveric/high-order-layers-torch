"""
This example is meant to demonstrate how you can map complex
functions using a single input and single output with polynomial
synaptic weights
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
import torch_optimizer as alt_optim
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from high_order_layers_torch.layers import *


class simple_func:
    def __init__(self):
        self.factor = 1.5 * 3.14159
        self.offset = 0.25

    def __call__(self, x):
        return 0.5 * torch.cos(self.factor * 1.0 / (abs(x) + self.offset))


xTest = np.arange(1000) / 500.0 - 1.0
xTest = torch.stack([torch.tensor(val) for val in xTest])

xTest = xTest.view(-1, 1)
yTest = simple_func()(xTest)
yTest = yTest.view(-1, 1)


class FunctionDataset(Dataset):
    """
    Loader for reading in a local dataset
    """

    def __init__(self, transform=None):
        self.x = (2.0 * torch.rand(1000) - 1.0).view(-1, 1)
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

    def __init__(
        self, n, segments=2, function=True, periodicity=None, opt: str = "adam"
    ):
        super().__init__()
        self.automatic_optimization = False
        self.optimizer = opt

        if function == "standard":
            print("Inside standard")
            alpha = 0.0
            layer1 = nn.Linear(in_features=1, out_features=n)
            layer2 = nn.Linear(in_features=n, out_features=n)
            layer3 = nn.Linear(in_features=n, out_features=1)

            self.layer = nn.Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3)

        elif function == "product":
            print("Inside product")
            alpha = 0.0
            layer1 = high_order_fc_layers(
                layer_type=function, in_features=1, out_features=n, alpha=1.0
            )
            layer2 = high_order_fc_layers(
                layer_type=function, in_features=n, out_features=1, alpha=1.0
            )
            self.layer = nn.Sequential(
                layer1,
                # nn.ReLU(),
                layer2,
                # nn.ReLU()
            )
        else:
            self.layer = high_order_fc_layers(
                layer_type=function,
                n=n,
                in_features=1,
                out_features=1,
                segments=segments,
                length=2.0,
                periodicity=periodicity,
            )

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, y = batch
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)

        opt.zero_grad()
        if self.optimizer in ["adahessian"]:
            self.manual_backward(loss, create_graph=True)
        else:
            self.manual_backward(loss, create_graph=False)

        opt.step()

        return {"loss": loss}

    def train_dataloader(self):
        return DataLoader(FunctionDataset(), batch_size=4)

    def configure_optimizers(self):
        if self.optimizer == "adahessian":
            return alt_optim.Adahessian(
                self.layer.parameters(),
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=0.0,
                hessian_power=1.0,
            )
        elif self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=0.001)
        elif self.optimizer == "lbfgs":
            return torch.optim.LBFGS(
                self.parameters(), lr=1, max_iter=20, history_size=100
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not recognized")


modelSetL = [
    {"name": "Relu 2", "n": 2},
    {"name": "Relu 3", "n": 8},
    {"name": "Relu 4", "n": 16},
]

modelSetProd = [
    {"name": "Product 2", "n": 2},
    {"name": "Product 3", "n": 8},
    {"name": "Product 4", "n": 16},
]

modelSetD = [
    {"name": "Discontinuous", "n": 2},
    # {'name': 'Discontinuous 2', 'order' : 2},
    {"name": "Discontinuous", "n": 4},
    # {'name': 'Discontinuous 4', 'order' : 4},
    {"name": "Discontinuous", "n": 6},
]

modelSetC = [
    {"name": "Continuous", "n": 2},
    # {'name': 'Continuous 2', 'order' : 2},
    {"name": "Continuous", "n": 4},
    # {'name': 'Continuous 4', 'order' : 4},
    {"name": "Continuous", "n": 6},
]

modelSetP = [
    {"name": "Polynomial", "n": 10},
    # {'name': 'Continuous 2', 'order' : 2},
    {"name": "Polynomial", "n": 20},
    # {'name': 'Continuous 4', 'order' : 4},
    {"name": "Polynomial", "n": 30},
]

modelSetF = [
    {"name": "Fourier", "n": 10},
    # {'name': 'Continuous 2', 'order' : 2},
    {"name": "Fourier", "n": 20},
    # {'name': 'Continuous 4', 'order' : 4},
    {"name": "Fourier", "n": 30},
]

colorIndex = ["red", "green", "blue", "purple", "black"]
symbol = ["+", "x", "o", "v", "."]


def plot_approximation(
    function,
    model_set,
    segments,
    epochs,
    gpus=0,
    periodicity=None,
    plot_result=True,
    opt="adam",
):
    for i in range(0, len(model_set)):

        trainer = Trainer(max_epochs=epochs, gpus=gpus)

        model = PolynomialFunctionApproximation(
            n=model_set[i]["n"],
            segments=segments,
            function=function,
            periodicity=periodicity,
            opt=opt,
        )

        trainer.fit(model)
        predictions = model(xTest.float())

        if plot_result is True:
            plt.scatter(
                xTest.data.numpy(),
                predictions.flatten().data.numpy(),
                c=colorIndex[i],
                marker=symbol[i],
                label=f"{model_set[i]['name']} {model_set[i]['n']}",
            )

    if plot_result is True:
        plt.plot(
            xTest.data.numpy(), yTest.data.numpy(), "-", label="actual", color="black"
        )
        plt.title("Piecewise Polynomial Function Approximation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()


def plot_results(epochs: int = 20, segments: int = 5, plot: bool = True):

    """
    plt.figure(0)
    plot_approximation("standard", modelSetL, 1, epochs, gpus=0)
    plt.title('Relu Function Approximation')
    """
    """
    plt.figure(0)
    plot_approximation("product", modelSetProd, 1, epochs, gpus=0)
    """

    data = [
        {
            "title": "Piecewise Discontinuous Function Approximation",
            "layer": "discontinuous",
            "model_set": modelSetD,
        },
        {
            "title": "Piecewise Continuous Function Approximation",
            "layer": "continuous",
            "model_set": modelSetC,
        },
        {
            "title": "Polynomial function approximation",
            "layer": "polynomial",
            "model_set": modelSetP,
        },
        {
            "title": "Fourier function approximation",
            "layer": "fourier",
            "model_set": modelSetF,
        },
    ]

    for index, element in enumerate(data):
        if plot is True:
            plt.figure(index)
        plot_approximation(
            element["layer"], element["model_set"], 5, epochs, gpus=0, periodicity=2
        )

        if plot is True:
            plt.title("Piecewise Discontinuous Function Approximation")

    if plot is True:
        plt.show()


if __name__ == "__main__":
    plot_results()
