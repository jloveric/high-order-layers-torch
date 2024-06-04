"""
This example is meant to demonstrate how you can map complex
functions using a single input and single output with polynomial
synaptic weights
"""

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from lion_pytorch import Lion
import hydra
from omegaconf import DictConfig


import high_order_layers_torch.PolynomialLayers as poly
from high_order_layers_torch.layers import *
from high_order_layers_torch.PolynomialLayers import *
from Sophia import SophiaG

elements = 100
a = torch.linspace(-1, 1, elements)
b = torch.linspace(-1, 1, elements)
xv, yv = torch.meshgrid([a, b])
xTest = torch.stack([xv.flatten(), yv.flatten()], dim=1)


class XorDataset(Dataset):
    def __init__(self, transform=None):
        x = (2.0 * torch.rand(1000) - 1.0).view(-1, 1)
        y = (2.0 * torch.rand(1000) - 1.0).view(-1, 1)
        z = torch.where(x * y > 0, -0.5 + 0 * x, 0.5 + 0 * x)

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
    def __init__(
        self,
        n,
        segments=2,
        layer_type="continuous",
        linear_part: float = 1.0,
        optimizer: str = "sophia",
        lr: float = 0.01,
        batch_size: int = 32,
    ):
        """
        Simple network consisting of 2 input and 1 output
        and no hidden layers.
        """
        super().__init__()

        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size

        self.layer1 = high_order_fc_layers(
            layer_type=layer_type,
            n=n,
            in_features=2,
            out_features=2,
            segments=segments,
            alpha=linear_part,
            intialization="constant_random",
        )
        self.layer2 = high_order_fc_layers(
            layer_type=layer_type,
            n=n,
            in_features=2,
            out_features=1,
            segments=segments,
            alpha=linear_part,
            initialization="constant_random",
        )

    def forward(self, x):
        out1 = self.layer2(self.layer1(x))
        return out1

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {"loss": F.mse_loss(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(XorDataset(), batch_size=self.batch_size)

    def configure_optimizers(self):
        if self.optimizer == "lion":
            return Lion(self.parameters(), lr=self.lr)
        elif self.optimizer == "sophia":
            return SophiaG(self.parameters(), lr=self.lr, rho=0.035)
        elif self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else :
            raise ValueError(f"optimizer must be lion, sophia or adam, got {self.optimizer}")


model_set_p = [
    {"name": f"Polynomial {i+1}", "order": i + 1, "layer": "polynomial"}
    for i in range(1, 9, 2)
]
model_set_c = [
    {"name": f"Continuous {i+1}", "order": i + 1, "layer": "continuous"}
    for i in range(1, 5)
]
model_set_d = [
    {"name": f"Discontinuous {i+1}", "order": i + 1, "layer": "discontinuous"}
    for i in range(1, 5)
]


def plot_approximation(
    model_set,
    segments,
    epochs,
    fig_start=0,
    linear_part=0.0,
    plot=True,
    optimizer="sophia",
    lr=0.01,
    batch_size=32,
):
    pred_set = []
    for i in range(0, len(model_set)):

        trainer = Trainer(max_epochs=epochs)
        model = NDFunctionApproximation(
            n=model_set[i]["order"],
            segments=segments,
            layer_type=model_set[i]["layer"],
            linear_part=linear_part,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size
        )
        trainer.fit(model)
        predictions = model(xTest.view(xTest.size(0), -1))
        pred_set.append(predictions)
        if plot is True:
            plt.subplot(2, 2, i + 1)
            plt.scatter(
                xTest.data.numpy()[:, 0],
                xTest.data.numpy()[:, 1],
                c=predictions.flatten().data.numpy(),
            )
            if model_set[i]["layer"] != "polynomial":
                plt.title(f"{model_set[i]['name']} with {segments} segments.")
            else:
                plt.title(f"{model_set[i]['name']}.")

    return pred_set


@hydra.main(config_path="../config", config_name="xor")
def run(cfg: DictConfig):

    plot_style = {
        "continuous": model_set_c,
        "discontinuous": model_set_d,
        "polynomial": model_set_p,
    }

    plot_approximation(
        plot_style[cfg.layer_type],
        segments=cfg.segments,
        epochs=cfg.epochs,
        linear_part=0,
        plot=True,
        optimizer=cfg.optimizer,
        lr=cfg.lr,
        batch_size=cfg.batch_size
    )
    plt.show()


if __name__ == "__main__":
    run()
