import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import accuracy
from high_order_layers_torch.PolynomialLayers import *
from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import torch_optimizer as alt_optim


class Net(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self._batch_size = cfg.batch_size
        self.criterion = nn.CrossEntropyLoss()
        try:
            self._data_dir = f"{hydra.utils.get_original_cwd()}/data"
        except:
            self._data_dir = "../data"
        self._train_fraction = cfg.train_fraction
        self._val_fraction = cfg.val_fraction
        self._transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # We want to use second order optimizers
        self.automatic_optimization = False

        self.layer = HighOrderMLP(
            layer_type=cfg.layer_type,
            n=cfg.n,
            in_width=784,
            out_width=10,
            in_segments=cfg.segments,
            out_segments=cfg.segments,
            hidden_segments=cfg.segments,
            periodicity=cfg.periodicity,
            hidden_layers=0,
            hidden_width=100,
        )

    def setup(self, stage):
        num_train = int(self._train_fraction * 50000)
        num_val = int(self._val_fraction * 10000)
        num_extra = 50000 - num_train

        train = torchvision.datasets.MNIST(
            root=self._data_dir, train=True, download=True, transform=self._transform
        )
        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train,
            [num_train, 10000, num_extra],
            generator=torch.Generator().manual_seed(1),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        loss = self.eval_step(batch, batch_idx, "train")

        opt.zero_grad()
        if self._cfg.optimizer in ["adahessian"]:
            self.manual_backward(loss, create_graph=True)
        else:
            self.manual_backward(loss, create_graph=False)
        # torch.nn.utils.clip_grad_norm_(
        #    self.parameters(), self._cfg.gradient_clip_val
        # )
        opt.step()

        # return loss

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self._train_subset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=10,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_subset, batch_size=self._batch_size, shuffle=False, num_workers=10
        )

    def test_dataloader(self):
        testset = torchvision.datasets.MNIST(
            root=self._data_dir, train=False, download=True, transform=self._transform
        )
        return torch.utils.data.DataLoader(
            testset, batch_size=self._batch_size, shuffle=False, num_workers=10
        )

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def eval_step(self, batch, batch_idx, name):
        x, y = batch
        x_new = x.view(x.shape[0], -1)
        logits = self(x_new)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self._cfg.optimizer == "adahessian":
            return alt_optim.Adahessian(
                self.parameters(),
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=0.0,
                hessian_power=1.0,
            )
        elif self._cfg.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=0.001)
        elif self._cfg.optimizer == "lbfgs":
            return optim.LBFGS(self.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise ValueError(f"Optimizer {self._cfg.optimizer} not recognized")

    # def configure_optimizers(self):
    #    return optim.Adam(self.parameters(), lr=0.001)


def invariant_mnist(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    try:
        print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    except:
        pass

    diff = 1
    if cfg.p_refine is False:
        trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus)
        model = Net(cfg)
        trainer.fit(model)
    else:
        diff = cfg.target_n - cfg.n
        model = Net(cfg)

        for order in range(cfg.n, cfg.target_n):
            trainer = Trainer(max_epochs=cfg.max_epochs // diff, gpus=cfg.gpus)
            print(f"Training order {order}")
            trainer.fit(model)
            trainer.test(model)
            cfg.n = order + 1
            next_model = Net(cfg)

            interpolate_high_order_mlp(
                network_in=model.layer, network_out=next_model.layer
            )
            model = next_model
        trainer = Trainer(max_epochs=cfg.max_epochs // diff, gpus=cfg.gpus)

    result = trainer.test(model)
    print("result", result)
    return result


@hydra.main(config_path="../config", config_name="invariant_mnist")
def run(cfg: DictConfig):
    invariant_mnist(cfg)


if __name__ == "__main__":
    run()
