import os

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_optimizer as alt_optim
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics.functional import accuracy
from lion_pytorch import Lion


from high_order_layers_torch.layers import *
from high_order_layers_torch.networks import *
from high_order_layers_torch.PolynomialLayers import *


class Net(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
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

        normalization = None
        if cfg.mlp.normalize is not False:
            normalization = normalization_layers[cfg.mlp.normalize]

        self.layer = HighOrderMLP(
            layer_type=cfg.mlp.layer_type,
            n=cfg.mlp.n,
            in_width=cfg.mlp.input.width,
            out_width=cfg.mlp.output.width,
            in_segments=cfg.mlp.input.segments,
            out_segments=cfg.mlp.output.segments,
            hidden_segments=cfg.mlp.hidden.segments,
            periodicity=cfg.periodicity,
            hidden_layers=cfg.mlp.hidden.layers,
            hidden_width=cfg.mlp.hidden.width,
            normalization=normalization,
        )

        initialize_network_polynomial_layers(self.layer, max_slope=1.0, max_offset=0.0)

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

        if self.cfg.optimizer in ["adahessian"]:
            self.manual_backward(loss, create_graph=True)
        else:
            self.manual_backward(loss, create_graph=False)
        # torch.nn.utils.clip_grad_norm_(
        #    self.parameters(), self._cfg.gradient_clip_val
        # )

        self.log(f"train_loss", loss, prog_bar=True)
        opt.step()
        opt.zero_grad()

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
        acc = accuracy(preds, y, task='multiclass',num_classes=10)

        self.log(f"{name}_loss", loss, prog_bar=True)
        self.log(f"{name}_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self.cfg.optimizer.name == "adahessian":
            return alt_optim.Adahessian(
                self.parameters(),
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1e-4,
                weight_decay=0.0,
                hessian_power=1.0,
            )
        elif self.cfg.optimizer.name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.cfg.optimizer.patience,
                factor=self.cfg.optimizer.factor,
                verbose=True,
            )
            return [optimizer], [lr_scheduler]
        elif self.cfg.optimizer.name == "lion":
            optimizer = Lion(self.parameters(), lr=self.cfg.optimizer.lr)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.cfg.optimizer.patience,
                factor=self.cfg.optimizer.factor,
                verbose=True,
            )
            return [optimizer], [lr_scheduler]
        elif self.cfg.optimizer.name == "lbfgs":
            return optim.LBFGS(self.parameters(), lr=1, max_iter=20, history_size=100)
        else:
            raise ValueError(f"Optimizer {self.cfg.optimizer} not recognized")

    # def configure_optimizers(self):
    #    return optim.Adam(self.parameters(), lr=0.001)


def invariant_mnist(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    try:
        print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    except:
        pass

    lr_monitor = LearningRateMonitor(logging_interval="step")

    diff = 1
    if cfg.mlp.p_refine is False:
        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            accelerator=cfg.accelerator,
            callbacks=[lr_monitor],
        )
        model = Net(cfg)
        trainer.fit(model)
    else:
        diff = cfg.mlp.target_n - cfg.mlp.n
        model = Net(cfg)

        for order in range(cfg.mlp.n, cfg.mlp.target_n):
            trainer = Trainer(
                max_epochs=cfg.max_epochs // diff, accelerator=cfg.accelerator
            )
            print(f"Training order {order}")
            trainer.fit(model)
            trainer.test(model)
            cfg.mlp.n = order + 1
            next_model = Net(cfg)

            interpolate_high_order_mlp(
                network_in=model.layer, network_out=next_model.layer
            )
            model = next_model
        trainer = Trainer(
            max_epochs=cfg.max_epochs // diff, accelerator=cfg.accelerator
        )

    result = trainer.test(model)
    print("result", result)
    return result


@hydra.main(config_path="../config", config_name="invariant_mnist")
def run(cfg: DictConfig):
    invariant_mnist(cfg)


if __name__ == "__main__":
    run()
