import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer, Callback
from high_order_layers_torch.FunctionalConvolution import (
    PolynomialConvolution2d as PolyConv2d,
)
from high_order_layers_torch.FunctionalConvolution import (
    PiecewisePolynomialConvolution2d as PiecewisePolyConv2d,
)
from high_order_layers_torch.FunctionalConvolution import (
    PiecewiseDiscontinuousPolynomialConvolution2d as PiecewiseDiscontinuousPolyConv2d,
)
from high_order_layers_torch.layers import *
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from torchmetrics import Metric
from high_order_layers_torch.networks import VanillaVAE, HighOrderFullyConvolutionalNetwork, HighOrderFullyDeconvolutionalNetwork
from pytorch_lightning.loggers import TensorBoardLogger
from matplotlib.pyplot import imshow, figure
from torchvision.utils import make_grid
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

class Net(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self._cfg = cfg
        try:
            self._data_dir = f"{hydra.utils.get_original_cwd()}/data"
        except:
            self._data_dir = "../data"

        self._train_fraction = cfg.train_fraction
        self._lr = cfg.lr
        self._batch_size = cfg.batch_size

        self.encoder = HighOrderFullyConvolutionalNetwork(
          layer_type = cfg.layer_type, 
          n=cfg.encoder.n,
          channels=cfg.encoder.channels,
          segments=cfg.encoder.segments,
          kernel_size=cfg.encoder.kernel_size,
          normalization=torch.nn.BatchNorm2d,
          stride=cfg.encoder.stride,
          periodicity=cfg.encoder.periodicity
        )
        self.decoder = HighOrderFullyDeconvolutionalNetwork(
          layer_type = cfg.layer_type,
          n = cfg.decoder.n,
          channels = cfg.decoder.channels,
          segments = cfg.decoder.segments,
          kernel_size = cfg.decoder.kernel_size,
          normalization = torch.nn.BatchNorm2d,
          stride = cfg.decoder.stride,
          periodicity=cfg.decoder.periodicity
        )
        self.model = VanillaVAE(in_channels = 3, latent_dim=cfg.latent_dim, hidden_dims = [], encoder=self.encoder, decoder=self.decoder, device=self.device)
        
    def forward(self, x):
        return self.model(x)

    def setup(self, stage):
        num_train = int(self._train_fraction * 40000)
        num_val = 10000
        num_extra = 40000 - num_train

        train = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=True, download=True, transform=transform
        )

        self._train_subset, self._val_subset, extra = torch.utils.data.random_split(
            train,
            [num_train, 10000, num_extra],
            generator=torch.Generator().manual_seed(1),
        )

    def training_step(self, batch, batch_idx):
        loss = self.eval_step(batch, batch_idx, 'train')
        return loss

    def train_dataloader(self):
        trainloader = torch.utils.data.DataLoader(
            self._train_subset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=10,
        )
        return trainloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self._val_subset, batch_size=self._batch_size, shuffle=False, num_workers=10
        )

    def test_dataloader(self):
        testset = torchvision.datasets.CIFAR100(
            root=self._data_dir, train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=4, shuffle=False, num_workers=10
        )
        return testloader

    def eval_step(self, batch, batch_idx, name):
        #print('eval_step batch', batch, len(batch))
        x, y = batch
        out = self(x)
        
        # Default value for M_N taken from the below test...
        # https://github.com/AntixK/PyTorch-VAE/blob/master/tests/test_vae.py
        loss = self.model.loss_function(*out, M_N = 0.005)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{name}_loss", loss['loss'], prog_bar=True)
        self.log(f"{name}_Reconstruction_Loss", loss['Reconstruction_Loss'], prog_bar=True)
        self.log(f"{name}_KLD", loss['KLD'], prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self._lr)


class ImageSampler(Callback):
    def __init__(self):
        super().__init__()
        self.img_size = None
        self.num_preds = 16

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        figure(figsize=(8, 3), dpi=300)

        with torch.no_grad() :
            samples = pl_module.model.sample(num_samples=5)

        # UNDO DATA NORMALIZATION
        '''
        Here is the transformation from cifar100
        '''
        mean=[n/255. for n in [129.3, 124.1, 112.4]] 
        std=[n/255. for n in [68.2,  65.4,  70.4]]
        

        img = make_grid(samples).permute(1, 2, 0).cpu().numpy() * std + mean

        # PLOT IMAGES
        trainer.logger.experiment.add_image(
            "img", torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step
        )


def vae(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))
    try :
        print(f"Orig working directory    : {hydra.utils.get_original_cwd()}")
    except :
        pass
    
    sampler = ImageSampler()
    logger = TensorBoardLogger("tb_logs", name="vae")

    # , overfit_batches=2
    trainer = Trainer(max_epochs=cfg.max_epochs, gpus=cfg.gpus, logger=logger, callbacks=[sampler] )
    model = Net(cfg)
    trainer.fit(model)
    
    print("testing")
    result = trainer.test(model)
    
    print("finished testing")
    print('result', result)
    return result


@hydra.main(config_path="../config", config_name="variational_autoencoder")
def run(cfg: DictConfig):
    ans = vae(cfg=cfg)
    
    # needed for nevergrad
    return ans[0]['test_loss']


if __name__ == "__main__":
    run()
