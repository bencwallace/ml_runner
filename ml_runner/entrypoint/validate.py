import pytorch_lightning as pl
from torch.utils.data import DataLoader


def validate(pl_module: pl.LightningModule, trainer: pl.Trainer, loader: DataLoader):
    trainer.validate(pl_module, loader)
