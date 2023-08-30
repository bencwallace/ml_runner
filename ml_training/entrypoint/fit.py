import pytorch_lightning as pl
from torch.utils.data import DataLoader


def fit(pl_module: pl.LightningModule, trainer: pl.Trainer, train_loader: DataLoader, val_loader: DataLoader):
    trainer.fit(pl_module, train_loader, val_loader)
