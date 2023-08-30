from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class GenericModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def training_step(self, batch, _) -> STEP_OUTPUT:
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _) -> STEP_OUTPUT | None:
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(params=self.parameters())
        return {
            "optimizer": optimizer,
        }
