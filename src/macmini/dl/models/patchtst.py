""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from transformers import PatchTSTConfig, PatchTSTForRegression


class PatchTSTRegressor(L.LightningModule):
    def __init__(
        self,
        n_channels: int,
        context_length: int = 64,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 8,
        ffn_dim: int = 256,
        dropout: float = 0.2,
        head_dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg = PatchTSTConfig(
            num_input_channels=n_channels,
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            head_dropout=head_dropout,
            num_targets=1,
            channel_attention=False,
        )
        self.model = PatchTSTForRegression(cfg)

        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.val_r2 = torchmetrics.R2Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(past_values=x)
        return out.regression_outputs.squeeze(-1)

    def _directional_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target.abs() > 0.005
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        return (pred[mask].sign() == target[mask].sign()).float().mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        self.val_mae.update(pred, y)
        self.val_mse.update(pred, y)
        self.val_r2.update(pred, y)
        dir_acc = self._directional_accuracy(pred, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", self.val_mae, prog_bar=True)
        self.log("val_rmse", self.val_mse.compute().sqrt() if self.val_mse.compute() > 0 else 0.0, prog_bar=False)
        self.log("val_r2", self.val_r2, prog_bar=False)
        self.log("val_dir_acc", dir_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=1e-6,
        )
        return {"optimizer": opt, "lr_scheduler": sched}
