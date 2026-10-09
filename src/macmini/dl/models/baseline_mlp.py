""
from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class BaselineMLP(L.LightningModule):
    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_classes),
        )

        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        preds = logits.argmax(dim=-1)
        if stage == "train":
            self.train_acc.update(preds, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.val_acc.update(preds, y)
            self.val_f1.update(preds, y)
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_f1_macro", self.val_f1, prog_bar=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return opt
