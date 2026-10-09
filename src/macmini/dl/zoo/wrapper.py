""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from macmini.dl.zoo import backbones as B

BACKBONES: dict[str, type] = {
    "nhits": B.NHiTS,
    "tft": B.TFTLite,
    "deepar": B.DeepAR,
    "attention_lstm": B.AttentionLSTM,
    "gru": B.GRUForecaster,
    "tcn": B.TCN,
    "transformer": B.VanillaTransformer,
    "wavenet": B.WaveNet,
    "autoencoder": B.AutoEncoderReg,
    "vae": B.VAEReg,
}


class ZooRegressor(L.LightningModule):
    def __init__(self, backbone: str, n_feat: int, seq_len: int = 64,
                 learning_rate: float = 3e-4, weight_decay: float = 1e-4,
                 max_epochs: int = 15, **backbone_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.net = BACKBONES[backbone](n_feat=n_feat, seq_len=seq_len,
                                       **backbone_kwargs)
        self.loss_family = self.net.LOSS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.loss_family == "gaussian_nll":
            return out[:, 0]
        if self.loss_family in ("ae", "vae"):
            return out[0]
        return out

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.loss_family == "mse":
            return F.mse_loss(out, y)
        if self.loss_family == "gaussian_nll":
            mu, sigma = out[:, 0], F.softplus(out[:, 1]) + 1e-4
            return F.gaussian_nll_loss(mu, y, sigma ** 2)
        if self.loss_family == "ae":
            pred, recon, flat = out
            return (F.mse_loss(pred, y)
                    + self.net.recon_weight * F.mse_loss(recon, flat))
        if self.loss_family == "vae":
            pred, recon, flat, mu, logvar = out
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return (F.mse_loss(pred, y)
                    + self.net.recon_weight * F.mse_loss(recon, flat)
                    + self.net.kl_weight * kl)
        raise ValueError(self.loss_family)

    def training_step(self, batch, _):
        x, y = batch
        loss = self._loss(x, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self._loss(x, y)
        pred = self(x)
        mask = y.abs() > 0.005
        dir_acc = ((pred[mask].sign() == y[mask].sign()).float().mean()
                   if mask.sum() > 0 else torch.tensor(0.0))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mae", F.l1_loss(pred, y), prog_bar=False)
        self.log("val_dir_acc", dir_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sched}
