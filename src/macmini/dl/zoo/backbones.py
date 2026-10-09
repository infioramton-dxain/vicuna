""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Sequential):
    def __init__(self, d_in: int, dropout: float = 0.1):
        super().__init__(nn.LayerNorm(d_in), nn.Dropout(dropout), nn.Linear(d_in, 1))


class NHiTS(nn.Module):
    ""

    LOSS = "mse"

    def __init__(self, n_feat: int, seq_len: int = 64, d_hidden: int = 256,
                 pool_rates: tuple[int, ...] = (8, 4, 1), dropout: float = 0.1):
        super().__init__()
        self.stacks = nn.ModuleList()
        for rate in pool_rates:
            pooled_len = seq_len // rate
            d_in = pooled_len * n_feat
            self.stacks.append(nn.ModuleDict({
                "pool": nn.AvgPool1d(kernel_size=rate, stride=rate),
                "mlp": nn.Sequential(
                    nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                ),
                "backcast": nn.Linear(d_hidden, d_in),
                "forecast": nn.Linear(d_hidden, 1),
            }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.transpose(1, 2)
        out = 0.0
        for stack in self.stacks:
            pooled = stack["pool"](residual)
            flat = pooled.flatten(1)
            h = stack["mlp"](flat)
            back = stack["backcast"](h).view_as(pooled)
            residual = residual - F.interpolate(back, size=residual.shape[-1],
                                                mode="nearest")
            out = out + stack["forecast"](h)
        return out.squeeze(-1)


class TFTLite(nn.Module):
    ""

    LOSS = "mse"

    def __init__(self, n_feat: int, seq_len: int = 64, d_model: int = 96,
                 n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.var_select = nn.Sequential(
            nn.Linear(n_feat, n_feat), nn.ReLU(), nn.Linear(n_feat, n_feat))
        self.proj = nn.Linear(n_feat, d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.head = _Head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.var_select(x), dim=-1)
        h = self.proj(x * weights)
        enc, _ = self.lstm(h)
        q = enc[:, -1:, :]
        ctx, _ = self.attn(q, enc, enc, need_weights=False)
        fused = enc[:, -1, :] + self.gate(ctx.squeeze(1)) * ctx.squeeze(1)
        return self.head(fused).squeeze(-1)


class DeepAR(nn.Module):
    ""

    LOSS = "gaussian_nll"

    def __init__(self, n_feat: int, seq_len: int = 64, d_model: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, d_model, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.out = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        return self.out(h[:, -1, :])


class AttentionLSTM(nn.Module):
    LOSS = "mse"

    def __init__(self, n_feat: int, seq_len: int = 64, d_model: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, d_model, batch_first=True)
        self.attn_score = nn.Linear(d_model, 1)
        self.head = _Head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        w = torch.softmax(self.attn_score(h), dim=1)
        return self.head((w * h).sum(dim=1)).squeeze(-1)


class GRUForecaster(nn.Module):
    LOSS = "mse"

    def __init__(self, n_feat: int, seq_len: int = 64, d_model: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(n_feat, d_model, num_layers=n_layers,
                          batch_first=True, dropout=dropout)
        self.head = _Head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1, :]).squeeze(-1)


class _CausalConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    LOSS = "mse"

    def __init__(self, n_feat: int, seq_len: int = 64, d_model: int = 96,
                 n_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        layers, c_in = [], n_feat
        for i in range(n_blocks):
            layers += [_CausalConv(c_in, d_model, dilation=2 ** i),
                       nn.ReLU(), nn.Dropout(dropout)]
            c_in = d_model
        self.net = nn.Sequential(*layers)
        self.head = _Head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))
        return self.head(h[:, :, -1]).squeeze(-1)


class VanillaTransformer(nn.Module):
    LOSS = "mse"

    def __init__(self, n_feat: int, seq_len: int = 64, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        pos = torch.zeros(seq_len, d_model)
        position = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pos[:, 0::2] = torch.sin(position * div)
        pos[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pos", pos.unsqueeze(0))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = _Head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(self.proj(x) + self.pos[:, : x.shape[1]])
        return self.head(h.mean(dim=1)).squeeze(-1)


class WaveNet(nn.Module):
    ""

    LOSS = "mse"

    def __init__(self, n_feat: int, seq_len: int = 64, d_model: int = 64,
                 n_blocks: int = 5, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, d_model, 1)
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(n_blocks):
            self.filters.append(_CausalConv(d_model, d_model, dilation=2 ** i))
            self.gates.append(_CausalConv(d_model, d_model, dilation=2 ** i))
            self.skips.append(nn.Conv1d(d_model, d_model, 1))
        self.head = _Head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))
        skip_sum = 0.0
        for f, g, s in zip(self.filters, self.gates, self.skips):
            z = torch.tanh(f(h)) * torch.sigmoid(g(h))
            skip_sum = skip_sum + s(z)
            h = h + z
        return self.head(torch.relu(skip_sum)[:, :, -1]).squeeze(-1)


class AutoEncoderReg(nn.Module):
    ""

    LOSS = "ae"

    def __init__(self, n_feat: int, seq_len: int = 64, d_latent: int = 32,
                 d_hidden: int = 256, recon_weight: float = 0.3):
        super().__init__()
        d_in = n_feat * seq_len
        self.recon_weight = recon_weight
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(),
                                 nn.Linear(d_hidden, d_latent))
        self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.ReLU(),
                                 nn.Linear(d_hidden, d_in))
        self.head = _Head(d_latent)

    def forward(self, x: torch.Tensor):
        flat = x.flatten(1)
        z = self.enc(flat)
        return self.head(z).squeeze(-1), self.dec(z), flat


class VAEReg(AutoEncoderReg):
    ""

    LOSS = "vae"

    def __init__(self, n_feat: int, seq_len: int = 64, d_latent: int = 32,
                 d_hidden: int = 256, recon_weight: float = 0.3,
                 kl_weight: float = 1e-3):
        super().__init__(n_feat, seq_len, d_latent, d_hidden, recon_weight)
        self.kl_weight = kl_weight
        self.enc_logvar = nn.Sequential(nn.Linear(n_feat * seq_len, d_hidden),
                                        nn.ReLU(), nn.Linear(d_hidden, d_latent))

    def forward(self, x: torch.Tensor):
        flat = x.flatten(1)
        mu = self.enc(flat)
        logvar = self.enc_logvar(flat).clamp(-10, 10)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) \
            if self.training else mu
        return (self.head(z).squeeze(-1), self.dec(z), flat, mu, logvar)
