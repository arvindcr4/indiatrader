"""Baseline PatchTST model implementation using PyTorch Lightning."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn


class PatchTSTModel(pl.LightningModule):
    """Minimal PatchTST style model for time series forecasting."""

    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 8, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.proj = nn.Linear(input_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        attn_output, _ = self.attn(x, x, x)
        return self.out(attn_output[:, -1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).squeeze()
        loss = nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train_model(data_module: pl.LightningDataModule, **kwargs) -> PatchTSTModel:
    """Train the PatchTST model with a given Lightning DataModule."""
    model = PatchTSTModel(input_dim=data_module.num_features, **kwargs)
    trainer = pl.Trainer(max_epochs=kwargs.get("epochs", 10), logger=False)
    trainer.fit(model, data_module)
    return model
