"""Bidirectional LSTM pair encoder."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn


class SNLIModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 3,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 4, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, premise, hypothesis):
        _, (prem_hn, _) = self.lstm(self.embedding(premise))
        _, (hypo_hn, _) = self.lstm(self.embedding(hypothesis))
        prem_repr = torch.cat((prem_hn[0], prem_hn[1]), dim=-1)
        hypo_repr = torch.cat((hypo_hn[0], hypo_hn[1]), dim=-1)
        combined = torch.cat([prem_repr, hypo_repr], dim=-1)
        return self.classifier(combined)

    def training_step(self, batch, batch_idx):
        logits = self(batch["premise"], batch["hypothesis"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["label"]).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["premise"], batch["hypothesis"])
        loss = self.loss_fn(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch["label"]).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
