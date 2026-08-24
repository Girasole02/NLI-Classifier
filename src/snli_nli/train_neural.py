"""Train the BiLSTM pair encoder with PyTorch Lightning."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from snli_nli.dataset import SNLIDataset
from snli_nli.evaluate import collect_predictions, save_neural_report
from snli_nli.model import SNLIModel
from snli_nli.tokenization import build_vocab, tokenize_pairs


def train_and_evaluate_neural(
    train,
    validation,
    output_dir: Path,
    max_len: int = 50,
    batch_size: int = 64,
    max_epochs: int = 10,
    patience: int = 3,
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    lr: float = 1e-3,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_prem, train_hypo = tokenize_pairs(
        train["premise"].tolist(),
        train["hypothesis"].tolist(),
    )
    val_prem, val_hypo = tokenize_pairs(
        validation["premise"].tolist(),
        validation["hypothesis"].tolist(),
    )
    vocab = build_vocab(train_prem + train_hypo)
    print(f"Vocabulary size: {len(vocab)}")

    train_loader = DataLoader(
        SNLIDataset(train_prem, train_hypo, train["label"].tolist(), vocab, max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SNLIDataset(val_prem, val_hypo, validation["label"].tolist(), vocab, max_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SNLIModel(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        lr=lr,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=True),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                monitor="val_loss",
                mode="min",
                filename="snli-bilstm-{epoch:02d}-{val_loss:.3f}",
                save_top_k=1,
            ),
        ],
        default_root_dir=str(output_dir / "lightning"),
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, val_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = collect_predictions(model, val_loader, device)
    report = save_neural_report(y_true, y_pred, output_dir)
    print("Classification report (BiLSTM)")
    print(report)
    print(f"Saved neural metrics to {output_dir}")
