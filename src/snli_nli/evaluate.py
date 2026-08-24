"""Run inference and write neural-model metrics."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from snli_nli.model import SNLIModel
from snli_nli.reporting import save_classification_artifacts


def collect_predictions(model: SNLIModel, dataloader: DataLoader, device: torch.device):
    model.eval()
    model.to(device)
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            premise = batch["premise"].to(device)
            hypothesis = batch["hypothesis"].to(device)
            labels = batch["label"].to(device)
            preds = torch.argmax(model(premise, hypothesis), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds


def save_neural_report(y_true, y_pred, output_dir) -> str:
    return save_classification_artifacts(
        y_true,
        y_pred,
        output_dir,
        stem="neural",
        title="BiLSTM confusion matrix",
    )
