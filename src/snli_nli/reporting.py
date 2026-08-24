"""Shared classification report and confusion-matrix helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from snli_nli.constants import LABEL_NAMES


def save_classification_artifacts(
    y_true,
    y_pred,
    output_dir: Path,
    stem: str,
    title: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES)
    (output_dir / f"{stem}_classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_confusion_matrix.png", dpi=150)
    plt.close()
    return report
