"""Exploratory plots for class balance and sentence length."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from snli_nli.data import add_length_columns


def run_eda(train: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = add_length_columns(train)

    print(frame["label_text"].value_counts())
    print(frame[["premise_len", "hypothesis_len"]].describe())

    plt.figure(figsize=(8, 4))
    sns.countplot(x="label_text", data=frame, order=["entailment", "neutral", "contradiction"])
    plt.title("Class distribution in the training set")
    plt.xlabel("Class")
    plt.ylabel("Number of examples")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.histplot(frame["premise_len"], bins=30, kde=True)
    plt.title("Premise length distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "premise_length.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.histplot(frame["hypothesis_len"], bins=30, kde=True, color="orange")
    plt.title("Hypothesis length distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "hypothesis_length.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=frame[["premise_len", "hypothesis_len"]])
    plt.title("Premise vs hypothesis length")
    plt.ylabel("Number of words")
    plt.tight_layout()
    plt.savefig(output_dir / "length_boxplot.png", dpi=150)
    plt.close()

    print(f"Saved EDA figures to {output_dir}")
