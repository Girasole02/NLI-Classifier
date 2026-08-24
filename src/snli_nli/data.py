"""Load SNLI from Hugging Face Datasets and drop unlabeled rows."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from snli_nli.constants import LABEL_MAP, VALID_LABELS


def load_snli_frames() -> dict[str, pd.DataFrame]:
    from datasets import load_dataset

    dataset = load_dataset("snli")
    return {
        "train": dataset["train"].to_pandas(),
        "validation": dataset["validation"].to_pandas(),
        "test": dataset["test"].to_pandas(),
    }


def clean_split(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep labels 0/1/2 and attach readable class names."""
    cleaned = frame[frame["label"].isin(VALID_LABELS)].copy()
    cleaned["label_text"] = cleaned["label"].map(LABEL_MAP)
    cleaned["premise"] = cleaned["premise"].astype(str)
    cleaned["hypothesis"] = cleaned["hypothesis"].astype(str)
    return cleaned.reset_index(drop=True)


def add_pair_text(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["text"] = out["premise"] + " [SEP] " + out["hypothesis"]
    return out


def add_length_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["premise_len"] = out["premise"].str.split().str.len()
    out["hypothesis_len"] = out["hypothesis"].str.split().str.len()
    return out


def maybe_subsample(frame: pd.DataFrame, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples >= len(frame):
        return frame
    n_classes = frame["label"].nunique()
    per_class = max(1, max_samples // n_classes)
    parts = [
        group.sample(n=min(len(group), per_class), random_state=seed)
        for _, group in frame.groupby("label")
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
