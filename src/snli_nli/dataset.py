"""PyTorch Dataset for SNLI sentence pairs."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from snli_nli.tokenization import encode_sentence


class SNLIDataset(Dataset):
    def __init__(
        self,
        premises: list[list[str]],
        hypotheses: list[list[str]],
        labels: list[int],
        vocab: dict[str, int],
        max_len: int = 50,
    ):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "premise": torch.tensor(
                encode_sentence(self.premises[idx], self.vocab, self.max_len),
                dtype=torch.long,
            ),
            "hypothesis": torch.tensor(
                encode_sentence(self.hypotheses[idx], self.vocab, self.max_len),
                dtype=torch.long,
            ),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
