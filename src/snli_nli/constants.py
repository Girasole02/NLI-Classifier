"""Shared SNLI label definitions."""

LABEL_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

LABEL_NAMES = ["entailment", "neutral", "contradiction"]
VALID_LABELS = (0, 1, 2)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
