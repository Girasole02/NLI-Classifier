"""Whitespace tokenizer and word-level vocabulary."""

from __future__ import annotations

import string
from collections import Counter

from tqdm import tqdm

from snli_nli.constants import PAD_TOKEN, UNK_TOKEN


def tokenize(text: str) -> list[str]:
    tokens = text.lower().split()
    cleaned = [token.strip(string.punctuation) for token in tokens]
    return [token for token in cleaned if token]


def tokenize_pairs(premises: list[str], hypotheses: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    prem_tokens = [tokenize(text) for text in tqdm(premises, desc="Tokenizing premises")]
    hypo_tokens = [tokenize(text) for text in tqdm(hypotheses, desc="Tokenizing hypotheses")]
    return prem_tokens, hypo_tokens


def build_vocab(token_lists: list[list[str]], min_freq: int = 1) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counts.most_common():
        if freq < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_sentence(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    unk_id = vocab[UNK_TOKEN]
    pad_id = vocab[PAD_TOKEN]
    encoded = [vocab.get(token, unk_id) for token in tokens]
    if len(encoded) < max_len:
        return encoded + [pad_id] * (max_len - len(encoded))
    return encoded[:max_len]
