# SNLI Natural Language Inference Classifier

Modular Python package for **natural language inference** on the [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) corpus.

Given a **premise** and a **hypothesis**, the model predicts **entailment**, **neutral**, or **contradiction**.

Two approaches are implemented as separate modules:

1. **Classical baseline** — TF-IDF n-grams + logistic regression  
2. **Neural model** — bidirectional LSTM pair encoder (PyTorch Lightning)

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Using the modules in Python](#using-the-modules-in-python)
- [Dataset](#dataset)
- [Models](#models)
- [Original notebook](#original-notebook)
- [License](#license)

---

## What this project does

- Loads SNLI from Hugging Face Datasets (`train` / `validation` / `test`)
- Drops unlabeled rows (`label == -1`) and maps `0/1/2` to class names
- Plots class balance and sentence lengths
- Trains a TF-IDF + logistic regression baseline (vectorizer fitted on **train only**)
- Builds a vocabulary on **train only**, encodes pairs, trains a BiLSTM
- Writes classification reports and confusion-matrix figures under `outputs/`

By default, evaluation uses the **official SNLI validation split**. Pass `--split internal` to reproduce the original notebook’s 80/20 split of the training table.

---

## Repository layout

```text
snli-nli-classifier/
├── README.md
├── LICENSE
├── pyproject.toml            # installable package + snli-nli CLI
├── requirements.txt
├── src/snli_nli/             # importable modules
├── docs/                     # dataset, models, architecture
└── notebooks/
    └── ml_sublime.py         # original Colab export (reference only)
```

Module map (see [docs/architecture.md](docs/architecture.md) for the data flow):

| Module | Role |
| --- | --- |
| `constants` | Label names, PAD / UNK tokens |
| `data` | Load, clean, concatenate pairs, optional subsample |
| `eda` | Exploratory plots |
| `reporting` | Shared metrics + confusion matrices |
| `baseline` | TF-IDF + logistic regression |
| `tokenization` | Tokenize, vocabulary, padding |
| `dataset` | `SNLIDataset` |
| `model` | `SNLIModel` Lightning module |
| `evaluate` | Collect predictions |
| `train_neural` | Lightning training loop |
| `cli` | Command-line interface |

---

## Installation

Python 3.9+ is recommended. A GPU is optional but much faster for the LSTM.

```bash
git clone https://github.com/Girasole02/snli-nli-classifier.git
cd snli-nli-classifier
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -e .
```

This installs the `snli_nli` package and the `snli-nli` command. First run downloads SNLI (network required).

---

## How to run

```bash
# Exploratory plots
snli-nli eda

# Classical baseline
snli-nli baseline

# BiLSTM (use a GPU if you can)
snli-nli neural --max-epochs 10

# All stages
snli-nli all
```

Equivalent without installing the script:

```bash
python -m snli_nli baseline
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--output-dir PATH` | Figures, reports, checkpoints (default: `outputs`) |
| `--split official` | SNLI validation set (default) |
| `--split internal` | 80/20 split of filtered train |
| `--max-train-samples N` | Stratified cap for a quick CPU test |
| `--max-epochs`, `--batch-size`, `--patience` | Neural training |
| `--seed` | Reproducibility |

Example smoke test:

```bash
snli-nli all --max-train-samples 3000 --max-epochs 1 --split official
```

---

## Using the modules in Python

```python
from pathlib import Path
from snli_nli.data import load_snli_frames, clean_split
from snli_nli.baseline import train_and_evaluate_baseline
from snli_nli.train_neural import train_and_evaluate_neural

frames = load_snli_frames()
train = clean_split(frames["train"])
validation = clean_split(frames["validation"])

train_and_evaluate_baseline(train, validation, Path("outputs/baseline"))
train_and_evaluate_neural(train, validation, Path("outputs/neural"), max_epochs=5)
```

Tokenizer only:

```python
from snli_nli.tokenization import tokenize, build_vocab

tokens = tokenize("A man is playing a guitar on stage.")
```

---

## Dataset

**SNLI** (Bowman et al., 2015): English premise–hypothesis pairs.

| Label | Meaning |
| --- | --- |
| Entailment | Hypothesis follows from the premise |
| Neutral | Neither entailed nor contradicted |
| Contradiction | Hypothesis conflicts with the premise |

Details and citation: [docs/dataset.md](docs/dataset.md)

---

## Models

- **TF-IDF + logistic regression** — concatenated `premise [SEP] hypothesis`, up to trigrams, optional class weights. [docs/classical-baseline.md](docs/classical-baseline.md)
- **BiLSTM** — shared embedding and bidirectional LSTM over each sentence; concatenated hidden states → 3-way classifier; early stopping on `val_loss`. [docs/neural-model.md](docs/neural-model.md)

---

## Original notebook

`notebooks/ml_sublime.py` is the unmodified Colab export. It uses `!pip` magics and is **not** the supported entry point. The package above is the maintained implementation.

---

## License

[MIT License](LICENSE). SNLI is a separate dataset; follow Stanford / Hugging Face terms when using the data.
