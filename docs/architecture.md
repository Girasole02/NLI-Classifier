# Package architecture

The runnable code lives in `src/snli_nli/`. Each module has a single responsibility so you can import only what you need.

```text
src/snli_nli/
├── __init__.py          # Version and label constants (no PyTorch import)
├── __main__.py          # python -m snli_nli
├── constants.py         # Label map, PAD/UNK tokens
├── data.py              # Load SNLI, filter labels, pair text, subsample
├── eda.py               # Class-balance and length plots
├── reporting.py         # Classification report + confusion matrix files
├── baseline.py          # TF-IDF + logistic regression
├── tokenization.py      # tokenize, vocabulary, encode_sentence
├── dataset.py           # SNLIDataset
├── model.py             # SNLIModel (LightningModule)
├── evaluate.py          # Validation-loop predictions
├── train_neural.py      # Lightning Trainer orchestration
└── cli.py               # argparse entry point
```

## Data flow

```text
load_snli_frames()
        │
        ▼
  clean_split()          drop label == -1, add label_text
        │
        ├── eda.run_eda()
        │
        ├── baseline.train_and_evaluate_baseline()
        │         uses data.add_pair_text + reporting
        │
        └── train_neural.train_and_evaluate_neural()
                  tokenize_pairs → build_vocab (train only)
                  SNLIDataset → DataLoader
                  SNLIModel + Trainer
                  evaluate.collect_predictions → reporting
```

## Import examples

Load and clean data without training:

```python
from snli_nli.data import load_snli_frames, clean_split

frames = load_snli_frames()
train = clean_split(frames["train"])
```

Train the classical model:

```python
from pathlib import Path
from snli_nli.baseline import train_and_evaluate_baseline

train_and_evaluate_baseline(train, validation, Path("outputs/baseline"))
```

Use the tokenizer on a single sentence:

```python
from snli_nli.tokenization import tokenize

tokenize("A person on a horse jumps over a broken down airplane.")
```

## Design choices

- **`src/` layout** — installing the project (`pip install -e .`) puts `snli_nli` on `PYTHONPATH` without mixing package code with docs.
- **No heavy imports in `__init__.py`** — importing `snli_nli.data` does not load Torch.
- **Vectorizer and vocabulary fit on training data only** — unlike the original Colab script, IDF and the word index are not computed on the validation rows.
- **`encode_sentence` returns the padded list** — the Colab helper built the list but did not return it; the Dataset class is the source of truth.
- **`notebooks/ml_sublime.py`** — original Colab export, kept for reference. The package is what you should run.
