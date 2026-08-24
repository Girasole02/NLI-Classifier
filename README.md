# SNLI Natural Language Inference Classifier

Train and compare two approaches to **natural language inference** on the [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) corpus:

1. **Classical baseline** — TF-IDF n-grams + logistic regression  
2. **Neural model** — bidirectional LSTM sentence pair encoder, trained with PyTorch Lightning  

Given a **premise** and a **hypothesis**, the model predicts one of three relations: **entailment**, **neutral**, or **contradiction**.

The executable source is an unmodified Google Colab export. Documentation in this repository explains the dataset, analysis, and modeling choices in English.

---

## Table of contents

- [What this project does](#what-this-project-does)
- [Repository layout](#repository-layout)
- [How the pipeline is organized](#how-the-pipeline-is-organized)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Dataset](#dataset)
- [Models](#models)
- [Further reading](#further-reading)
- [License](#license)

---

## What this project does

The notebook script:

- loads SNLI (`train` / `validation` / `test`) from Hugging Face Datasets;
- maps numeric labels to readable class names;
- removes invalid labels (`-1`);
- inspects class balance and sentence length;
- trains a TF-IDF + logistic regression baseline (including a larger n-gram setting and optional class weights);
- builds a word vocabulary, encodes sentence pairs, and trains a BiLSTM classifier;
- reports precision, recall, F1, and confusion matrices on held-out splits created from the training table.

It does **not** currently evaluate on the official SNLI test split as the main reported loop; validation metrics come from stratified 80/20 splits of the filtered training data (see [docs](docs/classical-baseline.md)).

---

## Repository layout

```text
snli-nli-classifier/
├── README.md                 # This file
├── LICENSE                   # MIT
├── requirements.txt          # Python dependencies
├── .gitignore
├── notebooks/
│   └── ml_sublime.py         # Original Colab script (code unchanged)
└── docs/
    ├── dataset.md            # SNLI task, labels, citation
    ├── exploratory-analysis.md
    ├── classical-baseline.md # TF-IDF + logistic regression
    └── neural-model.md       # BiLSTM + Lightning
```

| Path | Purpose |
| --- | --- |
| `notebooks/ml_sublime.py` | Full experiment as exported from Colab. **Do not treat this as a refactored library** — run it as a notebook or paste it into Colab. |
| `docs/` | Standalone explanations written for GitHub readers (no code changes). |

---

## How the pipeline is organized

The original file is a **linear notebook**, not a package. Sections appear in this order:

| Stage | What happens |
| --- | --- |
| **Load** | `load_dataset("snli")` → pandas frames for train, validation, and test |
| **Labels** | `{0,1,2}` → `entailment` / `neutral` / `contradiction`; drop `-1` |
| **EDA** | class counts, premise/hypothesis word-length histograms and boxplots |
| **Baseline** | concatenate sentences with `[SEP]`, TF-IDF, logistic regression, metrics |
| **Neural** | tokenize, build vocab, `SNLIDataset` + DataLoaders, `SNLIModel`, Lightning `Trainer` |
| **Evaluate** | classification report and confusion-matrix heatmap on the neural validation loader |
| **Early stopping** | second training run monitoring `val_loss` |

Detailed write-ups:

- [Dataset](docs/dataset.md)
- [Exploratory analysis](docs/exploratory-analysis.md)
- [Classical baseline](docs/classical-baseline.md)
- [Neural model](docs/neural-model.md)

---

## Requirements

- Python 3.9+ recommended  
- Optional but strongly recommended for the LSTM: a **CUDA GPU**  
- Network access on first run (SNLI download)

Install locally:

```bash
pip install -r requirements.txt
```

The script also contains Colab-style `!pip install` lines (`datasets`, `transformers`, `pytorch_lightning`, `tqdm`). Those work inside **Google Colab** and Jupyter magics; they are **not** valid in a standard `python notebooks/ml_sublime.py` invocation.

---

## How to run

### Option A — Google Colab (closest to the original)

1. Create a new Colab notebook.  
2. Open `notebooks/ml_sublime.py` and copy the cells (each `"""..."""` block is a markdown/comment cell from the export).  
3. Runtime → GPU if you want faster LSTM training.  
4. Run all cells from top to bottom.

### Option B — Jupyter

Convert or paste the file into a notebook environment that accepts `!pip` magics, then run sequentially. The script uses `tqdm.notebook`.

### Option C — Local Python

`python notebooks/ml_sublime.py` will **fail** on `!pip install` because that syntax is IPython/Colab-only. Use Colab/Jupyter, or run equivalent `pip install` commands yourself and execute only the Python portions in an interactive session.

First run downloads SNLI; subsequent runs can reuse the Hugging Face cache.

---

## Dataset

**SNLI** (Bowman et al., 2015) is a crowd-annotated corpus of English premise–hypothesis pairs.

| Label | Meaning |
| --- | --- |
| Entailment | Hypothesis follows from the premise |
| Neutral | Hypothesis is neither guaranteed nor contradicted |
| Contradiction | Hypothesis conflicts with the premise |

See [docs/dataset.md](docs/dataset.md) for splits, the `-1` label, and citation.

---

## Models

### TF-IDF + logistic regression

Sparse n-gram features over the concatenated pair. Fast CPU baseline; later cells increase vocabulary size and n-gram order, then try `class_weight="balanced"`.

### Bidirectional LSTM (PyTorch Lightning)

Shared embedding + BiLSTM over premise and hypothesis; concatenated hidden states go to a 3-way linear classifier. Trained with Adam and, in a later cell, early stopping on validation loss.

---

## Further reading

- SNLI project page: [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)  
- Hugging Face dataset: [https://huggingface.co/datasets/snli](https://huggingface.co/datasets/snli)  
- Original Colab (from the file header): [ML SUBLIME notebook](https://colab.research.google.com/drive/1lpkP8Fvu5j2lbW1OAO8EcU8FkeOg5LXm)

---

## License

This repository is released under the [MIT License](LICENSE). SNLI is a separate dataset with its own terms; follow the Stanford / Hugging Face dataset licenses when redistributing data.
