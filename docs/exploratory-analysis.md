# Exploratory data analysis

This page describes the analysis performed on the **filtered SNLI training set** in `notebooks/ml_sublime.py`. Plots are generated interactively (Seaborn / Matplotlib) when the notebook is run.

## Class balance

After dropping invalid labels, the notebook counts examples per `label_text` and draws a bar chart.

**Expected qualitative result:** the three classes (entailment, neutral, contradiction) are **well balanced**. That matters because:

- accuracy is a reasonable first metric (it is not dominated by a majority class);
- class-weighted logistic regression is an optional refinement, not a requirement to “fix” extreme imbalance.

## Sentence length

Two columns are added:

- `premise_len` — number of whitespace-separated tokens in the premise
- `hypothesis_len` — same for the hypothesis

Summary statistics (`describe`) and three plots are produced:

1. Histogram of premise length  
2. Histogram of hypothesis length  
3. Side-by-side boxplot of the two length distributions

**Qualitative findings recorded in the original notebook:**

- Premises are **longer on average** than hypotheses.
- Some sentences exceed **70 words**.
- Lengths are **roughly unimodal** (described as “fairly normal” in the original notes).

These observations justify later modeling choices:

- TF-IDF with unigrams and bigrams (and later trigrams) can capture short overlapping phrases.
- The neural encoder uses a **fixed maximum length** with padding and truncation, which is necessary because LSTM batches need aligned tensors.

## Missing values

A null check is run on the training frame. Invalid labels (`-1`) surface as missing `label_text` after the first mapping step. Filtering to `{0, 1, 2}` removes that issue.
