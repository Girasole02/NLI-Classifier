# Classical baseline: TF-IDF and logistic regression

This is the first modeling approach in the original notebook: a **linear classifier** on a **sparse bag-of-ngrams** representation. It is intentionally simpler than the neural model and serves as a reference.

## Input construction

Premise and hypothesis are concatenated into a single string:

```text
premise + " [SEP] " + hypothesis
```

The `[SEP]` marker is a lightweight way to tell the vectorizer that the two sentences are distinct, without using a transformer tokenizer.

## Vectorization

`TfidfVectorizer` maps each concatenated text to a sparse vector:

- **TF** (term frequency) rewards words that appear often in that example.
- **IDF** (inverse document frequency) down-weights words that appear in many examples (for example function words).
- **n-grams** capture local phrases, not only single tokens.

Two configurations appear in the notebook:

| Stage | `max_features` | `ngram_range` | Notes |
| --- | --- | --- | --- |
| First run | 10,000 | (1, 2) | Unigrams and bigrams |
| Improved run | 30,000 | (1, 3) | Larger vocabulary, up to trigrams |

A cleaning experiment lowercases text and strips punctuation into `text_clean`, but the **second fit still transforms `df_train["text"]`** (the original concatenated field). The Python file is kept as-is; if you re-run experiments, you may want to compare `text` vs `text_clean` yourself.

## Classifier

`LogisticRegression` with:

- `max_iter=1000`
- `solver="saga"` (works well with large sparse TF-IDF matrices)
- `n_jobs=-1`

A further run sets `class_weight="balanced"` so that the loss pays more attention to any slightly rarer class.

## Train / validation split

The filtered training table is split **80% / 20%**, stratified by label, with `random_state=42` for reproducibility.

This split is **internal** to the training dataframe. It is not the official SNLI validation split loaded at the beginning of the notebook.

## Evaluation

The notebook prints:

- a **classification report** (precision, recall, F1 per class, plus aggregates)
- a **confusion matrix**

Typical reading of the matrix for this task:

- **entailment vs contradiction** confusions are often the most informative errors;
- **neutral** is frequently the hardest class, because it overlaps lexically with both other labels.

## Why keep this baseline

Even if a BiLSTM scores higher, TF-IDF + logistic regression:

- trains quickly on CPU;
- is easy to debug;
- shows how much of the task is solvable from **word overlap and n-gram cues** alone.
