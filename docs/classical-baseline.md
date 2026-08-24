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

Default package settings (`src/snli_nli/baseline.py`):

| Setting | Value |
| --- | --- |
| `max_features` | 30,000 |
| `ngram_range` | (1, 3) |
| `class_weight` | `"balanced"` |

The vectorizer is **fitted on the training split only**, then applied to validation with `transform`. The original Colab script fitted TF-IDF on the full training table before splitting, which leaks validation statistics into IDF.

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
