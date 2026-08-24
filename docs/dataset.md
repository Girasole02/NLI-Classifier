# Dataset: Stanford Natural Language Inference (SNLI)

This project uses the **[SNLI corpus](https://nlp.stanford.edu/projects/snli/)**, loaded through Hugging Face Datasets as `snli`. SNLI is a large collection of English sentence pairs labeled for **natural language inference** (also called textual entailment).

## Task definition

Each example is a pair of sentences:

| Field | Role |
| --- | --- |
| `premise` | The starting sentence, treated as given. |
| `hypothesis` | A second sentence to be judged against the premise. |
| `label` | The relation between the two sentences. |

The model must decide which of the following holds:

| Numeric label | Name | Meaning |
| --- | --- | --- |
| `0` | **entailment** | The hypothesis is true if the premise is true. |
| `1` | **neutral** | The hypothesis may or may not be true given the premise. |
| `2` | **contradiction** | The hypothesis cannot be true if the premise is true. |

Example (illustrative):

- Premise: *A man is playing a guitar on stage.*
- Hypothesis: *Someone is performing music.* → **entailment**
- Hypothesis: *The man is wearing a red shirt.* → **neutral** (possible, not guaranteed)
- Hypothesis: *The man is sleeping in bed.* → **contradiction**

## Splits used in this project

The Hugging Face `snli` dataset is split into:

- **train** — used for exploratory analysis, vocabulary building, and model fitting
- **validation** — official development split (loaded in the notebook; the classical and neural pipelines also create an internal 80/20 split from the filtered training table)
- **test** — official held-out split (loaded, not used as the primary reported validation in the original notebook)

## Label cleaning

SNLI includes a small number of examples whose gold label is **`-1`**. Those rows are **unannotated / discarded** cases in the original corpus. In this project they are removed by keeping only labels in `{0, 1, 2}`.

After mapping numbers to names, missing `label_text` values correspond to those invalid labels. Filtering them is a required preprocessing step before training.

## Why SNLI is a good teaching dataset

- The task is a standard NLP benchmark, so results are easy to discuss.
- Classes are large and, after filtering, **roughly balanced**.
- Sentence pairs are short enough for classical TF-IDF models and for a modest BiLSTM.
- The same data can be used to compare a **linear baseline** with a **neural encoder**.

## Citation

If you use this dataset in a paper or report, cite the original SNLI paper:

> Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015).  
> *A large annotated corpus for learning natural language inference.*  
> Proceedings of EMNLP.

Dataset card on Hugging Face: [https://huggingface.co/datasets/snli](https://huggingface.co/datasets/snli)
