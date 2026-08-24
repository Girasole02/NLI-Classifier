# Neural model: bidirectional LSTM with PyTorch Lightning

The second half of the notebook trains a **pair encoder**: each sentence is embedded, passed through a shared bidirectional LSTM, then the two sentence vectors are concatenated and classified into three labels.

## Tokenization and vocabulary

1. Text is lowercased and split on whitespace.
2. Surrounding punctuation is stripped from each token.
3. All tokens from premises and hypotheses are counted.
4. A vocabulary maps tokens to integer IDs, with two special symbols:
   - `<PAD>` (id `0`) — padding shorter sentences to a fixed length
   - `<UNK>` (id `1`) — tokens not in the vocabulary at inference time

## Encoding and batching

`SNLIDataset` (`src/snli_nli/dataset.py`) converts token lists to fixed-length index sequences (`max_len=50` by default). `encode_sentence` in `tokenization.py` **returns** the padded list (the standalone helper in the Colab export built the list but did not return it).

Batches of size **64** are built with PyTorch `DataLoader`. Vocabulary is built from **training tokens only**.

## Architecture (`SNLIModel`)

For a batch of premises and hypotheses:

1. `nn.Embedding` looks up a vector per token (`padding_idx=0`).
2. A **bidirectional LSTM** reads each sentence.
3. Forward and backward hidden states are concatenated for the premise and for the hypothesis.
4. The two sentence representations are concatenated (`hidden_dim * 4` features).
5. A linear layer maps that vector to **3 logits**.

Loss is **cross-entropy**. The optimizer is **Adam** (`lr=1e-3`).

PyTorch Lightning wraps training (`training_step`, `validation_step`, logging of loss and accuracy). Device selection uses CUDA when available.

## Training schedule in the notebook

Two `Trainer` runs appear in sequence:

1. **Short run:** `max_epochs=5`, `accelerator="auto"`.
2. **Longer run with early stopping:** `max_epochs=20`, `EarlyStopping` on `val_loss` with patience 5.

After the first fit, predictions on the validation loader are collected to print a classification report and a Seaborn heatmap of the confusion matrix.

## Practical notes

- Training is much faster on a **GPU**.
- Hugging Face downloads SNLI on first run.
- Run `snli-nli neural` or `python -m snli_nli neural` from the installed package.
