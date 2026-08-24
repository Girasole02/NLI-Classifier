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

`SNLIDataset` converts token lists to fixed-length index sequences (`max_len=50` in the dataset class). A standalone helper `encode_sentence` earlier in the file uses `max_length=20`; the **DataLoader path uses the class method**, which is what training actually consumes.

Batches of size **64** are built with PyTorch `DataLoader`. The training table is again split **80/20**, stratified by label.

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

## Dependencies that are imported but not used for this LSTM

The original Colab file also installs / imports **Transformers** (`DistilBertTokenizerFast`, `DistilBertForSequenceClassification`, `Trainer`, `TrainingArguments`). Those objects are **not used** in the LSTM pipeline that follows. They are left untouched in the source file.

## Practical notes

- Training is much faster on a **GPU** (Colab with GPU runtime, or a local CUDA install).
- Hugging Face will **download SNLI** on first run; allow disk space and network access.
- `tqdm.notebook` is aimed at Jupyter / Colab. In a plain terminal you may prefer the standard `tqdm` progress bars (again: the source file is not changed here).
