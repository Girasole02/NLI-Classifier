"""TF-IDF n-grams + logistic regression baseline."""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from snli_nli.data import add_pair_text
from snli_nli.reporting import save_classification_artifacts


def train_and_evaluate_baseline(
    train,
    validation,
    output_dir: Path,
    max_features: int = 30000,
    ngram_range: tuple[int, int] = (1, 3),
    class_weight: str | None = "balanced",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train = add_pair_text(train)
    validation = add_pair_text(validation)

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(validation["text"])
    y_train = train["label"].to_numpy()
    y_val = validation["label"].to_numpy()

    model = LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        class_weight=class_weight,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    report = save_classification_artifacts(
        y_val,
        y_pred,
        output_dir,
        stem="baseline",
        title="Baseline confusion matrix",
    )
    print("Classification report (TF-IDF + logistic regression)")
    print(report)
    print(f"Saved baseline metrics to {output_dir}")
