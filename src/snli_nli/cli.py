"""Command-line entry point for EDA, baseline, and neural training."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from snli_nli.baseline import train_and_evaluate_baseline
from snli_nli.data import clean_split, load_snli_frames, maybe_subsample
from snli_nli.eda import run_eda
from snli_nli.train_neural import train_and_evaluate_neural


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SNLI natural language inference models.",
    )
    parser.add_argument(
        "command",
        choices=["eda", "baseline", "neural", "all"],
        help="Which stage to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for figures, reports, and checkpoints.",
    )
    parser.add_argument(
        "--split",
        choices=["official", "internal"],
        default="official",
        help="official = SNLI validation split; internal = 80/20 of filtered train.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional stratified cap on training rows (useful for a quick CPU run).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    return parser


def prepare_frames(args: argparse.Namespace):
    raw = load_snli_frames()
    train = clean_split(raw["train"])
    validation = clean_split(raw["validation"])
    test = clean_split(raw["test"])
    print(
        f"Filtered sizes — train: {len(train):,} | "
        f"validation: {len(validation):,} | test: {len(test):,}"
    )

    train = maybe_subsample(train, args.max_train_samples, args.seed)

    if args.split == "internal":
        train, validation = train_test_split(
            train,
            test_size=0.2,
            random_state=args.seed,
            stratify=train["label"],
        )
        print(f"Internal split — train: {len(train):,} | validation: {len(validation):,}")

    return train.reset_index(drop=True), validation.reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation = prepare_frames(args)

    if args.command in {"eda", "all"}:
        run_eda(train, args.output_dir / "eda")
    if args.command in {"baseline", "all"}:
        train_and_evaluate_baseline(train, validation, args.output_dir / "baseline")
    if args.command in {"neural", "all"}:
        train_and_evaluate_neural(
            train,
            validation,
            args.output_dir / "neural",
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
