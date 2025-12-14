"""Prepare DSPy training splits from correct/wrong JSONL files."""

from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent
DEFAULT_CORRECT = PROJECT_ROOT / "datasets" / "correct.jsonl"
DEFAULT_WRONG = PROJECT_ROOT / "datasets" / "wrong.jsonl"
OUTPUT_DIR = DATA_DIR / "data"


def load_examples(correct_path: Path, wrong_path: Path) -> list[dict]:
    examples: dict[str, dict] = {}

    # Load positives
    with correct_path.open("r", encoding="utf-8") as correct_file:
        for line in correct_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            example_id = record.get("example_id")
            if not example_id:
                continue

            guess = record.get("guess") or {}
            positive_title = (guess.get("title") or "").strip()
            if not positive_title:
                # Skip examples without a positive title
                continue

            examples[example_id] = {
                "example_id": example_id,
                "paper_id": record.get("paper_id"),
                "citation_context": record.get("citation_context"),
                "positive_title": positive_title,
                "positive_abstract": guess.get("abstract") or "",
                "negatives": [],
            }

    # Load negatives
    missing_positive = set()
    with wrong_path.open("r", encoding="utf-8") as wrong_file:
        for line in wrong_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            example_id = record.get("example_id")
            if not example_id or example_id not in examples:
                missing_positive.add(example_id or "<missing>")
                continue

            guess = record.get("guess") or {}
            title = (guess.get("title") or "").strip()
            if not title:
                continue

            negative = {
                "title": title,
                "abstract": guess.get("abstract") or "",
            }

            existing_titles = {neg["title"] for neg in examples[example_id]["negatives"]}
            if negative["title"] not in existing_titles:
                examples[example_id]["negatives"].append(negative)

    filtered = [ex for ex in examples.values() if ex["negatives"]]
    if not filtered:
        raise RuntimeError("No examples with both positives and negatives were found.")

    if missing_positive:
        print(f"Warning: {len(missing_positive)} negatives without matching positives (ignored)")

    return filtered


def split_examples(examples: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    random.Random(seed).shuffle(examples)

    n = len(examples)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)

    return {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }


def save_splits(splits: dict[str, list[dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(rows):5d} examples to {path}")


def main() -> None:
    correct_path = DEFAULT_CORRECT
    wrong_path = DEFAULT_WRONG

    print(f"Loading positives from {correct_path}")
    print(f"Loading negatives from {wrong_path}")

    examples = load_examples(correct_path, wrong_path)
    print(f"Loaded {len(examples)} examples with negatives")

    splits = split_examples(examples)
    save_splits(splits, OUTPUT_DIR)


if __name__ == "__main__":
    main()
