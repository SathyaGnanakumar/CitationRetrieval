"""
Data loader for ScholarCopilot citation retrieval dataset.

Handles loading, preprocessing, and splitting the dataset.
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import random


@dataclass
class CitationExample:
    """Single citation prediction example"""
    query_id: str
    citation_context: str
    true_title: str
    true_abstract: Optional[str]
    true_citation_key: str
    paper_id: str
    corpus_entries: List[Dict]  # All possible citations for this paper


@dataclass
class CitationCorpus:
    """Corpus of all possible citations"""
    entries: List[Dict]  # List of {title, abstract, citation_key}
    title_to_idx: Dict[str, int]

    def __len__(self):
        return len(self.entries)


class CitationDataLoader:
    """Load and preprocess ScholarCopilot citation data"""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to ScholarCopilot JSON file
        """
        self.data_path = Path(data_path)
        self.dataset = None
        self.examples = []
        self.corpus = None

    def load_data(self) -> List[Dict]:
        """Load raw JSON data"""
        print(f"📂 Loading dataset from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        print(f"✅ Loaded {len(self.dataset)} papers")
        return self.dataset

    def extract_citation_contexts(self, paper_text: str) -> List[Tuple[str, str]]:
        """
        Extract citation contexts from paper text.

        Args:
            paper_text: Full paper text with <|cite_X|> tags

        Returns:
            List of (cite_tag, context) tuples
        """
        contexts = []

        # Find all citation tags
        for match in re.finditer(r"<\|cite_(\d+)\|>", paper_text):
            cite_idx = match.group(1)
            cite_tag = f"<|cite_{cite_idx}|>"

            # Extract context (100 chars before and after)
            start, end = match.span()
            context = paper_text[max(0, start - 100):min(len(paper_text), end + 100)]

            contexts.append((cite_tag, context))

        return contexts

    def build_corpus_from_paper(self, paper: Dict) -> List[Dict]:
        """
        Build corpus of citations from a single paper's bibliography.

        Args:
            paper: Paper dict with bib_info

        Returns:
            List of corpus entries
        """
        corpus_entries = []
        bib = paper.get("bib_info", {})

        for cite_tag, refs in bib.items():
            for ref in refs:
                title = ref.get("title", "") or ""
                abstract = ref.get("abstract", "") or ""
                citation_key = ref.get("citation_key", cite_tag)

                if title:  # Only add if has title
                    corpus_entries.append({
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "citation_key": citation_key,
                        "text": (title + " " + abstract).strip()
                    })

        return corpus_entries

    def build_global_corpus(self) -> CitationCorpus:
        """
        Build global corpus from all papers.

        Returns:
            CitationCorpus object
        """
        print("🧱 Building global reference corpus...")

        all_entries = []
        seen_titles = set()

        for paper in self.dataset:
            corpus_entries = self.build_corpus_from_paper(paper)

            # Deduplicate by title
            for entry in corpus_entries:
                title_lower = entry["title"].lower()
                if title_lower not in seen_titles and entry["title"]:
                    seen_titles.add(title_lower)
                    all_entries.append(entry)

        # Build title index
        title_to_idx = {
            entry["title"].lower(): idx
            for idx, entry in enumerate(all_entries)
        }

        self.corpus = CitationCorpus(
            entries=all_entries,
            title_to_idx=title_to_idx
        )

        print(f"✅ Built corpus with {len(all_entries)} unique references")
        return self.corpus

    def extract_examples(self) -> List[CitationExample]:
        """
        Extract all citation prediction examples from dataset.

        Returns:
            List of CitationExample objects
        """
        if self.dataset is None:
            self.load_data()

        print("🔍 Extracting citation examples...")
        examples = []
        example_id = 0

        for paper_idx, paper in enumerate(self.dataset):
            paper_text = paper.get("paper", "")
            bib = paper.get("bib_info", {})
            paper_id = f"paper_{paper_idx}"

            # Build corpus for this paper
            corpus_entries = self.build_corpus_from_paper(paper)

            # Extract citation contexts
            contexts = self.extract_citation_contexts(paper_text)

            for cite_tag, context in contexts:
                # Get ground truth citation
                refs = bib.get(cite_tag, [])
                if not refs:
                    continue

                true_ref = refs[0]  # Take first reference
                true_title = true_ref.get("title", "")
                if not true_title:
                    continue

                example = CitationExample(
                    query_id=f"q_{example_id}",
                    citation_context=context,
                    true_title=true_title.strip(),
                    true_abstract=true_ref.get("abstract", ""),
                    true_citation_key=true_ref.get("citation_key", cite_tag),
                    paper_id=paper_id,
                    corpus_entries=corpus_entries
                )
                examples.append(example)
                example_id += 1

        self.examples = examples
        print(f"✅ Extracted {len(examples)} citation examples")
        return examples

    def create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[CitationExample]]:
        """
        Split dataset into train/val/test.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed for reproducibility

        Returns:
            Dict with 'train', 'val', 'test' splits
        """
        if not self.examples:
            self.extract_examples()

        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # Shuffle with seed
        random.seed(seed)
        shuffled = self.examples.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }

        print(f"\n📊 Dataset splits:")
        print(f"   Train: {len(splits['train'])} examples")
        print(f"   Val:   {len(splits['val'])} examples")
        print(f"   Test:  {len(splits['test'])} examples")

        return splits

    def save_splits(
        self,
        splits: Dict[str, List[CitationExample]],
        output_dir: str
    ):
        """
        Save splits to JSON files.

        Args:
            splits: Dict of train/val/test splits
            output_dir: Directory to save splits
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, examples in splits.items():
            output_file = output_path / f"{split_name}.json"

            # Convert to serializable format
            data = [
                {
                    'query_id': ex.query_id,
                    'citation_context': ex.citation_context,
                    'true_title': ex.true_title,
                    'true_abstract': ex.true_abstract,
                    'true_citation_key': ex.true_citation_key,
                    'paper_id': ex.paper_id,
                    'num_candidates': len(ex.corpus_entries)
                }
                for ex in examples
            ]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved {split_name} split to {output_file}")

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.examples:
            self.extract_examples()

        contexts_per_paper = {}
        for ex in self.examples:
            contexts_per_paper[ex.paper_id] = contexts_per_paper.get(ex.paper_id, 0) + 1

        corpus_sizes = [len(ex.corpus_entries) for ex in self.examples]

        stats = {
            'total_examples': len(self.examples),
            'total_papers': len(contexts_per_paper),
            'avg_citations_per_paper': sum(contexts_per_paper.values()) / len(contexts_per_paper),
            'avg_corpus_size': sum(corpus_sizes) / len(corpus_sizes),
            'min_corpus_size': min(corpus_sizes),
            'max_corpus_size': max(corpus_sizes)
        }

        return stats


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Load and split citation dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/scholar_copilot_eval_data_1k.json",
        help="Path to ScholarCopilot JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load data
    loader = CitationDataLoader(args.data_path)
    loader.load_data()
    loader.extract_examples()

    # Show statistics
    stats = loader.get_statistics()
    print("\n📈 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

    # Create and save splits
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    splits = loader.create_splits(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        seed=args.seed
    )
    loader.save_splits(splits, args.output_dir)

    print("\n✅ Dataset preparation complete!")


if __name__ == "__main__":
    main()
