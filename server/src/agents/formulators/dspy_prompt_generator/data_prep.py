"""
Data preparation for DSPy prompt generator.

Extracts positives and negatives from ScholarCopilot dataset
for contrastive training with DSPy.
"""

import json
import re
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple


@dataclass
class CitationTrainingExample:
    """A single training example for DSPy with positive and negatives."""
    
    example_id: str
    citation_context: str  # The text surrounding the citation marker
    
    # Positive (ground truth)
    positive_title: str
    positive_abstract: str
    
    # Negatives (other papers from same bibliography - hard negatives)
    negatives: List[Dict[str, str]]  # List of {title, abstract}
    
    # Metadata
    paper_id: str
    cite_tag: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_candidate_pool(self, shuffle: bool = True) -> List[Dict[str, str]]:
        """
        Get all candidates (positive + negatives) as a pool.
        
        Returns:
            List of {title, abstract, is_positive} dicts
        """
        candidates = [
            {
                "title": self.positive_title,
                "abstract": self.positive_abstract,
                "is_positive": True
            }
        ]
        
        for neg in self.negatives:
            candidates.append({
                "title": neg["title"],
                "abstract": neg["abstract"],
                "is_positive": False
            })
        
        if shuffle:
            random.shuffle(candidates)
        
        return candidates


class CitationDataPrep:
    """Prepare ScholarCopilot data for DSPy training."""
    
    def __init__(
        self,
        data_path: str = "datasets/scholar_copilot_eval_data_1k.json",
        context_window: int = 200,  # chars before and after citation
        max_negatives: int = 10,    # max negatives per example
        min_context_length: int = 50,  # minimum context length
    ):
        self.data_path = Path(data_path)
        self.context_window = context_window
        self.max_negatives = max_negatives
        self.min_context_length = min_context_length
        self.raw_data = None
        self.examples: List[CitationTrainingExample] = []
    
    def load_raw_data(self) -> List[Dict]:
        """Load the raw JSON dataset."""
        print(f" Loading data from {self.data_path}...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f" Loaded {len(self.raw_data)} papers")
        return self.raw_data
    
    def _extract_context(self, text: str, cite_tag: str) -> Optional[str]:
        """
        Extract citation context around a specific citation tag.
        
        Args:
            text: Full paper text
            cite_tag: Citation tag like <|cite_0|>
            
        Returns:
            Context string or None if not found
        """
        # Find the citation tag
        pattern = re.escape(cite_tag)
        match = re.search(pattern, text)
        
        if not match:
            return None
        
        start, end = match.span()
        
        # Extract context window
        context_start = max(0, start - self.context_window)
        context_end = min(len(text), end + self.context_window)
        
        context = text[context_start:context_end]
        
        # Clean up: replace the cite tag with [CITATION] marker
        context = context.replace(cite_tag, "[CITATION]")
        
        # Remove other citation tags for cleaner context
        context = re.sub(r'<\|cite_\d+\|>', '', context)
        
        # Clean whitespace
        context = ' '.join(context.split())
        
        if len(context) < self.min_context_length:
            return None
            
        return context
    
    def _build_negatives(
        self,
        bib_info: Dict,
        exclude_tag: str
    ) -> List[Dict[str, str]]:
        """
        Build list of negative examples from bibliography.
        
        Args:
            bib_info: Bibliography dictionary
            exclude_tag: Citation tag to exclude (the positive)
            
        Returns:
            List of {title, abstract} dicts
        """
        negatives = []
        
        for tag, refs in bib_info.items():
            if tag == exclude_tag:
                continue
            
            for ref in refs:
                title = ref.get("title", "").strip()
                abstract = ref.get("abstract", "").strip()
                
                # Skip if no title
                if not title:
                    continue
                
                # Clean title (remove leading { if present)
                title = title.lstrip('{').rstrip('}')
                
                negatives.append({
                    "title": title,
                    "abstract": abstract or ""
                })
        
        # Shuffle and limit
        random.shuffle(negatives)
        return negatives[:self.max_negatives]
    
    def extract_examples(self) -> List[CitationTrainingExample]:
        """
        Extract all citation examples with positives and negatives.
        
        Returns:
            List of CitationTrainingExample objects
        """
        if self.raw_data is None:
            self.load_raw_data()
        
        print("🔍 Extracting citation examples with positives/negatives...")
        
        example_id = 0
        skipped = {"no_context": 0, "no_positive": 0, "no_negatives": 0}
        
        for paper_idx, paper in enumerate(self.raw_data):
            paper_id = paper.get("paper_id", f"paper_{paper_idx}")
            paper_text = paper.get("paper", "")
            bib_info = paper.get("bib_info", {})
            
            # Process each citation in the paper
            for cite_tag, refs in bib_info.items():
                # Get the positive (ground truth citation)
                if not refs:
                    skipped["no_positive"] += 1
                    continue
                
                positive_ref = refs[0]
                positive_title = positive_ref.get("title", "").strip()
                positive_abstract = positive_ref.get("abstract", "").strip()
                
                if not positive_title:
                    skipped["no_positive"] += 1
                    continue
                
                # Clean title
                positive_title = positive_title.lstrip('{').rstrip('}')
                
                # Extract context
                context = self._extract_context(paper_text, cite_tag)
                if not context:
                    skipped["no_context"] += 1
                    continue
                
                # Build negatives
                negatives = self._build_negatives(bib_info, cite_tag)
                if not negatives:
                    skipped["no_negatives"] += 1
                    continue
                
                # Create example
                example = CitationTrainingExample(
                    example_id=f"ex_{example_id}",
                    citation_context=context,
                    positive_title=positive_title,
                    positive_abstract=positive_abstract or "",
                    negatives=negatives,
                    paper_id=paper_id,
                    cite_tag=cite_tag
                )
                
                self.examples.append(example)
                example_id += 1
        
        print(f" Extracted {len(self.examples)} training examples")
        print(f" Skipped: {skipped}")
        
        return self.examples
    
    def create_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[CitationTrainingExample]]:
        """
        Split examples into train/val/test sets.
        
        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed
            
        Returns:
            Dict with 'train', 'val', 'test' splits
        """
        if not self.examples:
            self.extract_examples()
        
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
    
    def save_examples(
        self,
        output_dir: str = "dspy_prompt_generator/data",
        splits: Optional[Dict] = None
    ):
        """
        Save examples to JSON files.
        
        Args:
            output_dir: Directory to save files
            splits: Optional pre-computed splits
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if splits is None:
            splits = self.create_splits()
        
        for split_name, examples in splits.items():
            output_file = output_path / f"{split_name}.json"
            
            data = [ex.to_dict() for ex in examples]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f" Saved {split_name} ({len(examples)} examples) to {output_file}")
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.examples:
            self.extract_examples()
        
        neg_counts = [len(ex.negatives) for ex in self.examples]
        context_lengths = [len(ex.citation_context) for ex in self.examples]
        
        stats = {
            "total_examples": len(self.examples),
            "avg_negatives_per_example": sum(neg_counts) / len(neg_counts),
            "min_negatives": min(neg_counts),
            "max_negatives": max(neg_counts),
            "avg_context_length": sum(context_lengths) / len(context_lengths),
            "examples_with_abstracts": sum(
                1 for ex in self.examples if ex.positive_abstract
            ),
        }
        
        return stats
    
    def preview_example(self, idx: int = 0):
        """Print a preview of an example for inspection."""
        if not self.examples:
            self.extract_examples()
        
        if idx >= len(self.examples):
            print(f"Index {idx} out of range. Max: {len(self.examples) - 1}")
            return
        
        ex = self.examples[idx]
        
        print("=" * 70)
        print(f"EXAMPLE {ex.example_id} (Paper: {ex.paper_id}, {ex.cite_tag})")
        print("=" * 70)
        
        print("\n CITATION CONTEXT:")
        print("-" * 40)
        print(ex.citation_context)
        
        print("\n POSITIVE (Ground Truth):")
        print("-" * 40)
        print(f"Title: {ex.positive_title}")
        print(f"Abstract: {ex.positive_abstract[:200]}..." if ex.positive_abstract else "Abstract: N/A")
        
        print(f"\nNEGATIVES ({len(ex.negatives)} hard negatives):")
        print("-" * 40)
        for i, neg in enumerate(ex.negatives[:3]):
            print(f"  {i+1}. {neg['title'][:60]}...")
        if len(ex.negatives) > 3:
            print(f"  ... and {len(ex.negatives) - 3} more")


def main():
    """Run data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for DSPy training")
    parser.add_argument(
        "--data-path",
        type=str,
        default="datasets/scholar_copilot_eval_data_1k.json",
        help="Path to ScholarCopilot JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dspy_prompt_generator/data",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=200,
        help="Characters before/after citation to include"
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=10,
        help="Maximum negatives per example"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview examples instead of saving"
    )
    
    args = parser.parse_args()
    
    # Prepare data
    prep = CitationDataPrep(
        data_path=args.data_path,
        context_window=args.context_window,
        max_negatives=args.max_negatives
    )
    
    prep.load_raw_data()
    prep.extract_examples()
    
    # Show statistics
    stats = prep.get_statistics()
    print("\n Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    if args.preview:
        # Preview a few examples
        print("\n" + "=" * 70)
        print("PREVIEW MODE - Showing sample examples")
        print("=" * 70)
        for i in range(min(3, len(prep.examples))):
            prep.preview_example(i)
            print()
    else:
        # Create splits and save
        splits = prep.create_splits()
        prep.save_examples(args.output_dir, splits)
        
        print("\n Data preparation complete!")
        print(f"   Files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

