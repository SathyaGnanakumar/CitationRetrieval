"""Data loader for citation retrieval dataset."""

import json
from typing import List, Tuple, Set
from src.utils.text_processing import normalize_title, clean_text, extract_contexts


class CitationDataLoader:
    """Loads and processes citation retrieval dataset."""

    def __init__(self, data_path: str):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the JSON dataset file
        """
        self.data_path = data_path
        self.data = None
        self.corpus_ids = []
        self.corpus_titles = []
        self.corpus_texts = []
        self.query_contexts = []
        self.query_true_titles = []

    def load_dataset(self) -> List[dict]:
        """
        Load the dataset from JSON file.

        Returns:
            List of paper dictionaries
        """
        print("📂 Loading dataset...")
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"✅ Loaded {len(self.data)} papers")
        return self.data

    def build_corpus(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Build deduplicated corpus from bib_info references.

        Returns:
            Tuple of (corpus_ids, corpus_titles, corpus_texts)
        """
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("🧱 Building + deduplicating reference corpus...")
        seen_titles: Set[str] = set()

        for paper in self.data:
            for refs in paper.get("bib_info", {}).values():
                for ref in refs:
                    title = ref.get("title", "") or ""
                    abstract = ref.get("abstract", "") or ""
                    if not title and not abstract:
                        continue

                    ntitle = normalize_title(title)
                    if not ntitle or ntitle in seen_titles:
                        continue

                    seen_titles.add(ntitle)
                    self.corpus_ids.append(ref.get("citation_key") or ntitle)
                    self.corpus_titles.append(title.strip())
                    text = (title + ". " + abstract).strip() if abstract else title
                    self.corpus_texts.append(text)

        print(f"✅ Corpus entries after dedup: {len(self.corpus_texts)}")
        return self.corpus_ids, self.corpus_titles, self.corpus_texts

    def extract_query_contexts(self) -> Tuple[List[str], List[str]]:
        """
        Extract citation contexts and ground truth titles from papers.

        Returns:
            Tuple of (query_contexts, query_true_titles)
        """
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("🔎 Extracting citation contexts...")
        for paper in self.data:
            paper_text = clean_text(paper.get("paper", ""))
            bib = paper.get("bib_info", {})

            for tag, ctx in extract_contexts(paper_text):
                refs = bib.get(tag, [])
                if not refs:
                    continue
                true_title = refs[0].get("title", "")
                if not true_title:
                    continue
                self.query_contexts.append(ctx)
                self.query_true_titles.append(true_title)

        print(f"✅ Extracted {len(self.query_contexts)} contexts")
        return self.query_contexts, self.query_true_titles

    def get_corpus(self) -> Tuple[List[str], List[str], List[str]]:
        """Get the corpus data (IDs, titles, texts)."""
        return self.corpus_ids, self.corpus_titles, self.corpus_texts

    def get_queries(self) -> Tuple[List[str], List[str]]:
        """Get the query data (contexts, true titles)."""
        return self.query_contexts, self.query_true_titles
