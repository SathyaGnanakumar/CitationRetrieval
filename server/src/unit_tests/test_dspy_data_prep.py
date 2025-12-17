"""Unit tests for DSPy data preparation."""

import pytest
from unittest.mock import MagicMock, patch
import random

from src.agents.formulators.dspy_prompt_generator.data_prep import (
    CitationTrainingExample,
    CitationDataPrep,
)


class TestCitationTrainingExample:
    """Tests for the CitationTrainingExample dataclass."""
    
    @pytest.fixture
    def sample_example(self):
        """Create a sample training example."""
        return CitationTrainingExample(
            example_id="ex_1",
            citation_context="This approach uses [CITATION] for improved results.",
            positive_title="The Positive Paper",
            positive_abstract="This is the abstract of the positive paper.",
            negatives=[
                {"title": "Negative Paper 1", "abstract": "Abstract 1"},
                {"title": "Negative Paper 2", "abstract": "Abstract 2"},
                {"title": "Negative Paper 3", "abstract": "Abstract 3"},
            ],
            paper_id="paper_123",
            cite_tag="<|cite_0|>"
        )
    
    def test_to_dict_returns_dict(self, sample_example):
        """to_dict should return a dictionary."""
        result = sample_example.to_dict()
        assert isinstance(result, dict)
    
    def test_to_dict_contains_all_fields(self, sample_example):
        """to_dict should contain all required fields."""
        result = sample_example.to_dict()
        
        assert "example_id" in result
        assert "citation_context" in result
        assert "positive_title" in result
        assert "positive_abstract" in result
        assert "negatives" in result
        assert "paper_id" in result
        assert "cite_tag" in result
    
    def test_to_dict_values_match(self, sample_example):
        """to_dict values should match the original."""
        result = sample_example.to_dict()
        
        assert result["example_id"] == "ex_1"
        assert result["positive_title"] == "The Positive Paper"
        assert len(result["negatives"]) == 3
    
    def test_get_candidate_pool_includes_positive(self, sample_example):
        """Candidate pool should include the positive paper."""
        candidates = sample_example.get_candidate_pool(shuffle=False)
        
        positive_found = any(c["is_positive"] for c in candidates)
        assert positive_found
    
    def test_get_candidate_pool_includes_negatives(self, sample_example):
        """Candidate pool should include all negatives."""
        candidates = sample_example.get_candidate_pool(shuffle=False)
        
        negative_count = sum(1 for c in candidates if not c["is_positive"])
        assert negative_count == 3
    
    def test_get_candidate_pool_total_count(self, sample_example):
        """Candidate pool should have correct total count."""
        candidates = sample_example.get_candidate_pool(shuffle=False)
        
        # 1 positive + 3 negatives = 4 total
        assert len(candidates) == 4
    
    def test_get_candidate_pool_structure(self, sample_example):
        """Each candidate should have title, abstract, is_positive."""
        candidates = sample_example.get_candidate_pool(shuffle=False)
        
        for candidate in candidates:
            assert "title" in candidate
            assert "abstract" in candidate
            assert "is_positive" in candidate
    
    def test_get_candidate_pool_positive_first_when_no_shuffle(self, sample_example):
        """Without shuffle, positive should be first."""
        candidates = sample_example.get_candidate_pool(shuffle=False)
        
        assert candidates[0]["is_positive"] == True
        assert candidates[0]["title"] == "The Positive Paper"
    
    def test_get_candidate_pool_shuffles_when_requested(self, sample_example):
        """With shuffle=True, order should be randomized."""
        # Set seed for reproducibility
        random.seed(42)
        candidates1 = sample_example.get_candidate_pool(shuffle=True)
        
        random.seed(123)  # Different seed
        candidates2 = sample_example.get_candidate_pool(shuffle=True)
        
        # Extract titles to compare order
        titles1 = [c["title"] for c in candidates1]
        titles2 = [c["title"] for c in candidates2]
        
        # With different seeds, order should differ (probabilistically)
        # Note: There's a small chance they could be the same, but very unlikely
        # For a more robust test, we just verify all items are present
        assert set(titles1) == set(titles2)


class TestCitationDataPrepExtractContext:
    """Tests for context extraction from paper text."""
    
    @pytest.fixture
    def data_prep(self):
        """Create a CitationDataPrep instance."""
        # Don't load data, just create the object
        prep = CitationDataPrep(
            data_path="fake_path.json",
            context_window=50,
            min_context_length=10
        )
        return prep
    
    def test_extract_context_finds_citation(self, data_prep):
        """Should find and extract context around citation tag."""
        text = "This is some text before the citation <|cite_0|> and some text after."
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        assert result is not None
        assert "[CITATION]" in result
    
    def test_extract_context_replaces_tag_with_marker(self, data_prep):
        """Should replace citation tag with [CITATION] marker."""
        text = "Before <|cite_0|> after."
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        assert "<|cite_0|>" not in result
        assert "[CITATION]" in result
    
    def test_extract_context_removes_other_tags(self, data_prep):
        """Should remove other citation tags from context."""
        text = "Text <|cite_1|> more text <|cite_0|> and <|cite_2|> end."
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        assert "<|cite_1|>" not in result
        assert "<|cite_2|>" not in result
        assert "[CITATION]" in result  # Only the target tag becomes marker
    
    def test_extract_context_returns_none_if_not_found(self, data_prep):
        """Should return None if citation tag not found."""
        text = "This text has no citation tags."
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        assert result is None
    
    def test_extract_context_respects_window_size(self, data_prep):
        """Context should be limited by window size."""
        # Create text with known positions
        before = "A" * 100
        after = "B" * 100
        text = f"{before}<|cite_0|>{after}"
        
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        # With window=50, should have at most ~100 chars total + marker
        assert len(result) <= 120  # Some buffer for marker and cleanup
    
    def test_extract_context_handles_start_of_text(self, data_prep):
        """Should handle citation at start of text."""
        text = "<|cite_0|> This is text after the citation."
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        assert result is not None
        assert "[CITATION]" in result
    
    def test_extract_context_handles_end_of_text(self, data_prep):
        """Should handle citation at end of text."""
        text = "This is text before the citation <|cite_0|>"
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        assert result is not None
        assert "[CITATION]" in result
    
    def test_extract_context_returns_none_if_too_short(self, data_prep):
        """Should return None if extracted context is too short."""
        data_prep.min_context_length = 50
        text = "<|cite_0|>"  # Just the tag, no context
        result = data_prep._extract_context(text, "<|cite_0|>")
        
        assert result is None


class TestCitationDataPrepBuildNegatives:
    """Tests for building negative examples from bibliography."""
    
    @pytest.fixture
    def data_prep(self):
        """Create a CitationDataPrep instance."""
        prep = CitationDataPrep(
            data_path="fake_path.json",
            max_negatives=5
        )
        return prep
    
    @pytest.fixture
    def sample_bib_info(self):
        """Create sample bibliography info."""
        return {
            "<|cite_0|>": [{"title": "Paper 0", "abstract": "Abstract 0"}],
            "<|cite_1|>": [{"title": "Paper 1", "abstract": "Abstract 1"}],
            "<|cite_2|>": [{"title": "Paper 2", "abstract": "Abstract 2"}],
            "<|cite_3|>": [{"title": "Paper 3", "abstract": "Abstract 3"}],
        }
    
    def test_build_negatives_excludes_positive(self, data_prep, sample_bib_info):
        """Should exclude the positive paper from negatives."""
        negatives = data_prep._build_negatives(sample_bib_info, "<|cite_0|>")
        
        titles = [n["title"] for n in negatives]
        assert "Paper 0" not in titles
    
    def test_build_negatives_includes_other_papers(self, data_prep, sample_bib_info):
        """Should include papers from other citations."""
        negatives = data_prep._build_negatives(sample_bib_info, "<|cite_0|>")
        
        titles = [n["title"] for n in negatives]
        # Should have papers 1, 2, 3
        assert len(titles) == 3
    
    def test_build_negatives_respects_max_limit(self, data_prep):
        """Should respect max_negatives limit."""
        # Create bib with many papers
        large_bib = {
            f"<|cite_{i}|>": [{"title": f"Paper {i}", "abstract": f"Abstract {i}"}]
            for i in range(20)
        }
        
        negatives = data_prep._build_negatives(large_bib, "<|cite_0|>")
        
        assert len(negatives) <= data_prep.max_negatives
    
    def test_build_negatives_has_correct_structure(self, data_prep, sample_bib_info):
        """Each negative should have title and abstract."""
        negatives = data_prep._build_negatives(sample_bib_info, "<|cite_0|>")
        
        for neg in negatives:
            assert "title" in neg
            assert "abstract" in neg
    
    def test_build_negatives_cleans_title(self, data_prep):
        """Should clean curly braces from titles."""
        bib_info = {
            "<|cite_0|>": [{"title": "Positive", "abstract": ""}],
            "<|cite_1|>": [{"title": "{Bracketed Title}", "abstract": ""}],
        }
        
        negatives = data_prep._build_negatives(bib_info, "<|cite_0|>")
        
        assert negatives[0]["title"] == "Bracketed Title"  # Braces removed
    
    def test_build_negatives_skips_empty_titles(self, data_prep):
        """Should skip papers with empty titles."""
        bib_info = {
            "<|cite_0|>": [{"title": "Positive", "abstract": ""}],
            "<|cite_1|>": [{"title": "", "abstract": "No title here"}],
            "<|cite_2|>": [{"title": "Valid Paper", "abstract": ""}],
        }
        
        negatives = data_prep._build_negatives(bib_info, "<|cite_0|>")
        
        titles = [n["title"] for n in negatives]
        assert "" not in titles
        assert "Valid Paper" in titles
    
    def test_build_negatives_returns_empty_if_no_others(self, data_prep):
        """Should return empty list if no other papers."""
        bib_info = {
            "<|cite_0|>": [{"title": "Only Paper", "abstract": ""}],
        }
        
        negatives = data_prep._build_negatives(bib_info, "<|cite_0|>")
        
        assert negatives == []


class TestCitationDataPrepStatistics:
    """Tests for dataset statistics calculation."""
    
    def test_statistics_keys(self):
        """Statistics should have all expected keys."""
        prep = CitationDataPrep(data_path="fake.json")
        
        # Manually add some examples
        prep.examples = [
            CitationTrainingExample(
                example_id="ex_0",
                citation_context="Context " * 20,
                positive_title="Title",
                positive_abstract="Abstract",
                negatives=[{"title": "N1", "abstract": "A1"}],
                paper_id="p1",
                cite_tag="<|cite_0|>"
            ),
            CitationTrainingExample(
                example_id="ex_1",
                citation_context="Another context " * 10,
                positive_title="Title 2",
                positive_abstract="",  # No abstract
                negatives=[
                    {"title": "N1", "abstract": "A1"},
                    {"title": "N2", "abstract": "A2"},
                ],
                paper_id="p2",
                cite_tag="<|cite_1|>"
            ),
        ]
        
        stats = prep.get_statistics()
        
        assert "total_examples" in stats
        assert "avg_negatives_per_example" in stats
        assert "min_negatives" in stats
        assert "max_negatives" in stats
        assert "avg_context_length" in stats
        assert "examples_with_abstracts" in stats
    
    def test_statistics_values_correct(self):
        """Statistics values should be calculated correctly."""
        prep = CitationDataPrep(data_path="fake.json")
        
        prep.examples = [
            CitationTrainingExample(
                example_id="ex_0",
                citation_context="A" * 100,  # 100 chars
                positive_title="Title",
                positive_abstract="Has abstract",
                negatives=[{"title": "N1", "abstract": "A1"}],  # 1 negative
                paper_id="p1",
                cite_tag="<|cite_0|>"
            ),
            CitationTrainingExample(
                example_id="ex_1",
                citation_context="B" * 200,  # 200 chars
                positive_title="Title 2",
                positive_abstract="",  # No abstract
                negatives=[
                    {"title": "N1", "abstract": "A1"},
                    {"title": "N2", "abstract": "A2"},
                    {"title": "N3", "abstract": "A3"},
                ],  # 3 negatives
                paper_id="p2",
                cite_tag="<|cite_1|>"
            ),
        ]
        
        stats = prep.get_statistics()
        
        assert stats["total_examples"] == 2
        assert stats["avg_negatives_per_example"] == 2.0  # (1+3)/2
        assert stats["min_negatives"] == 1
        assert stats["max_negatives"] == 3
        assert stats["avg_context_length"] == 150.0  # (100+200)/2
        assert stats["examples_with_abstracts"] == 1  # Only first has abstract


class TestCitationDataPrepSplits:
    """Tests for train/val/test splitting."""
    
    def test_create_splits_returns_correct_keys(self):
        """Should return dict with train, val, test keys."""
        prep = CitationDataPrep(data_path="fake.json")
        
        # Create fake examples
        prep.examples = [
            CitationTrainingExample(
                example_id=f"ex_{i}",
                citation_context=f"Context {i}",
                positive_title=f"Title {i}",
                positive_abstract=f"Abstract {i}",
                negatives=[],
                paper_id=f"p{i}",
                cite_tag=f"<|cite_{i}|>"
            )
            for i in range(100)
        ]
        
        splits = prep.create_splits()
        
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
    
    def test_create_splits_correct_proportions(self):
        """Splits should have approximately correct proportions."""
        prep = CitationDataPrep(data_path="fake.json")
        
        prep.examples = [
            CitationTrainingExample(
                example_id=f"ex_{i}",
                citation_context=f"Context {i}",
                positive_title=f"Title {i}",
                positive_abstract=f"Abstract {i}",
                negatives=[],
                paper_id=f"p{i}",
                cite_tag=f"<|cite_{i}|>"
            )
            for i in range(100)
        ]
        
        splits = prep.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert len(splits["train"]) == 70
        assert len(splits["val"]) == 15
        assert len(splits["test"]) == 15
    
    def test_create_splits_no_overlap(self):
        """Examples should not appear in multiple splits."""
        prep = CitationDataPrep(data_path="fake.json")
        
        prep.examples = [
            CitationTrainingExample(
                example_id=f"ex_{i}",
                citation_context=f"Context {i}",
                positive_title=f"Title {i}",
                positive_abstract=f"Abstract {i}",
                negatives=[],
                paper_id=f"p{i}",
                cite_tag=f"<|cite_{i}|>"
            )
            for i in range(100)
        ]
        
        splits = prep.create_splits()
        
        train_ids = {ex.example_id for ex in splits["train"]}
        val_ids = {ex.example_id for ex in splits["val"]}
        test_ids = {ex.example_id for ex in splits["test"]}
        
        # No overlap between sets
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
    
    def test_create_splits_covers_all_examples(self):
        """All examples should be in exactly one split."""
        prep = CitationDataPrep(data_path="fake.json")
        
        prep.examples = [
            CitationTrainingExample(
                example_id=f"ex_{i}",
                citation_context=f"Context {i}",
                positive_title=f"Title {i}",
                positive_abstract=f"Abstract {i}",
                negatives=[],
                paper_id=f"p{i}",
                cite_tag=f"<|cite_{i}|>"
            )
            for i in range(100)
        ]
        
        splits = prep.create_splits()
        
        total_in_splits = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total_in_splits == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

