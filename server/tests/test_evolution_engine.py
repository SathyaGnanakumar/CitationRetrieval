"""
Unit tests for self-evolution engine.

Tests evaluation storage, version tracking, and core functionality.
"""

import os
import tempfile
from pathlib import Path

import pytest
import dspy

from src.evaluation.eval_store import EvaluationStore, QueryEvaluation, is_evolution_enabled
from src.evaluation.dspy_metrics import citation_retrieval_metric, query_reformulation_metric
from src.agents.self_evolve.version_tracker import VersionTracker, ModuleVersion


class TestEvolutionFlag:
    """Test evolution flag checking."""
    
    def test_flag_disabled_by_default(self):
        """Evolution should be disabled by default."""
        # Save current env var
        old_value = os.environ.get("ENABLE_DSPY_EVOLUTION")
        
        # Remove env var
        if "ENABLE_DSPY_EVOLUTION" in os.environ:
            del os.environ["ENABLE_DSPY_EVOLUTION"]
        
        assert is_evolution_enabled() == False
        
        # Restore old value
        if old_value is not None:
            os.environ["ENABLE_DSPY_EVOLUTION"] = old_value
    
    def test_flag_enabled(self):
        """Test flag can be enabled."""
        old_value = os.environ.get("ENABLE_DSPY_EVOLUTION")
        
        os.environ["ENABLE_DSPY_EVOLUTION"] = "true"
        assert is_evolution_enabled() == True
        
        if old_value is not None:
            os.environ["ENABLE_DSPY_EVOLUTION"] = old_value
        else:
            del os.environ["ENABLE_DSPY_EVOLUTION"]


class TestEvaluationStore:
    """Test evaluation storage."""
    
    def test_create_evaluation(self):
        """Test creating a query evaluation."""
        eval = QueryEvaluation(
            query="test query",
            paper_id="test_paper",
            retrieved_ids=["doc1", "doc2"],
            relevant_ids=["doc1"],
            metrics={"R@5": 0.5},
            inputs={"query": "test"},
            outputs={"results": []},
            score=0.5
        )
        
        assert eval.query == "test query"
        assert eval.score == 0.5
        assert eval.timestamp is not None
    
    def test_evaluation_to_dspy_example(self):
        """Test converting evaluation to DSPy example."""
        eval = QueryEvaluation(
            query="test query",
            paper_id="test_paper",
            retrieved_ids=["doc1", "doc2"],
            relevant_ids=["doc1"],
            metrics={"R@5": 0.5},
            inputs={"query": "test"},
            outputs={"results": []},
            score=0.5
        )
        
        example = eval.to_dspy_example()
        assert isinstance(example, dspy.Example)
        assert example.citation_context == "test query"
        assert example.ground_truth_ids == ["doc1"]
    
    def test_evaluation_store_disabled(self):
        """Test evaluation store when disabled."""
        old_value = os.environ.get("ENABLE_DSPY_EVOLUTION")
        if "ENABLE_DSPY_EVOLUTION" in os.environ:
            del os.environ["ENABLE_DSPY_EVOLUTION"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvaluationStore(storage_dir=tmpdir)
            
            eval = QueryEvaluation(
                query="test",
                paper_id="test",
                retrieved_ids=[],
                relevant_ids=[],
                metrics={},
                inputs={},
                outputs={},
                score=0.5
            )
            
            # Should be no-op when disabled
            store.add_evaluation(eval)
            assert len(store.evaluations) == 0
        
        if old_value is not None:
            os.environ["ENABLE_DSPY_EVOLUTION"] = old_value
    
    def test_evaluation_store_enabled(self):
        """Test evaluation store when enabled."""
        old_value = os.environ.get("ENABLE_DSPY_EVOLUTION")
        os.environ["ENABLE_DSPY_EVOLUTION"] = "true"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvaluationStore(storage_dir=tmpdir)
            
            eval = QueryEvaluation(
                query="test",
                paper_id="test",
                retrieved_ids=[],
                relevant_ids=[],
                metrics={},
                inputs={},
                outputs={},
                score=0.5
            )
            
            store.add_evaluation(eval)
            assert len(store.evaluations) == 1
        
        if old_value is not None:
            os.environ["ENABLE_DSPY_EVOLUTION"] = old_value
        else:
            del os.environ["ENABLE_DSPY_EVOLUTION"]
    
    def test_get_failures_and_successes(self):
        """Test filtering evaluations by score."""
        old_value = os.environ.get("ENABLE_DSPY_EVOLUTION")
        os.environ["ENABLE_DSPY_EVOLUTION"] = "true"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = EvaluationStore(storage_dir=tmpdir)
            
            # Add evaluations with different scores
            for score in [0.1, 0.5, 0.8]:
                eval = QueryEvaluation(
                    query=f"test_{score}",
                    paper_id="test",
                    retrieved_ids=[],
                    relevant_ids=[],
                    metrics={},
                    inputs={},
                    outputs={},
                    score=score
                )
                store.add_evaluation(eval)
            
            failures = store.get_failures(threshold=0.3)
            assert len(failures) == 1
            assert failures[0].score == 0.1
            
            successes = store.get_successes(threshold=0.7)
            assert len(successes) == 1
            assert successes[0].score == 0.8
        
        if old_value is not None:
            os.environ["ENABLE_DSPY_EVOLUTION"] = old_value
        else:
            del os.environ["ENABLE_DSPY_EVOLUTION"]


class TestVersionTracker:
    """Test module version tracking."""
    
    def test_create_version_tracker(self):
        """Test creating a version tracker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = VersionTracker("test_module", storage_dir=tmpdir)
            assert tracker.module_name == "test_module"
            assert len(tracker.versions) == 0
    
    def test_add_version(self):
        """Test adding a module version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = VersionTracker("test_module", storage_dir=tmpdir)
            
            # Create a simple module
            class DummyModule(dspy.Module):
                def forward(self, x):
                    return x
            
            module = DummyModule()
            version = tracker.add_version(module, score=0.75)
            
            assert version == 0
            assert len(tracker.versions) == 1
            assert tracker.versions[0].score == 0.75
    
    def test_get_best_version(self):
        """Test getting best performing version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = VersionTracker("test_module", storage_dir=tmpdir)
            
            class DummyModule(dspy.Module):
                def forward(self, x):
                    return x
            
            # Add versions with different scores
            for score in [0.5, 0.8, 0.6]:
                module = DummyModule()
                tracker.add_version(module, score=score)
            
            best_version_num = tracker.get_best_version_number()
            assert best_version_num == 1  # Second version has score 0.8


class TestDSPyMetrics:
    """Test DSPy optimization metrics."""
    
    def test_citation_retrieval_metric_perfect(self):
        """Test metric with perfect retrieval."""
        example = dspy.Example(
            citation_context="test query",
            ground_truth_ids=["doc1", "doc2"]
        ).with_inputs('citation_context')
        
        prediction = dspy.Prediction(
            selected_paper={"id": "doc1", "title": "Test"},
            ranked_papers=[
                ({"id": "doc1", "title": "Test"}, 1.0),
                ({"id": "doc2", "title": "Test2"}, 0.9)
            ]
        )
        
        score = citation_retrieval_metric(example, prediction)
        assert score > 0.8  # Should be high for perfect retrieval
    
    def test_citation_retrieval_metric_failure(self):
        """Test metric with failed retrieval."""
        example = dspy.Example(
            citation_context="test query",
            ground_truth_ids=["doc1", "doc2"]
        ).with_inputs('citation_context')
        
        prediction = dspy.Prediction(
            selected_paper={"id": "doc3", "title": "Wrong"},
            ranked_papers=[
                ({"id": "doc3", "title": "Wrong"}, 1.0),
                ({"id": "doc4", "title": "Wrong2"}, 0.9)
            ]
        )
        
        score = citation_retrieval_metric(example, prediction)
        assert score == 0.0  # Should be 0 for complete failure
    
    def test_query_reformulation_metric(self):
        """Test query reformulation metric."""
        example = dspy.Example(
            query="test query"
        )
        
        # Good prediction with diverse queries
        prediction = dspy.Prediction(
            queries=[
                "test query original",
                "alternative test query",
                "different query formulation",
                "another query variant"
            ],
            reasoning="Generated diverse queries"
        )
        
        score = query_reformulation_metric(example, prediction)
        assert score > 0.5  # Should be decent for diverse queries
