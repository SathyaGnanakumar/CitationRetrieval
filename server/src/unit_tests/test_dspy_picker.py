"""Unit tests for the DSPy picker agent."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.formulators.dspy_picker import (
    dspy_picker,
    _get_query,
    _truthy,
)


class TestGetQuery:
    """Tests for the _get_query helper function."""
    
    def test_returns_query_from_state(self):
        """Should return query when directly in state."""
        state = {"query": "machine learning papers"}
        result = _get_query(state)
        assert result == "machine learning papers"
    
    def test_strips_whitespace(self):
        """Should strip whitespace from query."""
        state = {"query": "  transformer models  "}
        result = _get_query(state)
        assert result == "transformer models"
    
    def test_falls_back_to_human_message(self):
        """Should use last HumanMessage if no query in state."""
        state = {
            "messages": [
                HumanMessage(content="find citation papers"),
            ]
        }
        result = _get_query(state)
        assert result == "find citation papers"
    
    def test_uses_last_human_message(self):
        """Should use the LAST HumanMessage, not the first."""
        state = {
            "messages": [
                HumanMessage(content="first query"),
                AIMessage(content="some response"),
                HumanMessage(content="second query"),
            ]
        }
        result = _get_query(state)
        assert result == "second query"
    
    def test_returns_none_if_no_query(self):
        """Should return None if no query found."""
        state = {"messages": []}
        result = _get_query(state)
        assert result is None
    
    def test_returns_none_for_empty_query(self):
        """Should return None for empty/whitespace query."""
        state = {"query": "   "}
        result = _get_query(state)
        assert result is None
    
    def test_prefers_state_query_over_messages(self):
        """Should prefer state['query'] over messages."""
        state = {
            "query": "state query",
            "messages": [HumanMessage(content="message query")]
        }
        result = _get_query(state)
        assert result == "state query"


class TestTruthy:
    """Tests for the _truthy helper function."""
    
    def test_true_boolean(self):
        """Should return True for True."""
        assert _truthy(True) == True
    
    def test_false_boolean(self):
        """Should return False for False."""
        assert _truthy(False) == False
    
    def test_none(self):
        """Should return False for None."""
        assert _truthy(None) == False
    
    def test_positive_int(self):
        """Should return True for positive int."""
        assert _truthy(1) == True
    
    def test_zero_int(self):
        """Should return False for 0."""
        assert _truthy(0) == False
    
    def test_string_true(self):
        """Should return True for 'true' string."""
        assert _truthy("true") == True
        assert _truthy("True") == True
        assert _truthy("TRUE") == True
    
    def test_string_yes(self):
        """Should return True for 'yes' string."""
        assert _truthy("yes") == True
        assert _truthy("Yes") == True
    
    def test_string_1(self):
        """Should return True for '1' string."""
        assert _truthy("1") == True
    
    def test_string_on(self):
        """Should return True for 'on' string."""
        assert _truthy("on") == True
    
    def test_string_y(self):
        """Should return True for 'y' string."""
        assert _truthy("y") == True
    
    def test_string_false(self):
        """Should return False for 'false' string."""
        assert _truthy("false") == False
    
    def test_string_no(self):
        """Should return False for 'no' string."""
        assert _truthy("no") == False
    
    def test_string_0(self):
        """Should return False for '0' string."""
        assert _truthy("0") == False
    
    def test_empty_string(self):
        """Should return False for empty string."""
        assert _truthy("") == False
    
    def test_whitespace_string(self):
        """Should return False for whitespace string."""
        assert _truthy("  ") == False


class TestDspyPickerDisabled:
    """Tests for dspy_picker when disabled."""
    
    def test_returns_empty_when_not_enabled(self):
        """Should return empty dict when not enabled."""
        state = {
            "query": "test query",
            "ranked_papers": [{"title": "Paper 1"}],
            "config": {"enable_dspy_picker": False}
        }
        result = dspy_picker(state)
        assert result == {}
    
    def test_returns_empty_when_no_config(self):
        """Should return empty dict when no config."""
        state = {
            "query": "test query",
            "ranked_papers": [{"title": "Paper 1"}],
        }
        result = dspy_picker(state)
        assert result == {}


class TestDspyPickerErrorHandling:
    """Tests for dspy_picker error handling."""
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    def test_error_when_no_ranked_papers(self):
        """Should return error when no ranked_papers."""
        state = {
            "query": "test query",
            "config": {"enable_dspy_picker": True}
        }
        result = dspy_picker(state)
        
        assert "messages" in result
        assert "no ranked_papers" in result["messages"][0].content
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    def test_error_when_empty_ranked_papers(self):
        """Should return error when ranked_papers is empty list."""
        state = {
            "query": "test query",
            "ranked_papers": [],
            "config": {"enable_dspy_picker": True}
        }
        result = dspy_picker(state)
        
        assert "messages" in result
        assert "no ranked_papers" in result["messages"][0].content
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    def test_error_when_no_query(self):
        """Should return error when no query found."""
        state = {
            "ranked_papers": [{"title": "Paper 1", "abstract": "Abstract 1"}],
            "config": {"enable_dspy_picker": True}
        }
        result = dspy_picker(state)
        
        assert "messages" in result
        assert "missing query" in result["messages"][0].content
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    def test_error_when_no_usable_candidates(self):
        """Should return error when candidates have no titles."""
        state = {
            "query": "test query",
            "ranked_papers": [{"abstract": "No title here"}],  # Missing title
            "config": {"enable_dspy_picker": True}
        }
        result = dspy_picker(state)
        
        assert "messages" in result
        assert "no usable candidates" in result["messages"][0].content


class TestDspyPickerCandidateBuilding:
    """Tests for how dspy_picker builds candidates."""
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    @patch('src.agents.formulators.dspy_prompt_generator.modules.get_module')
    def test_respects_top_n_config(self, mock_get_module):
        """Should only use top N candidates."""
        # Setup mock module
        mock_module = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.selected_title = "Paper 1"
        mock_prediction.reasoning = "test reasoning"
        mock_module.return_value = mock_prediction
        mock_get_module.return_value = mock_module
        
        state = {
            "query": "test query",
            "ranked_papers": [
                {"title": f"Paper {i}", "abstract": f"Abstract {i}"}
                for i in range(20)
            ],
            "config": {"enable_dspy_picker": True, "dspy_top_n": 5}
        }
        
        result = dspy_picker(state)
        
        # Check module was called with limited candidates
        call_args = mock_module.call_args
        candidates = call_args.kwargs.get("candidates") or call_args[1].get("candidates")
        assert len(candidates) <= 5
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    @patch('src.agents.formulators.dspy_prompt_generator.modules.get_module')
    def test_filters_papers_without_titles(self, mock_get_module):
        """Should filter out papers without titles."""
        mock_module = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.selected_title = "Valid Paper"
        mock_prediction.reasoning = "test"
        mock_module.return_value = mock_prediction
        mock_get_module.return_value = mock_module
        
        state = {
            "query": "test query",
            "ranked_papers": [
                {"title": "Valid Paper", "abstract": "Good"},
                {"abstract": "No title"},  # Should be filtered
                {"title": "", "abstract": "Empty title"},  # Should be filtered
                {"title": "Another Valid", "abstract": "Also good"},
            ],
            "config": {"enable_dspy_picker": True}
        }
        
        result = dspy_picker(state)
        
        # Verify only valid papers were passed
        call_args = mock_module.call_args
        candidates = call_args.kwargs.get("candidates") or call_args[1].get("candidates")
        titles = [c["title"] for c in candidates]
        assert "Valid Paper" in titles
        assert "Another Valid" in titles
        assert "" not in titles


class TestDspyPickerSuccess:
    """Tests for successful dspy_picker execution."""
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    @patch('src.agents.formulators.dspy_prompt_generator.modules.get_module')
    def test_returns_selected_paper(self, mock_get_module):
        """Should return selected_paper on success."""
        mock_module = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.selected_title = "Paper 2"
        mock_prediction.reasoning = "This paper is most relevant"
        mock_module.return_value = mock_prediction
        mock_get_module.return_value = mock_module
        
        state = {
            "query": "test query",
            "ranked_papers": [
                {"title": "Paper 1", "abstract": "Abstract 1", "id": "id1"},
                {"title": "Paper 2", "abstract": "Abstract 2", "id": "id2"},
                {"title": "Paper 3", "abstract": "Abstract 3", "id": "id3"},
            ],
            "config": {"enable_dspy_picker": True}
        }
        
        result = dspy_picker(state)
        
        assert "selected_paper" in result
        assert result["selected_paper"]["title"] == "Paper 2"
        assert result["dspy_selected_title"] == "Paper 2"
        assert result["dspy_reasoning"] == "This paper is most relevant"
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    @patch('src.agents.formulators.dspy_prompt_generator.modules.get_module')
    def test_case_insensitive_title_matching(self, mock_get_module):
        """Should match titles case-insensitively."""
        mock_module = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.selected_title = "PAPER TWO"  # Different case
        mock_prediction.reasoning = "test"
        mock_module.return_value = mock_prediction
        mock_get_module.return_value = mock_module
        
        state = {
            "query": "test query",
            "ranked_papers": [
                {"title": "Paper One", "abstract": ""},
                {"title": "Paper Two", "abstract": ""},  # Should match
            ],
            "config": {"enable_dspy_picker": True}
        }
        
        result = dspy_picker(state)
        
        assert result["selected_paper"]["title"] == "Paper Two"
    
    @patch.dict('os.environ', {'ENABLE_DSPY_PICKER': '1'})
    @patch('src.agents.formulators.dspy_prompt_generator.modules.get_module')
    def test_falls_back_to_first_candidate(self, mock_get_module):
        """Should fall back to first candidate if no match found."""
        mock_module = MagicMock()
        mock_prediction = MagicMock()
        mock_prediction.selected_title = "Nonexistent Paper"
        mock_prediction.reasoning = "test"
        mock_module.return_value = mock_prediction
        mock_get_module.return_value = mock_module
        
        state = {
            "query": "test query",
            "ranked_papers": [
                {"title": "Paper One", "abstract": ""},
                {"title": "Paper Two", "abstract": ""},
            ],
            "config": {"enable_dspy_picker": True}
        }
        
        result = dspy_picker(state)
        
        # Falls back to first candidate
        assert result["selected_paper"]["title"] == "Paper One"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

