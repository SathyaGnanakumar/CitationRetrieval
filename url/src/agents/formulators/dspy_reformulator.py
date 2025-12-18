"""
DSPy-based query reformulator module.

Provides optimizable query reformulation using DSPy.
"""

import dspy
from typing import List, Dict, Any


class QueryReformulationSignature(dspy.Signature):
    """Signature for query reformulation task."""
    
    query = dspy.InputField(desc="Original query text")
    queries = dspy.OutputField(desc="List of 4 reformulated queries separated by newlines")
    reasoning = dspy.OutputField(desc="Brief explanation of reformulation strategy")


class DSPyQueryReformulator(dspy.Module):
    """
    DSPy module for query reformulation.
    
    Uses Chain-of-Thought reasoning to generate multiple query variations.
    """
    
    def __init__(self):
        super().__init__()
        self.reformulate = dspy.ChainOfThought(QueryReformulationSignature)
    
    def forward(self, query: str) -> dspy.Prediction:
        """
        Generate reformulated queries.
        
        Args:
            query: Original query text
            
        Returns:
            Prediction with queries and reasoning
        """
        result = self.reformulate(query=query)
        
        # Parse queries from output
        queries_text = result.queries
        if isinstance(queries_text, str):
            # Split by newlines and clean up
            query_list = [
                q.strip() 
                for q in queries_text.split('\n') 
                if q.strip()
            ]
            
            # Ensure we have at least the original query
            if not query_list:
                query_list = [query]
            
            # Add original query if not present
            if query not in query_list:
                query_list.insert(0, query)
            
            # Limit to 4 queries
            query_list = query_list[:4]
        else:
            query_list = [query]
        
        # Return with parsed queries
        result.query_list = query_list
        
        return result


def query_reformulator_dspy(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    DSPy-based query reformulator agent for LangGraph workflow.
    
    Compatible with existing workflow interface.
    
    Args:
        state: LangGraph state with messages
        
    Returns:
        Updated state with reformulated queries
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    # Extract query from messages
    msgs = state.get("messages", [])
    user_msg = None
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            user_msg = m.content
            break
    
    if not user_msg:
        return {"messages": [AIMessage(name="reformulator", content="[]")]}
    
    base_query = user_msg.strip()
    
    # Check if DSPy reformulation is enabled
    config = state.get("config", {}) or {}
    use_dspy = config.get("use_dspy_reformulator", False)
    
    if not use_dspy:
        # Fall back to original rule-based reformulation
        from src.agents.formulators.query_reformulator import query_reformulator
        return query_reformulator(state)
    
    # Use DSPy reformulator
    try:
        import dspy
        
        # Get or create reformulator module
        resources = state.get("resources", {}) or {}
        reformulator = resources.get("dspy_reformulator")
        
        if reformulator is None:
            # Create default reformulator
            reformulator = DSPyQueryReformulator()
        
        # Generate reformulations
        result = reformulator(query=base_query)
        expanded_queries = result.query_list
        
        return {
            "query": base_query,
            "queries": expanded_queries,
            "dspy_reasoning": result.reasoning,
            "messages": [AIMessage(name="reformulator", content=str(expanded_queries))],
        }
        
    except Exception as e:
        print(f"DSPy reformulator error: {e}, falling back to rule-based")
        from src.agents.formulators.query_reformulator import query_reformulator
        return query_reformulator(state)
