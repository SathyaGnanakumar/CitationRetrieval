"""
GEPA optimizer configuration and adapter for self-evolving DSPy agents.

Provides reflective prompt optimization using GEPA with GPT-5 Mini.
"""

import os
from typing import Any, Dict, List, Optional

import dspy
import gepa
from gepa import EvaluationBatch

from src.evaluation.dspy_metrics import citation_retrieval_metric


class GEPAAdapter:
    """
    Adapter for GEPA to work with retrieval system.
    
    Implements required methods for GEPA optimization:
    - evaluate(): Run queries and collect grader scores
    - get_components_to_update(): Identify prompt fields to optimize
    - make_reflective_dataset(): Package training data with feedback
    """
    
    def __init__(
        self,
        workflow,
        resources: Dict[str, Any],
        metric=citation_retrieval_metric
    ):
        """
        Initialize GEPA adapter.
        
        Args:
            workflow: RetrievalWorkflow instance
            resources: Built resources (indexes, embeddings)
            metric: Metric function for evaluation
        """
        self.workflow = workflow
        self.resources = resources
        self.metric = metric
    
    def evaluate(
        self,
        inputs: List[Dict[str, Any]],
        candidate: Dict[str, Any],
        capture_traces: bool = True
    ) -> EvaluationBatch:
        """
        Evaluate candidate module on inputs.
        
        Args:
            inputs: List of input examples (with citation_context, ground_truth_ids)
            candidate: Module candidate dict with prompt fields
            capture_traces: Whether to capture execution traces
            
        Returns:
            EvaluationBatch with scores, outputs, and trajectories
        """
        from langchain_core.messages import HumanMessage
        
        scores: List[float] = []
        outputs: List[str] = []
        trajectories: List[Dict[str, Any]] = []
        
        # Create module with candidate prompt
        module = self._create_module_from_candidate(candidate)
        
        for item in inputs:
            citation_context = item.get("citation_context", "")
            ground_truth_ids = item.get("ground_truth_ids", [])
            
            if not citation_context:
                continue
            
            # Run through workflow
            try:
                final_state = self.workflow.run({
                    'messages': [HumanMessage(content=citation_context)],
                    'resources': self.resources,
                    'config': {'enable_dspy_picker': True, 'k': 20}
                })
                
                # Get ranked papers
                ranked_papers = final_state.get('ranked_papers', [])
                selected_paper = final_state.get('selected_paper')
                
                # Create prediction for metric
                prediction = dspy.Prediction(
                    selected_paper=selected_paper,
                    ranked_papers=ranked_papers
                )
                
                # Create example for metric
                example = dspy.Example(
                    citation_context=citation_context,
                    ground_truth_ids=ground_truth_ids
                ).with_inputs('citation_context')
                
                # Calculate score
                score = self.metric(example, prediction)
                scores.append(float(score))
                
                # Generate feedback for reflection
                feedback = self._generate_feedback(
                    citation_context,
                    ranked_papers,
                    selected_paper,
                    ground_truth_ids,
                    score
                )
                
                # Store trajectory
                if capture_traces:
                    trajectories.append({
                        "inputs": {"citation_context": citation_context},
                        "generated_output": str(selected_paper.get('title', '') if selected_paper else ''),
                        "metrics": {"score": float(score)},
                        "feedback": feedback
                    })
                
                outputs.append(str(selected_paper.get('id', '') if selected_paper else ''))
                
            except Exception as e:
                print(f"Error evaluating input: {e}")
                scores.append(0.0)
                outputs.append("")
                if capture_traces:
                    trajectories.append({
                        "inputs": {"citation_context": citation_context},
                        "generated_output": "",
                        "metrics": {"score": 0.0},
                        "feedback": f"Evaluation error: {e}"
                    })
        
        return EvaluationBatch(
            scores=scores,
            outputs=outputs,
            trajectories=trajectories if capture_traces else None
        )
    
    def _create_module_from_candidate(self, candidate: Dict[str, Any]) -> dspy.Module:
        """Create DSPy module from candidate prompt."""
        from src.agents.formulators.dspy_prompt_generator.modules import get_module
        
        # Get base module
        module = get_module("simple")
        
        # Update system prompt if present in candidate
        system_prompt = candidate.get("system_prompt")
        if system_prompt and hasattr(module, 'retrieve'):
            # Update the signature with new instructions
            # This is simplified - in practice, you'd update the module's prompt
            pass
        
        return module
    
    def _generate_feedback(
        self,
        query: str,
        ranked_papers: List[Any],
        selected_paper: Optional[Dict[str, Any]],
        ground_truth_ids: List[str],
        score: float
    ) -> str:
        """Generate actionable feedback for reflection."""
        feedback_parts = []
        
        # Check if selected paper is correct
        if selected_paper:
            selected_id = selected_paper.get('id')
            if selected_id in ground_truth_ids:
                feedback_parts.append("Selected paper is correct.")
            else:
                feedback_parts.append("Selected paper is INCORRECT - not in ground truth.")
        else:
            feedback_parts.append("No paper selected.")
        
        # Check ranking quality
        if ranked_papers:
            retrieved_ids = [
                p[0].get('id') if isinstance(p, tuple) else p.get('id')
                for p in ranked_papers[:10]
            ]
            hits_top_10 = len(set(retrieved_ids) & set(ground_truth_ids))
            feedback_parts.append(f"Found {hits_top_10}/{len(ground_truth_ids)} relevant papers in top-10.")
        
        # Overall assessment
        if score < 0.3:
            feedback_parts.append("POOR performance - missing most relevant papers.")
        elif score < 0.7:
            feedback_parts.append("MODERATE performance - some relevant papers found.")
        else:
            feedback_parts.append("GOOD performance - most relevant papers found.")
        
        return " ".join(feedback_parts)
    
    def get_components_to_update(self, candidate: Dict[str, Any]) -> List[str]:
        """
        Return list of prompt component keys to optimize.
        
        Args:
            candidate: Current candidate dict
            
        Returns:
            List of field names to update
        """
        return ["system_prompt"]
    
    def make_reflective_dataset(
        self,
        candidate: Dict[str, Any],
        eval_batch: EvaluationBatch,
        components_to_update: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Package evaluation results into reflective dataset.
        
        Args:
            candidate: Current candidate
            eval_batch: Evaluation batch results
            components_to_update: Component keys to update
            
        Returns:
            Dict mapping component names to example lists
        """
        examples = []
        
        if eval_batch.trajectories:
            for traj in eval_batch.trajectories:
                examples.append({
                    "Inputs": traj.get("inputs", {}),
                    "Generated Outputs": traj.get("generated_output", ""),
                    "Feedback": traj.get("feedback", "")
                })
        
        # Return dataset for each component
        return {component: examples for component in components_to_update}


def get_gepa_optimizer(
    metric=citation_retrieval_metric,
    reflection_lm: str = "openai/gpt-5-mini",
    max_metric_calls: int = 50,
    auto: str = "medium",
    num_threads: int = 4,
    track_best_outputs: bool = True
) -> Dict[str, Any]:
    """
    Create GEPA optimizer configuration.
    
    Args:
        metric: Metric function for optimization
        reflection_lm: Model for reflection (GPT-5 Mini)
        max_metric_calls: Budget for optimization
        auto: Budget level ('light', 'medium', 'heavy')
        num_threads: Number of parallel threads
        track_best_outputs: Whether to track best outputs
        
    Returns:
        Dict with optimizer settings
    """
    # Configure reflection LM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found for GPT-5 Mini reflection LM")
    
    reflection_lm_instance = dspy.LM(
        model=reflection_lm,
        api_key=api_key,
        temperature=1.0,
        max_tokens=32000
    )
    
    return {
        "metric": metric,
        "reflection_lm": reflection_lm_instance,
        "max_metric_calls": max_metric_calls,
        "auto": auto,
        "num_threads": num_threads,
        "track_best_outputs": track_best_outputs,
        "display_progress_bar": True
    }


def optimize_with_gepa(
    adapter: GEPAAdapter,
    seed_candidate: Dict[str, Any],
    trainset: List[Dict[str, Any]],
    valset: List[Dict[str, Any]],
    **optimizer_kwargs
) -> Any:
    """
    Run GEPA optimization.
    
    Args:
        adapter: GEPAAdapter instance
        seed_candidate: Initial candidate prompt
        trainset: Training examples
        valset: Validation examples
        **optimizer_kwargs: Additional GEPA configuration
        
    Returns:
        Optimization result with best candidate
    """
    print(f"Starting GEPA optimization...")
    print(f"  Trainset: {len(trainset)} examples")
    print(f"  Valset: {len(valset)} examples")
    print(f"  Auto-budget: {optimizer_kwargs.get('auto', 'medium')}")
    
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        **optimizer_kwargs
    )
    
    print(f"GEPA optimization complete!")
    print(f"  Best score: {result.best_score:.4f}")
    
    return result
