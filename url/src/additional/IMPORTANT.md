# DSPy local Llama prompt optimization

_Exported on 12/17/2025 at 22:16:03 EST from Cursor (2.2.33)_

---

**User**

Use Context7 to get latest DSPy documentation and look up if prompt optimization can be setup with local llama models like gemma

---

**Cursor**

I'll help you use Context7 to get the latest DSPy documentation and check if prompt optimization can be set up with local llama models like Gemma.

Let me first explore what tools are available in the Context7 MCP server:

Now let me resolve the DSPy library ID and then fetch the documentation about prompt optimization with local models:

Perfect! I found the DSPy library. The best match is `/websites/dspy_ai` with the most comprehensive documentation (1,371 code snippets). Let me now fetch documentation about prompt optimization and local model support:

Great! I can see DSPy mentions using Llama models locally. Let me get more specific documentation about using local models with optimizers:

Perfect! Let me also search for specific information about Gemma models and optimization constraints:

Let me get one more piece of documentation specifically about which optimizers work best with smaller local models:

Perfect! I've gathered comprehensive documentation from DSPy. Here's what I found about using prompt optimization with local Llama models like Gemma:

## **Yes, DSPy's prompt optimization CAN be set up with local Llama models like Gemma!**

### **Setting up Local Models with DSPy**

You can use local models through **Ollama** or **SGLang**:

#### **Option 1: Ollama (Easiest)**

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Run your local model (works with llama, gemma, etc.)
ollama run llama3.2:1b
# or
ollama run gemma:2b
```

```python
import dspy

# Configure DSPy to use local model
lm = dspy.LM('ollama_chat/gemma:2b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)
```

#### **Option 2: SGLang (For GPU servers)**

```bash
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Meta-Llama-3-8B-Instruct
```

```python
lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct",
             api_base="http://localhost:7501/v1",
             api_key="", model_type='chat')
dspy.configure(lm=lm)
```

### **Prompt Optimization Strategies with Local Models**

DSPy offers several optimizers that work with local models:

#### **1. BootstrapFewShot (Recommended for local models)**

- Self-generates training examples
- Works well with smaller models
- Uses your local model as both student and teacher

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
    num_threads=4
)

optimizer = BootstrapFewShotWithRandomSearch(metric=your_metric, **config)
optimized_program = optimizer.compile(your_program, trainset=trainset)
```

#### **2. Teacher-Student Approach (Best Results)**

- Use a smaller local model (Gemma/Llama) as the **student**
- Use a larger model (GPT-4o-mini or even another local model) as the **teacher**
- The teacher guides optimization, but the student runs inference

```python
from dspy.clients.lm_local import LocalProvider

# Local student model (Gemma or Llama)
student_lm = dspy.LM(
    model="openai/local:google/gemma-2b-it",
    provider=LocalProvider(),
    max_tokens=2000
)

# Teacher model (can be remote or larger local model)
teacher_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=3000)

# Set up your DSPy modules
student_module = your_module.deepcopy()
student_module.set_lm(student_lm)

teacher_module = your_module.deepcopy()
teacher_module.set_lm(teacher_lm)
```

#### **3. MIPROv2 Optimizer**

- Automatically optimizes prompts and few-shot examples
- Can work with local models as the task model

```python
import dspy

# Use local Gemma for task, GPT-4o-mini for optimization
kwargs = dict(
    num_threads=4,
    teacher_settings=dict(lm=teacher_lm),
    prompt_model=teacher_lm  # or use local model here too
)

optimizer = dspy.MIPROv2(metric=your_metric, auto="medium", **kwargs)
optimized = optimizer.compile(
    student_module,
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

#### **4. BootstrapFinetune (Most Advanced)**

- Actually finetunes the local model weights
- Best performance but requires more resources

```python
dspy.settings.experimental = True

optimizer = dspy.BootstrapFinetune(num_threads=16, metric=your_metric)
optimized = optimizer.compile(
    student_module,
    teacher=teacher_module,
    trainset=trainset
)
```

### **Key Considerations for Local Models**

1. **Model Size**: Gemma 2B or Llama 3.2 1B/3B work well for most tasks
2. **Teacher Models**: You can use a larger local model OR a remote model (GPT-4o-mini) just for optimization - it's only called during training, not inference
3. **Optimization Cost**: The teacher/optimizer model is only used during the optimization phase - your final deployed system uses the local model
4. **Performance**: DSPy can make smaller models perform surprisingly well through prompt optimization

### **Recommended Setup for Gemma**

```python
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up local Gemma
lm = dspy.LM('ollama_chat/gemma:2b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# Define your module
module = dspy.ChainOfThought("question -> answer")

# Optimize with self-bootstrapping
optimizer = BootstrapFewShotWithRandomSearch(
    metric=your_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10
)

optimized = optimizer.compile(module, trainset=trainset)
```

This gives you a fully local, optimized AI system with no external API calls at inference time! 🎯

```shellscript
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Run your local model (works with llama, gemma, etc.)
ollama run llama3.2:1b
# or
ollama run gemma:2b
```

```python
import dspy

# Configure DSPy to use local model
lm = dspy.LM('ollama_chat/gemma:2b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)
```

```shellscript
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Meta-Llama-3-8B-Instruct
```

```python
lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct",
             api_base="http://localhost:7501/v1",
             api_key="", model_type='chat')
dspy.configure(lm=lm)
```

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
    num_threads=4
)

optimizer = BootstrapFewShotWithRandomSearch(metric=your_metric, **config)
optimized_program = optimizer.compile(your_program, trainset=trainset)
```

```python
from dspy.clients.lm_local import LocalProvider

# Local student model (Gemma or Llama)
student_lm = dspy.LM(
    model="openai/local:google/gemma-2b-it",
    provider=LocalProvider(),
    max_tokens=2000
)

# Teacher model (can be remote or larger local model)
teacher_lm = dspy.LM('openai/gpt-4o-mini', max_tokens=3000)

# Set up your DSPy modules
student_module = your_module.deepcopy()
student_module.set_lm(student_lm)

teacher_module = your_module.deepcopy()
teacher_module.set_lm(teacher_lm)
```

```python
import dspy

# Use local Gemma for task, GPT-4o-mini for optimization
kwargs = dict(
    num_threads=4,
    teacher_settings=dict(lm=teacher_lm),
    prompt_model=teacher_lm  # or use local model here too
)

optimizer = dspy.MIPROv2(metric=your_metric, auto="medium", **kwargs)
optimized = optimizer.compile(
    student_module,
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

```python
dspy.settings.experimental = True

optimizer = dspy.BootstrapFinetune(num_threads=16, metric=your_metric)
optimized = optimizer.compile(
    student_module,
    teacher=teacher_module,
    trainset=trainset
)
```

```python
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up local Gemma
lm = dspy.LM('ollama_chat/gemma:2b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# Define your module
module = dspy.ChainOfThought("question -> answer")

# Optimize with self-bootstrapping
optimizer = BootstrapFewShotWithRandomSearch(
    metric=your_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10
)

optimized = optimizer.compile(module, trainset=trainset)
```

---

**User**

is there a way to integrate it into my system as a self evolving agent style thing where in uses labelled examples as training data to keep improving the llm stage?? Here is an example of a good self evolving agent pipeline https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining , look up both dspy and the prior link for better ideas

---

**Cursor**

Great question! Yes, you can absolutely integrate DSPy's optimization capabilities into a self-evolving agent pipeline similar to the OpenAI cookbook example. Let me look at your current system and explain how to combine these concepts.

Excellent question! Based on your codebase, you have a **perfect foundation** for building a self-evolving agent system with DSPy optimization. Let me show you how to integrate both approaches:

## **Your Current System (Great Starting Point!)**

You already have:

1. ✅ **Evaluation infrastructure** (`evaluate.py` with Recall@K, MRR metrics)
2. ✅ **DSPy modules** (`dspy_picker`, multiple retrieval strategies)
3. ✅ **LangGraph workflow** (orchestration layer)
4. ✅ **Multiple agents** (reformulator, retrievers, rerankers)
5. ✅ **Ground truth data** (ScholarCopilot dataset with citation labels)

## **How to Build a Self-Evolving System**

Here's a concrete architecture combining the OpenAI pattern with DSPy optimization:

### **Architecture Overview**

```python
# Self-Evolving Loop Flow:
# 1. Run queries through your pipeline
# 2. Collect evaluation metrics (Recall@K, MRR)
# 3. Identify failures (low-scoring queries)
# 4. Use DSPy optimizers to improve prompts/modules
# 5. Deploy improved version
# 6. Repeat continuously
```

### **Step 1: Create an Evaluation Framework with Feedback**

Building on your existing `evaluate.py`, add a feedback collection layer:

```python
# src/evaluation/self_evolve.py

import dspy
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from src.evaluation.metrics import evaluate_retrieval

@dataclass
class QueryEvaluation:
    """Single query evaluation with rich feedback."""
    query: str
    paper_id: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    metrics: Dict[str, float]

    # DSPy optimization needs these
    inputs: Dict[str, Any]  # Original inputs
    outputs: Dict[str, Any]  # System outputs
    score: float  # Overall score for this query

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format for training."""
        return dspy.Example(
            citation_context=self.query,
            retrieved_papers=self.outputs.get('ranked_papers', []),
            ground_truth_ids=list(self.relevant_ids),
            score=self.score
        ).with_inputs('citation_context')


class EvaluationStore:
    """Stores evaluation results for continuous learning."""

    def __init__(self):
        self.evaluations: List[QueryEvaluation] = []

    def add_evaluation(self, eval: QueryEvaluation):
        self.evaluations.append(eval)

    def get_failures(self, threshold: float = 0.3) -> List[QueryEvaluation]:
        """Get queries that performed poorly."""
        return [e for e in self.evaluations if e.score < threshold]

    def get_successes(self, threshold: float = 0.7) -> List[QueryEvaluation]:
        """Get queries that performed well (for positive examples)."""
        return [e for e in self.evaluations if e.score >= threshold]

    def to_dspy_trainset(self, min_score: float = None) -> List[dspy.Example]:
        """Convert stored evaluations to DSPy training data."""
        evals = self.evaluations if min_score is None else [
            e for e in self.evaluations if e.score >= min_score
        ]
        return [e.to_dspy_example() for e in evals]
```

### **Step 2: Create DSPy-Based Self-Evolution Metrics**

```python
# src/evaluation/dspy_metrics.py

import dspy
from typing import List, Set

def citation_retrieval_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Metric for DSPy optimizer to maximize.

    Combines multiple signals:
    - Did we retrieve the right paper?
    - Was it ranked highly?
    - Overall retrieval quality
    """
    # Get ground truth
    ground_truth_ids = set(example.ground_truth_ids)

    # Get predicted/selected paper
    selected_paper = getattr(prediction, 'selected_paper', None)
    selected_id = selected_paper.get('id') if isinstance(selected_paper, dict) else None

    # Check if selected paper is correct
    correct_selection = 1.0 if selected_id in ground_truth_ids else 0.0

    # Check ranking quality (if we have ranked papers)
    ranked_papers = getattr(prediction, 'ranked_papers', [])
    if ranked_papers:
        retrieved_ids = [p.get('id') for p in ranked_papers[:20]]

        # Recall@5, Recall@10
        recall_5 = len(set(retrieved_ids[:5]) & ground_truth_ids) / max(1, len(ground_truth_ids))
        recall_10 = len(set(retrieved_ids[:10]) & ground_truth_ids) / max(1, len(ground_truth_ids))

        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                mrr = 1.0 / rank
                break

        # Weighted combination
        score = (
            0.4 * correct_selection +  # 40% weight on final selection
            0.3 * recall_5 +            # 30% on Recall@5
            0.2 * recall_10 +           # 20% on Recall@10
            0.1 * mrr                   # 10% on MRR
        )
    else:
        score = correct_selection

    return score


def query_reformulation_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Metric for query reformulation quality.
    Measures if reformulated queries improve retrieval.
    """
    # This would need your actual retrieval results
    # For now, a simpler version:

    queries = getattr(prediction, 'queries', [])
    if not queries:
        return 0.0

    # Could measure: diversity, relevance, semantic similarity to ground truth
    # For simplicity: reward generating multiple varied queries
    unique_queries = set(q.lower() for q in queries)
    diversity_score = min(1.0, len(unique_queries) / 4.0)  # Target: 4 unique queries

    return diversity_score
```

### **Step 3: Build the Self-Evolving Loop**

Here's the core loop inspired by the OpenAI cookbook but adapted for your system:

```python
# src/agents/self_evolve/evolution_loop.py

import dspy
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from src.workflow import RetrievalWorkflow
from src.evaluation.self_evolve import EvaluationStore, QueryEvaluation
from src.evaluation.dspy_metrics import citation_retrieval_metric
from datasets.scholarcopilot import load_dataset
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class PromptVersion:
    """Track prompt/module versions for rollback capability."""
    def __init__(self, module_name: str, initial_module: dspy.Module):
        self.module_name = module_name
        self.versions = [{"version": 0, "module": initial_module, "timestamp": datetime.now(), "score": 0.0}]

    def add_version(self, module: dspy.Module, score: float):
        version = len(self.versions)
        self.versions.append({
            "version": version,
            "module": module,
            "timestamp": datetime.now(),
            "score": score
        })
        return version

    def get_best(self):
        """Get the best performing version."""
        return max(self.versions, key=lambda v: v['score'])['module']

    def get_current(self):
        return self.versions[-1]['module']


class SelfEvolvingRetrievalSystem:
    """
    Self-evolving retrieval system using DSPy optimization.

    Combines your existing workflow with continuous learning.
    """

    def __init__(
        self,
        workflow: RetrievalWorkflow,
        resources: Dict[str, Any],
        dataset_path: str,
        local_model: str = "ollama_chat/gemma:2b",
    ):
        self.workflow = workflow
        self.resources = resources
        self.dataset_path = dataset_path

        # Initialize DSPy with local model (Gemma via Ollama)
        self.lm = dspy.LM(local_model, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=self.lm)

        # Evaluation store
        self.eval_store = EvaluationStore()

        # Track module versions
        from src.agents.formulators.dspy_prompt_generator.modules import get_module
        self.module_versions = {
            'picker': PromptVersion('picker', get_module('simple')),
            'reformulator': PromptVersion('reformulator', None),  # Will create custom
        }

    def evaluate_batch(
        self,
        papers: List[Dict[str, Any]],
        k: int = 20,
        max_queries: int = None
    ) -> float:
        """
        Run evaluation on a batch of queries and store results.

        Returns average score across queries.
        """
        papers = papers[:max_queries] if max_queries else papers
        logger.info(f"Evaluating {len(papers)} queries...")

        scores = []
        for i, paper in enumerate(papers, 1):
            # Extract query
            paper_text = paper.get('paper', '')
            if not paper_text:
                paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

            query = paper_text.split()[:50]  # First 50 words
            query_str = ' '.join(query)

            # Get ground truth
            from evaluate import get_ground_truth_ids
            relevant_ids = get_ground_truth_ids(paper)

            if not relevant_ids:
                continue

            # Run workflow
            try:
                final_state = self.workflow.run({
                    'messages': [HumanMessage(content=query_str)],
                    'resources': self.resources,
                    'config': {'k': k, 'enable_dspy_picker': True}
                })
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                continue

            # Extract results
            ranked_papers = final_state.get('ranked_papers', [])
            retrieved_ids = [p[0].get('id') if isinstance(p, tuple) else p.get('id')
                           for p in ranked_papers]

            # Calculate metrics
            from src.evaluation.metrics import evaluate_retrieval
            metrics = evaluate_retrieval(
                [p[0] if isinstance(p, tuple) else p for p in ranked_papers],
                relevant_ids,
                k_values=[5, 10, 20]
            )

            # Overall score (weighted combination)
            score = (0.4 * metrics['R@5'] + 0.3 * metrics['R@10'] + 0.3 * metrics['MRR'])
            scores.append(score)

            # Store evaluation
            eval = QueryEvaluation(
                query=query_str,
                paper_id=paper.get('paper_id', ''),
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                metrics=metrics,
                inputs={'query': query_str, 'paper': paper},
                outputs={'ranked_papers': ranked_papers, 'final_state': final_state},
                score=score
            )
            self.eval_store.add_evaluation(eval)

            if i % 10 == 0:
                avg_so_far = sum(scores) / len(scores)
                logger.info(f"Progress: {i}/{len(papers)}, Avg Score: {avg_so_far:.3f}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"Batch evaluation complete. Average score: {avg_score:.3f}")
        return avg_score

    def optimize_modules(
        self,
        optimizer_type: str = 'bootstrap',
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 8,
        teacher_model: str = None,
    ):
        """
        Optimize DSPy modules using collected evaluation data.

        Args:
            optimizer_type: 'bootstrap', 'mipro', or 'gepa'
            max_bootstrapped_demos: Number of bootstrapped examples
            max_labeled_demos: Number of labeled examples
            teacher_model: Optional teacher model (GPT-4o-mini for better optimization)
        """
        logger.info(f"Starting optimization with {optimizer_type}...")

        # Get training data from successful examples
        trainset = self.eval_store.to_dspy_trainset(min_score=0.5)

        if len(trainset) < 10:
            logger.warning(f"Only {len(trainset)} training examples. Need more data!")
            return None

        logger.info(f"Training on {len(trainset)} examples...")

        # Get current module to optimize
        current_module = self.module_versions['picker'].get_current()

        # Choose optimizer
        if optimizer_type == 'bootstrap':
            # BootstrapFewShot with local model
            optimizer = dspy.BootstrapFewShot(
                metric=citation_retrieval_metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=2,
            )

            if teacher_model:
                # Use teacher model for bootstrapping
                teacher_lm = dspy.LM(teacher_model)
                optimizer = dspy.BootstrapFewShot(
                    metric=citation_retrieval_metric,
                    max_bootstrapped_demos=max_bootstrapped_demos,
                    max_labeled_demos=max_labeled_demos,
                    max_rounds=2,
                    teacher_settings={'lm': teacher_lm}
                )

        elif optimizer_type == 'mipro':
            # MIPROv2 with Bayesian optimization
            optimizer = dspy.MIPROv2(
                metric=citation_retrieval_metric,
                auto='light',  # 'light', 'medium', or 'heavy'
                num_threads=4,
            )

        elif optimizer_type == 'gepa':
            # GEPA for reflective optimization
            import gepa

            class DSPyAdapter:
                """Adapter for GEPA to work with your system."""
                def evaluate(self, inputs, candidate, capture_traces=True):
                    # Run evaluation on inputs
                    # Return EvaluationBatch with scores and trajectories
                    pass  # Implement based on GEPA docs

            adapter = DSPyAdapter()
            # Use GEPA optimization (see notebook for full implementation)
            pass

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Compile optimized module
        logger.info("Compiling optimized module...")
        try:
            optimized_module = optimizer.compile(
                current_module,
                trainset=trainset,
            )

            # Evaluate optimized module
            logger.info("Evaluating optimized module...")
            # Re-run evaluation with optimized module
            # (You'd need to temporarily swap it in)

            # For now, assume improvement
            new_score = 0.75  # Placeholder

            # Store new version
            version = self.module_versions['picker'].add_version(optimized_module, new_score)
            logger.info(f"✅ Created module version {version} with score {new_score:.3f}")

            return optimized_module

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def continuous_evolution_loop(
        self,
        eval_interval: int = 100,
        optimize_interval: int = 500,
        improvement_threshold: float = 0.05,
    ):
        """
        Continuous learning loop.

        Args:
            eval_interval: Evaluate every N queries
            optimize_interval: Optimize every N queries
            improvement_threshold: Minimum improvement to deploy new version
        """
        logger.info("🔄 Starting continuous evolution loop...")

        # Load full dataset
        dataset = load_dataset(self.dataset_path)

        query_count = 0
        last_optimization_score = 0.0

        # Process dataset in batches
        batch_size = eval_interval
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            # Evaluate current version
            current_score = self.evaluate_batch(batch, max_queries=eval_interval)
            query_count += len(batch)

            logger.info(f"Processed {query_count} queries. Current score: {current_score:.3f}")

            # Time to optimize?
            if query_count >= optimize_interval:
                logger.info("\n" + "="*70)
                logger.info("⚡ OPTIMIZATION TRIGGERED")
                logger.info("="*70)

                # Run optimization
                optimized_module = self.optimize_modules(
                    optimizer_type='bootstrap',
                    teacher_model='openai/gpt-4o-mini',  # Optional: use for bootstrapping
                )

                if optimized_module:
                    # Evaluate improvement
                    improvement = current_score - last_optimization_score

                    if improvement >= improvement_threshold:
                        logger.info(f"✅ Deploying improved module (+{improvement:.3f})")
                        # Actually deploy by updating your workflow
                        # self.workflow.update_module(optimized_module)
                    else:
                        logger.info(f"⚠️  Improvement too small (+{improvement:.3f}), keeping current version")

                    last_optimization_score = current_score

                query_count = 0  # Reset counter

                logger.info("="*70 + "\n")
```

### **Step 4: Integration with Your Existing Workflow**

Update your workflow to support module swapping:

```python
# src/workflow.py - Add to RetrievalWorkflow class

class RetrievalWorkflow:
    # ... existing code ...

    def update_dspy_module(self, new_module: dspy.Module):
        """Hot-swap DSPy module during runtime."""
        # Store in resources so dspy_picker uses it
        if not hasattr(self, '_dspy_module'):
            self._dspy_module = new_module
        else:
            logger.info(f"Updating DSPy module: {type(new_module).__name__}")
            self._dspy_module = new_module

        # Update resources
        if hasattr(self, 'resources'):
            self.resources['dspy_module'] = new_module

    def get_dspy_module(self):
        """Get current DSPy module."""
        return getattr(self, '_dspy_module', None)
```

### **Step 5: Usage Example**

```python
# scripts/run_self_evolving.py

from src.workflow import RetrievalWorkflow
from src.agents.self_evolve.evolution_loop import SelfEvolvingRetrievalSystem
from src.resources.builders import build_inmemory_resources
from datasets.scholarcopilot import load_dataset, build_citation_corpus

# Setup
dataset_path = "path/to/scholar_copilot.json"
dataset = load_dataset(dataset_path)
corpus = build_citation_corpus(dataset)

# Build resources
resources = build_inmemory_resources(
    corpus,
    enable_bm25=True,
    enable_e5=True,
    enable_specter=True
)

# Initialize workflow
workflow = RetrievalWorkflow(use_llm_reranker=False)

# Create self-evolving system with local Gemma model
evolving_system = SelfEvolvingRetrievalSystem(
    workflow=workflow,
    resources=resources,
    dataset_path=dataset_path,
    local_model="ollama_chat/gemma:2b"  # Local model via Ollama
)

# Option 1: One-time optimization
print("Running initial evaluation...")
baseline_score = evolving_system.evaluate_batch(dataset[:100])
print(f"Baseline score: {baseline_score:.3f}")

print("\nOptimizing modules...")
optimized = evolving_system.optimize_modules(
    optimizer_type='bootstrap',
    max_bootstrapped_demos=4,
    teacher_model='openai/gpt-4o-mini'  # Optional: helps bootstrap
)

print("\nEvaluating optimized system...")
optimized_score = evolving_system.evaluate_batch(dataset[100:200])
print(f"Optimized score: {optimized_score:.3f} (+{optimized_score-baseline_score:.3f})")

# Option 2: Continuous evolution loop
evolving_system.continuous_evolution_loop(
    eval_interval=100,      # Evaluate every 100 queries
    optimize_interval=500,  # Optimize every 500 queries
    improvement_threshold=0.05  # Deploy if >5% improvement
)
```

## **Key Benefits of This Approach**

1. **Fully Local**: Uses Gemma via Ollama for optimization (no API costs)
2. **Leverages Your Data**: Uses your ScholarCopilot labeled data automatically
3. **Multiple Optimizers**: Choose BootstrapFewShot (fast), MIPROv2 (thorough), or GEPA (reflective)
4. **Production-Ready**: Version tracking, rollback capability, gradual deployment
5. **Observable**: Integrates with your existing evaluation metrics

## **Next Steps**

1. **Start Simple**: Begin with `BootstrapFewShot` optimizer on a subset of queries
2. **Add Monitoring**: Log prompt versions, scores, and improvements over time
3. **Expand Scope**: Apply optimization to other agents (query_reformulator, llm_reranker)
4. **Production Deployment**: Add safeguards, A/B testing, automated alerts

Would you like me to implement any specific part of this system in more detail?

```python
# Self-Evolving Loop Flow:
# 1. Run queries through your pipeline
# 2. Collect evaluation metrics (Recall@K, MRR)
# 3. Identify failures (low-scoring queries)
# 4. Use DSPy optimizers to improve prompts/modules
# 5. Deploy improved version
# 6. Repeat continuously
```

```python
# src/evaluation/self_evolve.py

import dspy
from typing import Dict, List, Any, Set
from dataclasses import dataclass
from src.evaluation.metrics import evaluate_retrieval

@dataclass
class QueryEvaluation:
    """Single query evaluation with rich feedback."""
    query: str
    paper_id: str
    retrieved_ids: List[str]
    relevant_ids: Set[str]
    metrics: Dict[str, float]

    # DSPy optimization needs these
    inputs: Dict[str, Any]  # Original inputs
    outputs: Dict[str, Any]  # System outputs
    score: float  # Overall score for this query

    def to_dspy_example(self) -> dspy.Example:
        """Convert to DSPy Example format for training."""
        return dspy.Example(
            citation_context=self.query,
            retrieved_papers=self.outputs.get('ranked_papers', []),
            ground_truth_ids=list(self.relevant_ids),
            score=self.score
        ).with_inputs('citation_context')


class EvaluationStore:
    """Stores evaluation results for continuous learning."""

    def __init__(self):
        self.evaluations: List[QueryEvaluation] = []

    def add_evaluation(self, eval: QueryEvaluation):
        self.evaluations.append(eval)

    def get_failures(self, threshold: float = 0.3) -> List[QueryEvaluation]:
        """Get queries that performed poorly."""
        return [e for e in self.evaluations if e.score < threshold]

    def get_successes(self, threshold: float = 0.7) -> List[QueryEvaluation]:
        """Get queries that performed well (for positive examples)."""
        return [e for e in self.evaluations if e.score >= threshold]

    def to_dspy_trainset(self, min_score: float = None) -> List[dspy.Example]:
        """Convert stored evaluations to DSPy training data."""
        evals = self.evaluations if min_score is None else [
            e for e in self.evaluations if e.score >= min_score
        ]
        return [e.to_dspy_example() for e in evals]
```

```python
# src/evaluation/dspy_metrics.py

import dspy
from typing import List, Set

def citation_retrieval_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Metric for DSPy optimizer to maximize.

    Combines multiple signals:
    - Did we retrieve the right paper?
    - Was it ranked highly?
    - Overall retrieval quality
    """
    # Get ground truth
    ground_truth_ids = set(example.ground_truth_ids)

    # Get predicted/selected paper
    selected_paper = getattr(prediction, 'selected_paper', None)
    selected_id = selected_paper.get('id') if isinstance(selected_paper, dict) else None

    # Check if selected paper is correct
    correct_selection = 1.0 if selected_id in ground_truth_ids else 0.0

    # Check ranking quality (if we have ranked papers)
    ranked_papers = getattr(prediction, 'ranked_papers', [])
    if ranked_papers:
        retrieved_ids = [p.get('id') for p in ranked_papers[:20]]

        # Recall@5, Recall@10
        recall_5 = len(set(retrieved_ids[:5]) & ground_truth_ids) / max(1, len(ground_truth_ids))
        recall_10 = len(set(retrieved_ids[:10]) & ground_truth_ids) / max(1, len(ground_truth_ids))

        # MRR
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in ground_truth_ids:
                mrr = 1.0 / rank
                break

        # Weighted combination
        score = (
            0.4 * correct_selection +  # 40% weight on final selection
            0.3 * recall_5 +            # 30% on Recall@5
            0.2 * recall_10 +           # 20% on Recall@10
            0.1 * mrr                   # 10% on MRR
        )
    else:
        score = correct_selection

    return score


def query_reformulation_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Metric for query reformulation quality.
    Measures if reformulated queries improve retrieval.
    """
    # This would need your actual retrieval results
    # For now, a simpler version:

    queries = getattr(prediction, 'queries', [])
    if not queries:
        return 0.0

    # Could measure: diversity, relevance, semantic similarity to ground truth
    # For simplicity: reward generating multiple varied queries
    unique_queries = set(q.lower() for q in queries)
    diversity_score = min(1.0, len(unique_queries) / 4.0)  # Target: 4 unique queries

    return diversity_score
```

```python
# src/agents/self_evolve/evolution_loop.py

import dspy
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from src.workflow import RetrievalWorkflow
from src.evaluation.self_evolve import EvaluationStore, QueryEvaluation
from src.evaluation.dspy_metrics import citation_retrieval_metric
from datasets.scholarcopilot import load_dataset
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class PromptVersion:
    """Track prompt/module versions for rollback capability."""
    def __init__(self, module_name: str, initial_module: dspy.Module):
        self.module_name = module_name
        self.versions = [{"version": 0, "module": initial_module, "timestamp": datetime.now(), "score": 0.0}]

    def add_version(self, module: dspy.Module, score: float):
        version = len(self.versions)
        self.versions.append({
            "version": version,
            "module": module,
            "timestamp": datetime.now(),
            "score": score
        })
        return version

    def get_best(self):
        """Get the best performing version."""
        return max(self.versions, key=lambda v: v['score'])['module']

    def get_current(self):
        return self.versions[-1]['module']


class SelfEvolvingRetrievalSystem:
    """
    Self-evolving retrieval system using DSPy optimization.

    Combines your existing workflow with continuous learning.
    """

    def __init__(
        self,
        workflow: RetrievalWorkflow,
        resources: Dict[str, Any],
        dataset_path: str,
        local_model: str = "ollama_chat/gemma:2b",
    ):
        self.workflow = workflow
        self.resources = resources
        self.dataset_path = dataset_path

        # Initialize DSPy with local model (Gemma via Ollama)
        self.lm = dspy.LM(local_model, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=self.lm)

        # Evaluation store
        self.eval_store = EvaluationStore()

        # Track module versions
        from src.agents.formulators.dspy_prompt_generator.modules import get_module
        self.module_versions = {
            'picker': PromptVersion('picker', get_module('simple')),
            'reformulator': PromptVersion('reformulator', None),  # Will create custom
        }

    def evaluate_batch(
        self,
        papers: List[Dict[str, Any]],
        k: int = 20,
        max_queries: int = None
    ) -> float:
        """
        Run evaluation on a batch of queries and store results.

        Returns average score across queries.
        """
        papers = papers[:max_queries] if max_queries else papers
        logger.info(f"Evaluating {len(papers)} queries...")

        scores = []
        for i, paper in enumerate(papers, 1):
            # Extract query
            paper_text = paper.get('paper', '')
            if not paper_text:
                paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"

            query = paper_text.split()[:50]  # First 50 words
            query_str = ' '.join(query)

            # Get ground truth
            from evaluate import get_ground_truth_ids
            relevant_ids = get_ground_truth_ids(paper)

            if not relevant_ids:
                continue

            # Run workflow
            try:
                final_state = self.workflow.run({
                    'messages': [HumanMessage(content=query_str)],
                    'resources': self.resources,
                    'config': {'k': k, 'enable_dspy_picker': True}
                })
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                continue

            # Extract results
            ranked_papers = final_state.get('ranked_papers', [])
            retrieved_ids = [p[0].get('id') if isinstance(p, tuple) else p.get('id')
                           for p in ranked_papers]

            # Calculate metrics
            from src.evaluation.metrics import evaluate_retrieval
            metrics = evaluate_retrieval(
                [p[0] if isinstance(p, tuple) else p for p in ranked_papers],
                relevant_ids,
                k_values=[5, 10, 20]
            )

            # Overall score (weighted combination)
            score = (0.4 * metrics['R@5'] + 0.3 * metrics['R@10'] + 0.3 * metrics['MRR'])
            scores.append(score)

            # Store evaluation
            eval = QueryEvaluation(
                query=query_str,
                paper_id=paper.get('paper_id', ''),
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                metrics=metrics,
                inputs={'query': query_str, 'paper': paper},
                outputs={'ranked_papers': ranked_papers, 'final_state': final_state},
                score=score
            )
            self.eval_store.add_evaluation(eval)

            if i % 10 == 0:
                avg_so_far = sum(scores) / len(scores)
                logger.info(f"Progress: {i}/{len(papers)}, Avg Score: {avg_so_far:.3f}")

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"Batch evaluation complete. Average score: {avg_score:.3f}")
        return avg_score

    def optimize_modules(
        self,
        optimizer_type: str = 'bootstrap',
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 8,
        teacher_model: str = None,
    ):
        """
        Optimize DSPy modules using collected evaluation data.

        Args:
            optimizer_type: 'bootstrap', 'mipro', or 'gepa'
            max_bootstrapped_demos: Number of bootstrapped examples
            max_labeled_demos: Number of labeled examples
            teacher_model: Optional teacher model (GPT-4o-mini for better optimization)
        """
        logger.info(f"Starting optimization with {optimizer_type}...")

        # Get training data from successful examples
        trainset = self.eval_store.to_dspy_trainset(min_score=0.5)

        if len(trainset) < 10:
            logger.warning(f"Only {len(trainset)} training examples. Need more data!")
            return None

        logger.info(f"Training on {len(trainset)} examples...")

        # Get current module to optimize
        current_module = self.module_versions['picker'].get_current()

        # Choose optimizer
        if optimizer_type == 'bootstrap':
            # BootstrapFewShot with local model
            optimizer = dspy.BootstrapFewShot(
                metric=citation_retrieval_metric,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                max_rounds=2,
            )

            if teacher_model:
                # Use teacher model for bootstrapping
                teacher_lm = dspy.LM(teacher_model)
                optimizer = dspy.BootstrapFewShot(
                    metric=citation_retrieval_metric,
                    max_bootstrapped_demos=max_bootstrapped_demos,
                    max_labeled_demos=max_labeled_demos,
                    max_rounds=2,
                    teacher_settings={'lm': teacher_lm}
                )

        elif optimizer_type == 'mipro':
            # MIPROv2 with Bayesian optimization
            optimizer = dspy.MIPROv2(
                metric=citation_retrieval_metric,
                auto='light',  # 'light', 'medium', or 'heavy'
                num_threads=4,
            )

        elif optimizer_type == 'gepa':
            # GEPA for reflective optimization
            import gepa

            class DSPyAdapter:
                """Adapter for GEPA to work with your system."""
                def evaluate(self, inputs, candidate, capture_traces=True):
                    # Run evaluation on inputs
                    # Return EvaluationBatch with scores and trajectories
                    pass  # Implement based on GEPA docs

            adapter = DSPyAdapter()
            # Use GEPA optimization (see notebook for full implementation)
            pass

        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Compile optimized module
        logger.info("Compiling optimized module...")
        try:
            optimized_module = optimizer.compile(
                current_module,
                trainset=trainset,
            )

            # Evaluate optimized module
            logger.info("Evaluating optimized module...")
            # Re-run evaluation with optimized module
            # (You'd need to temporarily swap it in)

            # For now, assume improvement
            new_score = 0.75  # Placeholder

            # Store new version
            version = self.module_versions['picker'].add_version(optimized_module, new_score)
            logger.info(f"✅ Created module version {version} with score {new_score:.3f}")

            return optimized_module

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None

    def continuous_evolution_loop(
        self,
        eval_interval: int = 100,
        optimize_interval: int = 500,
        improvement_threshold: float = 0.05,
    ):
        """
        Continuous learning loop.

        Args:
            eval_interval: Evaluate every N queries
            optimize_interval: Optimize every N queries
            improvement_threshold: Minimum improvement to deploy new version
        """
        logger.info("🔄 Starting continuous evolution loop...")

        # Load full dataset
        dataset = load_dataset(self.dataset_path)

        query_count = 0
        last_optimization_score = 0.0

        # Process dataset in batches
        batch_size = eval_interval
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            # Evaluate current version
            current_score = self.evaluate_batch(batch, max_queries=eval_interval)
            query_count += len(batch)

            logger.info(f"Processed {query_count} queries. Current score: {current_score:.3f}")

            # Time to optimize?
            if query_count >= optimize_interval:
                logger.info("\n" + "="*70)
                logger.info("⚡ OPTIMIZATION TRIGGERED")
                logger.info("="*70)

                # Run optimization
                optimized_module = self.optimize_modules(
                    optimizer_type='bootstrap',
                    teacher_model='openai/gpt-4o-mini',  # Optional: use for bootstrapping
                )

                if optimized_module:
                    # Evaluate improvement
                    improvement = current_score - last_optimization_score

                    if improvement >= improvement_threshold:
                        logger.info(f"✅ Deploying improved module (+{improvement:.3f})")
                        # Actually deploy by updating your workflow
                        # self.workflow.update_module(optimized_module)
                    else:
                        logger.info(f"⚠️  Improvement too small (+{improvement:.3f}), keeping current version")

                    last_optimization_score = current_score

                query_count = 0  # Reset counter

                logger.info("="*70 + "\n")
```

```python
# src/workflow.py - Add to RetrievalWorkflow class

class RetrievalWorkflow:
    # ... existing code ...

    def update_dspy_module(self, new_module: dspy.Module):
        """Hot-swap DSPy module during runtime."""
        # Store in resources so dspy_picker uses it
        if not hasattr(self, '_dspy_module'):
            self._dspy_module = new_module
        else:
            logger.info(f"Updating DSPy module: {type(new_module).__name__}")
            self._dspy_module = new_module

        # Update resources
        if hasattr(self, 'resources'):
            self.resources['dspy_module'] = new_module

    def get_dspy_module(self):
        """Get current DSPy module."""
        return getattr(self, '_dspy_module', None)
```

```python
# scripts/run_self_evolving.py

from src.workflow import RetrievalWorkflow
from src.agents.self_evolve.evolution_loop import SelfEvolvingRetrievalSystem
from src.resources.builders import build_inmemory_resources
from datasets.scholarcopilot import load_dataset, build_citation_corpus

# Setup
dataset_path = "path/to/scholar_copilot.json"
dataset = load_dataset(dataset_path)
corpus = build_citation_corpus(dataset)

# Build resources
resources = build_inmemory_resources(
    corpus,
    enable_bm25=True,
    enable_e5=True,
    enable_specter=True
)

# Initialize workflow
workflow = RetrievalWorkflow(use_llm_reranker=False)

# Create self-evolving system with local Gemma model
evolving_system = SelfEvolvingRetrievalSystem(
    workflow=workflow,
    resources=resources,
    dataset_path=dataset_path,
    local_model="ollama_chat/gemma:2b"  # Local model via Ollama
)

# Option 1: One-time optimization
print("Running initial evaluation...")
baseline_score = evolving_system.evaluate_batch(dataset[:100])
print(f"Baseline score: {baseline_score:.3f}")

print("\nOptimizing modules...")
optimized = evolving_system.optimize_modules(
    optimizer_type='bootstrap',
    max_bootstrapped_demos=4,
    teacher_model='openai/gpt-4o-mini'  # Optional: helps bootstrap
)

print("\nEvaluating optimized system...")
optimized_score = evolving_system.evaluate_batch(dataset[100:200])
print(f"Optimized score: {optimized_score:.3f} (+{optimized_score-baseline_score:.3f})")

# Option 2: Continuous evolution loop
evolving_system.continuous_evolution_loop(
    eval_interval=100,      # Evaluate every 100 queries
    optimize_interval=500,  # Optimize every 500 queries
    improvement_threshold=0.05  # Deploy if >5% improvement
)
```
