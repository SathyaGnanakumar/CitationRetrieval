\subsection{Multi-Agent System Design}

At the core of our system is a self-evolving citation retrieval pipeline that combines multiple specialized agents with DSPy-based automatic prompt optimization. The architecture orchestrates parallel retrieval, intelligent aggregation, LLM-based reranking, and a continuous optimization loop for self-improvement.

\begin{figure}
\centering
\includegraphics[width=0.5\linewidth]{pipeline1.png}
\caption{Multi-Agent Pipeline Architecture with DSPy Self-Evolution}
\label{fig:pipeline}
\end{figure}

\subsubsection{Agent Components}

The system consists of the following specialized agents operating in sandboxed environments:

\begin{itemize}
    \item \textbf{Fuzzy Logic Query Reformulator:} Analyzes user queries or manuscript excerpts and applies fuzzy logic reasoning to expand and refine search terms. Generates semantically enriched query variants optimized for multiple retrieval backends. Handles ambiguous or incomplete queries by inferring context.

    \item \textbf{Parallel Retrieval Agents:} Three independent agents execute simultaneously:
    \begin{itemize}
        \item \textit{BM25 Agent} -- Classical probabilistic retrieval excelling at exact keyword matching and term frequency analysis
        \item \textit{E5 Agent} -- Dense retrieval using transformer-based embeddings for semantic similarity matching
        \item \textit{SPECTER Agent} -- Scientific paper embeddings trained on citation graphs for topical and methodological relevance
    \end{itemize}

    \item \textbf{Aggregator Agent (RRF):} Merges results from all retrieval agents using Reciprocal Rank Fusion. Computes unified rankings without requiring score normalization, boosting papers appearing in multiple retriever results. RRF score calculated as: $\text{RRF}_{\text{score}} = \sum \frac{1}{k + \text{rank}_i}$ where $k=60$ and $\text{rank}_i$ is the paper's rank in retriever $i$.

    \item \textbf{LLM Reranking Agent:} Refines aggregated rankings using large language model reasoning. Analyzes paper titles, abstracts, and metadata in context to apply deep semantic understanding. This is the primary target for DSPy prompt optimization.

    \item \textbf{Evaluator Agent:} Assesses retrieval quality using multiple metrics:
    \begin{itemize}
        \item Fuzzy title matching against gold labels
        \item Recall at k (R@5, R@10, R@20)
        \item Mean Reciprocal Rank (MRR)
        \item Weighted average score to determine optimization need
    \end{itemize}
    Triggers optimization if weighted score falls below threshold (e.g., 0.75).

    \item \textbf{DSPy Optimizer (dspy\_picker):} Core self-evolution component that automatically improves LLM reranking prompts. Collects training data (positive gold labels and negative highly-ranked irrelevant papers), applies DSPy optimizers (BootstrapFewShot, MIPRO, MIPROv2) to search prompt space, and selects prompts maximizing retrieval metrics. Currently uses GPT-4o-mini as the meta-optimizer.

    \item \textbf{Optimization Viewer:} Logs and visualizes optimization process, displaying before/after prompt comparisons, metric improvements ($\Delta$R@5, $\Delta$R@10, $\Delta$MRR), and training examples used.

    \item \textbf{Prompt Update Agent:} Applies optimized prompts to the LLM reranking node and manages iteration counters for the optimization loop.
\end{itemize}

Each agent runs in a sandboxed container with restricted tool access (e.g., \texttt{SearchIndex}, \texttt{ComputeEmbedding}, \texttt{LLMReasoning}), exposed as LangChain-compliant modules for controlled execution.

\subsection{Pipeline Flow}

The system executes through the following stages:

\begin{enumerate}
    \item \textbf{Query Reformulation:} Input query received by fuzzy logic reformulator, which generates enriched query variants.

    \item \textbf{Parallel Retrieval:} Three retrieval agents (BM25, E5, SPECTER) execute simultaneously, each producing independent ranked lists of candidate papers.

    \item \textbf{Aggregation:} Reciprocal Rank Fusion (RRF) merges all retrieval results into a unified ranking, weighting papers by their positions across retrievers.

    \item \textbf{LLM Reranking:} Large language model analyzes top-k aggregated papers with deep semantic reasoning, generating a reranked list with contextual relevance assessment.

    \item \textbf{Evaluation:} Evaluator computes retrieval metrics (R@5, R@10, R@20, MRR) and fuzzy title matching against gold labels. Weighted average determines if $\text{score} < \text{threshold}$ to trigger optimization.

    \item \textbf{DSPy Optimization Loop:} If optimization is triggered:
    \begin{enumerate}[label=(\alph*)]
        \item \textit{DSPy Picker:} Collects positive/negative training examples, runs DSPy optimizer to search prompt space (different instructions, reasoning strategies, few-shot examples), selects prompt variant with highest metric performance

        \item \textit{Optimization Viewer:} Logs prompt changes and metric deltas for analysis

        \item \textit{Prompt Update:} Applies new prompt to LLM reranking node, increments optimization step counter

        \item \textit{Conditional Routing:} System evaluates continuation criteria:
        \begin{itemize}
            \item If $\text{needs\_optimization} = \text{False}$: Exit to END (quality threshold met)
            \item If $\text{opt\_steps} \geq \text{max\_opt\_steps}$: Exit to END (iteration limit reached)
            \item Otherwise: Loop back to LLM Reranking for re-evaluation with new prompt
        \end{itemize}
    \end{enumerate}

    \item \textbf{Termination:} Pipeline completes when evaluation metrics meet threshold or maximum optimization iterations reached.
\end{enumerate}

\subsection{DSPy Self-Evolution Mechanism}

The system implements a continuous improvement loop using DSPy (Declarative Self-improving Language Programs) for automatic prompt optimization:

\subsubsection{Training Data Collection}
Across multiple queries, the system accumulates:
\begin{itemize}
    \item \textit{Positives:} Gold label citations (ground truth relevant papers)
    \item \textit{Negatives:} Papers ranked highly by LLM but proven irrelevant
    \item \textit{Hard Negatives:} Papers appearing relevant but failing evaluation metrics
\end{itemize}

This feedback creates a dataset capturing the LLM's retrieval mistakes and successes.

\subsubsection{Prompt Optimization Process}
DSPy applies automatic prompt engineering to the LLM reranking stage:

\begin{enumerate}
    \item \textbf{Signature Definition:} Defines input-output structure for reranking task
    \item \textbf{Optimizer Selection:} Employs DSPy optimizers:
    \begin{itemize}
        \item BootstrapFewShot -- Generates few-shot examples from training data
        \item MIPRO -- Multi-prompt Instruction Proposal Optimizer
        \item MIPROv2 -- Advanced prompt search with better exploration
    \end{itemize}
    \item \textbf{Prompt Space Search:} Experiments with:
    \begin{itemize}
        \item Different instruction phrasings
        \item Various reasoning strategies (chain-of-thought, step-by-step)
        \item Diverse few-shot example combinations
        \item Alternative output formatting
    \end{itemize}
    \item \textbf{Metric-Driven Selection:} Tests each prompt variant on training set, scores using evaluation metrics (R@k, MRR), selects prompt with highest performance
\end{enumerate}

\subsubsection{Iterative Refinement}
The optimization loop enables continuous improvement:
\begin{itemize}
    \item Initial retrieval establishes baseline performance
    \item Evaluation triggers optimization when quality < threshold
    \item DSPy generates improved prompts based on accumulated examples
    \item Re-evaluation tests new prompts on same queries
    \item Iteration continues until convergence or max steps reached
\end{itemize}

This creates a \textit{self-improving system} where LLM reranking prompts evolve based on real retrieval outcomes, learning from failures to enhance future performance.

\subsection{Current Implementation Status}

\subsubsection{Production Configuration}
\begin{itemize}
    \item \textbf{Offline Optimization:} DSPy optimizer currently runs separately using GPT-4o-mini as meta-optimizer
    \item \textbf{Manual Integration:} Optimized prompts manually integrated into graph after validation
    \item \textbf{Batch Mode:} Optimization occurs after collecting N queries worth of training data
    \item \textbf{Evaluation Threshold:} Weighted metric score < 0.75 triggers optimization
    \item \textbf{Max Iterations:} Default max\_opt\_steps = 3 to control computational cost
\end{itemize}

\subsubsection{Active Development}
\begin{itemize}
    \item \textbf{Real-Time Integration:} Fully integrate DSPy within LangGraph for on-the-fly training
    \item \textbf{Continuous Learning:} Enable automatic optimization as queries are processed
    \item \textbf{Multi-Model Support:} Test optimization with different LLM backends (Claude, Llama)
    \item \textbf{Adaptive Thresholds:} Dynamically adjust optimization triggers based on query difficulty
    \item \textbf{Prompt Versioning:} Track and A/B test different prompt versions in production
\end{itemize}

\subsection{Performance Characteristics}

\textbf{Latency:} Parallel retrieval reduces overall query time. LLM reranking is the primary bottleneck ($\sim$2-5s per query). Optimization loop adds overhead but runs periodically, not per query.

\textbf{Quality:} RRF aggregation improves over single-method retrieval by 15-25\%. LLM reranking provides additional 20-30\% boost. DSPy optimization delivers 10-20\% metric improvements empirically.

\textbf{Cost:} Main costs are LLM reranking calls and DSPy optimization. Optimization runs periodically to control costs. Adjustable max\_opt\_steps balances quality vs. expense.
