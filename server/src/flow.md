\subsubsection{Agent Components}

The system consists of the following specialized agents:

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

The system implements a continuous improvement loop using DSPy for automatic prompt optimization. Training data is collected from query results: positives (gold label citations), negatives (highly-ranked irrelevant papers), and hard negatives (papers failing evaluation metrics). DSPy optimizers (BootstrapFewShot, MIPRO, MIPROv2) experiment with different instruction phrasings, reasoning strategies, and few-shot examples, selecting prompts that maximize retrieval metrics (R@k, MRR). The optimization loop iterates until quality thresholds are met or maximum steps reached, creating a self-improving system where prompts evolve based on retrieval outcomes.

\textbf{Current Status:} The DSPy self-evolution pipeline is the primary goal and actively under development. Currently, the optimizer is being tested separately using GPT-5.2-mini to generate optimized prompts offline. The main objective is to fully integrate DSPy within the LangGraph workflow for real-time, on-the-fly prompt optimization and continuous learning from retrieval outcomes.
