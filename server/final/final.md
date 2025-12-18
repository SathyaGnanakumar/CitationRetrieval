\documentclass[11pt,a4paper]{article}
\usepackage{times}
\usepackage{latexsym}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{subcaption}

\usepackage{url}

\usepackage{indentfirst}

\title{Multi-Agent System for Reliable Citation Retrieval}

\author{
Sathya Gnanakumar \quad {\tt sgnanaku@umd.edu} \and
Ishaan Kalra \quad {\tt ishaank@umd.edu} \and
Vishnu Sreekanth \quad {\tt vishnus@umd.edu} \and
Dhruv Suri \quad {\tt dsuri@umd.edu} \and
Vibhu Singh \quad {\tt vibhu307@umd.edu} \and
Kushal Kapoor \quad {\tt kushalk@umd.edu} \and
}

% \setlength\textwidth{16.0cm}
\date{}

\begin{document}
\maketitle

\section{Abstract}

Accurate citation is vital to upholding scientific integrity. However, the process of identifying and formatting references remains a significant bottleneck in the academic writing workflow. This project proposes the development of a Citation Finder, a multi-agent system designed to autonomously retrieve, verify, and recommend academic references given a query or document excerpt. We evaluate our system against information retrieval baselines and single-agent models using the ScholarCopilot benchmark. By focusing on metrics such as recall@K and citation accuracy, we aim to demonstrate that a multi-agent approach can significantly reduce hallucination rates and streamline the literature review process for researchers.

ADD OUR EXACT METRICS TO ABSTRACT WHEN WE ARE DONE TO QUANTIFY IMPACT

\section{Introduction}

Trustworthy citations are at the heart of quality research papers. Citations serve as the bridge between past work and current research, demonstrating credibility and providing further resources for readers. However, the process of determining, verifying, and formatting citations is not so simple. When starting a paper, it can be a tedious process to find papers that are relevant to the topic at hand, and even when sources are found, it can be difficult for one to keep track of the reference that they are currently sourcing information from. Another issue is hallucinated references, as people have increasingly become reliant on large language models (LLMs), which often provide references to papers that do not exist or provide incorrect evidence.

Thus, a streamlined, accurate method of finding relevant research papers that goes beyond a straightforward keyword-based search would provide extreme utility to authors and researchers. As such, for our project, we seek to develop and train a multi-agent model system capable of retrieving and recommending relevant papers for citation purposes, given an input such as a research topic, query, or existing manuscript.

To evaluate the performance of our multi-agent system, we rigorously compared its performance against baseline BM25 search, dense retrieval models, and an LLM model with re-ranking. We used ScholarCopilot's large training dataset of computer science papers scraped from ArXiv as a training corpus to help guide our model in learning about writing patterns and best citation practices (Wang et al. 2025). As part of this study, we also leveraged the metrics presented in the following two studies: \textit{LitSearch: A Retrieval Benchmark for Scientific Literature Search} by Ajith et al., 2024, and \textit{CiteME: Can Language Models
Accurately Cite Scientific Claims?} by Press et al., 2024. Our goal is to help authors quickly identify relevant prior work to cite in their papers, thereby reducing the risk of hallucinated citations. We hope to apply this system to various fields such as law and medicine.

\section{Related Work}
To better understand the performance of current state-of-the-art citation retrieval models and obtain a clearer direction for our project, we examined multiple papers, each approaching the issue of citation retrieval in different ways.

Ajith et al. introduce LitSearch, a comprehensive retrieval benchmark based on 597 curated literature-search queries, each paired with cited papers beforehand, to evaluate the performance of sparse versus dense retrieval models. It finds that dense, neural network-based retrieval models significantly outperform sparse retrieval models, namely BM25, by 24.8\% in recall@5. Furthermore, the authors found that reranking with LLMs provided an additional +4.4 improvement over the best retriever. It also highlights that standard searches such as Google and Google Scholar lag significantly behind both sparse and dense retrieval models \cite{ajith2024litsearch}.

Press et al. present CiteME, which reframes the problem by asking whether humans, closed-source language models, and an autonomous retrieval-augmented agent can correctly recover a citation when it is masked from a scientific excerpt. It finds that currently, humans perform the best, achieving 69.7\% accuracy, followed by CiteAgent, an autonomous model-based agent, achieving 35.5\%, followed by closed language models at a mere 4.2 - 18.5\% \cite{press2024citeme}.

Wang et al. present ScholarCopilot, a framework which explores a different dimension of citation retrieval. ScholarCopilot, which is trained on 500K papers sourced from arXiv, generates a retrieval token which is matched against a citation database in order to determine whether a certain citation should be retrieved for LLM-generated text during the generation process. It yields a 40.1\% top-1 retrieval accuracy, while achieving 100\% preference in citation quality over ChatGPT. In the training dataset, in-text citations in the body of a paper are replaced by placeholder keys, which correspond to entries in a bibliography info field included alongside that paper \cite{wang2025scholarcopilot}.

Lalá et al. present PaperQA, a RAG agent that retrieves scientific papers not merely to provide citations, but to improve the quality of LLM-generated answers to a query. On LitQA, a benchmark comprising of 50 multiple choice questions, PaperQA achieves an accuracy of 69.5\%, outperforming both expert humans (66.8\%) and standalone LLMs/frameworks (in the range of 24\% - 40.6\%). PaperQA is also the only subject to achieve a hallucination rate of 0\%, underscoring how RAG can enhance answer accuracy while minimizing the risk of potentially generating false information \cite{lala2023paperqa}.

Ajith et al. and Press et al. provide strong evidence that searches, retrieval models, and language models still have a long way to go in terms of effective citation retrieval \cite{ajith2024litsearch, press2024citeme}. However, CiteME reveals that an autonomous agent with searching and information retrieval capabilities beyond closed models has growing potential, being nearly twice as effective as the best closed language model \cite{press2024citeme}. Furthermore, through PaperQA's outstanding performance, Lalá et al. reveal the potential of a RAG agent in ensuring accuracy and eliminating hallucinations \cite{lala2023paperqa}.

Cherian et al. present a multi-agent literature retrieval framework that combines sparse BM25 indexing with dense FAISS similarity search. Their architecture consists of a Query Agent for input refinement, a Retrieval Agent for hybrid search, and a Learning Agent that incorporates user feedback to improve personalization over time. The system reports modest improvements in precision (+8.5\%), recall (+7.3\%), and response latency relative to PaperQA and Semantic Scholar, and additionally measures hallucination rates for RAG-based answers. Their work demonstrates the value of multi-agent systems and hybrid retrieval alongside personalization features for improved literature search \cite{cherian2025multiagent}.

This motivates our project idea of building a multi-agent citation finder system that performs more structured literature retrieval and leverages multiple retrieval methods \cite{ajith2024litsearch,cherian2025multiagent,lala2023paperqa,press2024citeme}. Additionally, the strong performance of ScholarCopilot suggests that organizing training data with explicit citation placeholders linked to bibliographic entries provides a useful structure for citation prediction \cite{wang2025scholarcopilot}. Building on these insights, we aim to design a multi-agent system that improves the speed and consistency of scientific literature search.

\section{Methods}

Our approach combines traditional information retrieval with a multi-agent architecture powered by LangGraph. The goal is to balance the reliability of various baseline keyword searches. Below we describe the various components of our approach, our baseline results, and the architecture of our pipeline.

\subsection{Key Components}

\subsubsection{Sparse Retrieval}

Sparse retrieval relies on exact keyword matching to score documents based on the frequency and distribution of query terms within the corpus. Due to its computational efficiency and low latency, it serves as a strong baseline for understanding retrieval performance over our citation corpus. In this work, we employ BM25s, a Python library implementing the BM25 ranking algorithm, to estimate document relevance with respect to a given search query. The resulting ranking function assigns higher scores to documents that are more likely to be relevant to the query, enabling effective initial retrieval. \cite{lù2024bm25s}

\subsubsection{Dense Retrieval}

We also leverage the power of dense retrieval in our approach, in order to go beyond a simple keyword-based comparison. The embeddings created by dense retrieval models attempt to capture the true essence of what a document actually means through many iterations of fine-tuning across a set of carefully configured layers, a dimension largely unexplored by sparse retrieval.
Specifically, we use E5-Large and SPECTER-2. E5-Large serves as a strong, generalized semantic dense retriever that can effectively adapt to a large variety of contexts, in our case academic excerpts, and produce meaningful embeddings. \cite{wang2024textembeddings} SPECTER-2 complements this by providing a retrieval approach tailored specifically to scientific papers, as it focuses on document-level relatedness through using citation graphs, ensuring that its embeddings better capture the full semantic scope of scientific documents. \cite{cohan2020specter}

\subsubsection{Aggregation and Reranking}

We aggregate the outputs of multiple heterogeneous retrievers—specifically a sparse BM25 retriever and dense embedding-based retrievers (E5-Large and SPECTER-2)—to produce results that capture both lexical and semantic relevance. Rather than relying on raw retrieval scores, we employ \emph{Reciprocal Rank Fusion} (RRF) as our aggregation method. RRF combines ranked lists by assigning each document a fused score based solely on its rank within each retriever:
\[
\text{RRF}(d) = \sum\_{i=1}^{N} \frac{1}{k + \text{rank}\_i(d)}
\]
where $\text{rank}_i(d)$ denotes the rank of document $d$ in retriever $i$, and $k$ is a smoothing constant that controls the influence of top-ranked results. Documents that appear consistently across multiple retrievers are rewarded with higher aggregate scores, while the method remains robust to score calibration issues and outliers.

Prior work has shown that RRF outperforms both Condorcet-based fusion and learned rank aggregation methods while remaining simple and computationally efficient, making it well-suited for multi-retriever systems \cite{cormack2009reciprocal}. In our implementation, we set $k=60$, following common practice, and use RRF as the default aggregation mechanism. We additionally support a simple max-score aggregation baseline for debugging and ablation purposes.

After aggregation, we apply a reranking stage to refine the candidate list. This reranker leverages a stronger cross-encoder-style model to reassess relevance at a finer granularity, improving final citation quality prior to selection \cite{gao2025llm4rerank}.

% \subsubsection{DSPy Integration}

% To select the final cited paper, we leverage DSPy, a framework that enables modular LLM reasoning by organizing the text transformation process into structured, learnable components. We will integrate DSPy as the final paper selection stage in our citation retrieval pipeline. After traditional information retrieval methods retrieve and rank candidate papers, DSPy will use LLM-based reasoning to identify the correct citation from the candidate pool. \cite{khattab2023dspy}

% \paragraph{How DSPy Works: Problem Setup}
% \begin{itemize}
% \item \textbf{Input}: A citation context (text excerpt where a citation was removed, marked with \texttt{[CITATION]}) and a list of candidate papers with titles and abstracts
% \item \textbf{Output}: The paper that should be cited at that location, along with reasoning
% \end{itemize}

% \paragraph{DSPy Signatures}

% Four task-specific signatures define the input/output contracts:

% \begin{itemize}
% \item \textbf{Citation Retrieval}: Given context and candidates, select the cited paper with step-by-step reasoning
% \item \textbf{Query Generation}: Generate an effective search query from the citation context
% \item \textbf{Citation Reranking}: Rank all candidates by relevance to the context
% \item \textbf{Citation Verification}: Verify if a specific paper matches the citation (returns match/no-match with confidence)
% \end{itemize}

\subsubsection{DSPy Prompt Design for Final Citation Retrieval}
\paragraph{Core Idea}
To select the final cited paper, we leverage DSPy, a framework that enables modular LLM reasoning by organizing the text transformation process into structured, learnable components \cite{khattab2023dspy}. We use it as the \textbf{final citation picker}: given a citation context and a closed set of candidate papers (already retrieved and reranked), it selects exactly one paper to fill the missing \texttt{[CITATION]}. The core contribution is the prompt/program design: we use a DSPy signature to define a structured prompt with explicit input/output fields, and we serialize candidates into a constrained format so the model must choose from the provided set rather than inventing references.
\paragraph{Signature-Driven Prompt Construction}
Instead of hand-writing a single large prompt string, we define the citation selection task via a DSPy \emph{signature}, which specifies:
\begin{itemize}
\item required inputs (citation context + candidate list),
\item required outputs (selection + reasoning),
\item task instructions embedded in the signature description.
\end{itemize}
Concretely, our signature \texttt{CitationRetrieval} takes:
\begin{itemize}
\item \textbf{\texttt{citation_context}}: excerpt containing \texttt{[CITATION]} where a reference was removed,
\item \textbf{\texttt{candidate_papers}}: text-formatted list of candidate paper titles and abstracts,
\end{itemize}
and produces:
\begin{itemize}
\item \textbf{\texttt{reasoning}}: step-by-step analysis of which candidate matches the context,
\item \textbf{\texttt{selected_title}}: the exact title of the selected paper.
\end{itemize}
We implement this using \texttt{dspy.ChainOfThought(CitationRetrieval)}, which forces the model to generate an explanation and an explicit selection.
\paragraph{Exact Prompt Template Used}
Below is the actual prompt template produced by our DSPy signature and candidate formatting. It is designed as a \emph{closed-world} selection task: the model must select from the candidate list.
\begingroup\footnotesize
\begin{verbatim}
You are an expert citation retrieval system as described in the paper
"Multi-Agent System for Reliable Citation Retrieval".
Your goal is to autonomously retrieve, verify, and recommend academic
references given a query or document excerpt.
Task:
Given a citation context from a scientific paper (where a citation is missing),
identify the correct paper from a list of candidates.
Analyze the context to understand the specific claim, method, or result being cited.
Then, evaluate each candidate paper to see if it matches the context.
Finally, select the best matching paper.
Citation Context: \_\_\_,
Candidate Papers:

1. Title:
   Abstract:
2. Title:
   Abstract:
   Think step-by-step about which candidate best matches the context.
   Summarize your reasoning, then provide the exact title that should fill the citation.
   \end{verbatim}
   \endgroup
   \paragraph{Candidate Serialization (What the Prompt Sees)}
   DSPy expects \texttt{candidate_papers} as text, so we convert the top-$N$ reranked candidates into a numbered list. Each candidate includes a title and a truncated abstract to control prompt length. This structure makes it easy for the model to compare candidates and helps reduce hallucinations by grounding the decision in a fixed list.
   \paragraph{Output Contract and Grounding}
   The DSPy output is constrained to:
   \begin{itemize}
   \item \textbf{a natural-language rationale} (\texttt{reasoning}),
   \item \textbf{a single exact selection} (\texttt{selected_title}).
   \end{itemize}
   The system then maps \texttt{selected_title} back to an actual candidate paper object via exact title matching. If the predicted title does not match any candidate, the system falls back to the top reranked candidate to ensure it always outputs a valid retrieved paper.
   \paragraph{Continual Prompt Updating with DSPy (Prompt/Program Optimization)}

A key reason we use DSPy is that it provides an explicit mechanism to \textbf{continually improve} the final-picker prompt/program as new supervision becomes available. Importantly, this continual improvement does \emph{not} update the underlying LLM weights; instead, DSPy optimizes the \textbf{instructions and demonstrations} used at inference time (i.e., the prompt/program).

\paragraph{Training signal.}
From ScholarCopilot-style data, we construct labeled examples of the form:
(citation context, ground-truth cited paper title) along with a candidate pool containing the positive paper and sampled negative papers (titles + abstracts). Candidates are shuffled so the positive is not always in a fixed position.

\paragraph{Optimization objective.}
We optimize top-1 citation selection by maximizing an exact-match metric on the predicted \texttt{selected_title} (case-insensitive exact match against the ground-truth title). This directly measures whether the picker chooses the correct citation from the candidate set.

\paragraph{Prompt compilation loop.}
We treat the DSPy picker as a continuously improving component via periodic recompilation:
\begin{enumerate}
\item Collect new labeled citation examples (contexts with verified ground-truth citations).
\item Add them to the training set and regenerate candidate pools with hard negatives.
\item Run DSPy compilation (e.g., \texttt{BootstrapFewShot} or \texttt{MIPROv2}) to optimize the citation-selection program for the metric above.
\item Save the compiled program to disk (serialized) and deploy it for inference.
\end{enumerate}

At inference time, the system always uses the latest compiled DSPy program, so as the dataset grows, the selection prompt/program is continually updated without changing the retrieval pipeline or retraining the base LLM.

\subsection{Baselines}
We first establish baselines using our chosen retrieval methods. For the sparse baseline, we run BM25, and for the dense baselines, we use E5-Large, SPECTER-2, and OpenAI embeddings coupled with LLM reranking. The datasets that we use for these baselines are the ScholarCopilot evaluation dataset, which contains 1000 academic papers in the same format as the ScholarCopilot training dataset, and a 200-example citation evaluation set. We evaluate performance using recall@5. We also collected recall@10 for the baselines using the latter dataset.

\begin{itemize}
\item \textbf{BM25:} BM25 provides a strong baseline despite having zero training and no deeper understanding of surrounding contexts. This was run with a reference corpus built from all cited papers found in the 1000 papers of the ScholarCopilot evaluation dataset. The queries included all sentences with removed in-text citations and their surrounding contexts. With this training setup, a recall@5 score of 29.2\% is achieved. On the 200-example evaluation set, BM25 achieves a recall@5 of 36.6\%, and a recall@10 of 50\%.

    \item \textbf{E5-Large:} When similarly run with the corpus built from the 1000-paper evaluation dataset and excerpts with removed citations as queries, E5 performs poorly, with a recall@5 of 19.3\%. However, on the 200-example evaluation set, it achieves a recall@5 of 51.5\%, and a recall@10 of 67\%.

    \item \textbf{SPECTER-2:} With the same 1000-paper evaluation dataset paradigm, SPECTER-2 performs very poorly, with a recall@5 of 9.7\%. Due to compute constraints that we faced early on, we were not able to run it with the 200-example evaluation set.

    \item \textbf{OpenAI Embeddings:} Running OpenAI embeddings with the 200-example evaluation set only, we observe a recall@5 of 47.4\% and  recall@10 of 66\%, a performance on par with E5-Large on this same 200-example set.

    \item \textbf{OpenAI Embeddings with LLM Reranking:} Due to further compute constraints, we only implement LLM reranking on top of the OpenAI embeddings, again on the 200-example evaluation set. We find a significant increase in performance, with a recall@5 of 57.2\% and a recall@10 of 73.2\%.

    \begin{figure}
        \centering
        \includegraphics[width=\linewidth]{baselines1.png}
        \caption{Baseline Results for Retrieval Models}
        \label{fig:placeholder}
    \end{figure}

\end{itemize}

These baselines are essential for benchmarking our multi-agent system and provide fallback options for difficult queries.

\subsection{Multi-Agent System Design}
At the core of our system is a set of specialized agents operating in sandboxed environments with access to controlled tools:
\begin{figure}
\centering
\includegraphics[width=0.5\linewidth]{pipeline1.png}
\caption{Multi-Agent Pipeline Architecture with Enhanced Retrieval}
\label{fig:placeholder}
\end{figure}
\begin{itemize}
\item \textbf{Coordinator Agent:} Decomposes the user query or manuscript excerpt into sub-tasks, orchestrates the workflow, and aggregates results. Implemented using LangGraph’s Deep Agent framework for planning and delegation.
\item \textbf{Retrieval Agents:} Parallel agents that independently query BM25, E5, and SPECTER-2. Their results are forwarded to the aggregator agent..
\item \textbf{Aggregator Agent}: Accumulates the results from the three retrieval agents and uses RRF.
\item \textbf{Cross-Encoder Reranking} $\rightarrow$ Neural reranker scores query-title pairs
\item \textbf{DSPy Selection} $\rightarrow$ LLM analyzes context and selects the final citation
\end{itemize}

Each agent runs in a sandboxed container, similar to Manus’s design, with a restricted virtual file system and scoped tool access. Tools are exposed as LangChain or MCP-compliant modules, e.g., \texttt{SearchWeb}, \texttt{ReadPDF}, or \texttt{RunPython}, allowing controlled execution.

\subsection{Pipeline Flow}
The system executes as follows:
\begin{enumerate}
\item Input query or excerpt is received by the Coordinator.
\item Retrieval Agents perform parallel searches (BM25, dense).
\item Results are collated and passed to the Aggregator Agent for filtering.
\item Cross-Encoding Reranker compiles the final ranked citation list based on the highest scores generated for the aggregated citations.
\item DSPy will modularly reason to determine the final citation from this list.
\end{enumerate}

\section{Deliverables}
We have completed the following items:
\begin{itemize}
\item \textbf{Codebase:} Sparse retrieval implementation, dense retrieval model implementation, autonomous model-based agents, citation verification, citation formatting, unit tests, and a web demo interface.
\item \textbf{Data:} A processed corpus of ScholarCopilot Data
\item \textbf{Documentation:} A usage guide provided in our README with specific instructions on how to run our pipeline and various flags to test specific agents, change batch size, number of queries, and recall
\item \textbf{Visualizations:} Multi-Agent Pipeline Architecture Diagram and evaluation plots (recall curves)
\end{itemize}

\section{Validation Methods}

To rigorously assess our system, we will evaluate on two recently introduced benchmarks designed for scientific citation retrieval. First, we use LitSearch (Ajith et al., 2024), a retrieval benchmark with 597 curated search queries paired with ground-truth citations. Performance is measured using recall@K, which calculates the proportion of relevant cited papers that appear in the system’s top-K results, averaged across all queries. For example, if a query cites three papers and two of them are retrieved within the top 5 results, that query achieves a recall@5 of 66.7\%. Averaged across all queries, prior work finds BM25 achieves ≈50\% recall@5, while dense retrievers reach ≈75\%. This metric captures how well a system surfaces known-relevant literature within the top results that a researcher is likely to inspect.
% First, we use LitSearch (Ajith et al., 2024), a retrieval benchmark with 597 curated search queries paired with ground-truth citations. We report recall@K, measuring the proportion of queries where the cited paper appears among the top K retrieved results. This allows direct comparison against established baselines: BM25 (≈50\% recall@5) and dense neural retrieval models (≈75\% recall@5).

Second, we evaluate on CiteME (Press et al., 2024), a benchmark for citation attribution that presents text excerpts with masked citations and requires systems to recover the referenced paper. Accuracy is computed as the percentage of excerpts for which the system’s predicted citation exactly matches the ground-truth cited paper. For example, a system that correctly identifies the referenced work in 46 out of 130 excerpts would score 35.3\% accuracy, as reported for CiteAgent. Prior results show humans achieving 69.7\% accuracy, autonomous single-agent systems such as CiteAgent at ~35\%, and closed-source LMs at only 4–18\%.
% Performance is measured in terms of accuracy, with prior results showing humans at 69.7\%, autonomous single-agent systems such as CiteAgent at ~35\%, and closed-source LMs at 4–18\%.

Beyond benchmark-specific metrics, we will also measure the hallucination rate, defined as the percentage of outputs that reference nonexistent or fabricated papers. While hallucination is not part of the LitSearch or CiteME benchmarks, it is highly relevant for real-world usage where LLM-based systems are prone to generating invalid references.

Our multi-agent system will be validated against both baselines and single-agent models. We will use the LitSearch and CiteME benchmarks, along with hallucination rate analysis, to test our multi-agent model and evaluate its effectiveness in citation retrieval. These tests will be run using PyTest and a Github Actions workflow that executes our tests upon committing code.

\section{Unit Tests}

In order to ensure that our system works as a whole, we run unit tests on each individual agent in the pipeline. Our unit tests primarily look for proper formatting of the output from each agent. We do not validate correctness of the output values or evaluate them; rather, we ensure that the format which the agent produces can be used by the next agent in the pipeline.

For all agents in the pipeline, we wrote unit tests to ensure that the data returned was of the proper type. For most of the agents, this means a dictionary containing values used by the next agent in the query: for example, we check that the retrieval agents return a document ID, score for that document, and the source. To ensure that inputs were controlled throughout each unit test, we created mock results that might have been returned dby prior agents in the pipeline, such as a fake query and query expansion to provide to the retrieval agents.

For each unit in our system (that is, each agent in the pipeline), we created the following unit tests:

\begin{itemize}
\item \textbf{Query Reformulator:}

    \textbf{Keyword extraction} (extracts only words with more than 3 characters, converts all keywords to lowercase, handles empty query), \textbf{Keyword expansion} (expands known keywords, does not expand unknown keywords, handles empty keywords list), \textbf{Academic style rewriting} (produces formatted string, includes first three expansions), \textbf{Query reformulation} (returns dictionary, includes query, queries, and messages keys, handles no human message, uses last human message, handles empty keywords list, expanded queries include expansion, no whitespace in query)

    \item \textbf{BM25:}

    \textbf{Get queries} (returns queries when provided as a list, filters out empty strings from queries list, returns empty list when no queries provided), \textbf{No resources} (returns error message when no queries provided, resources missing), \textbf{Mocked resources} (returns dictionary, result contains id, title, score, source, returns correct number of results, passes k value from config to retriever)

    \item \textbf{E5:}

    \textbf{Get queries} (returns queries when provided as a list, filters out empty strings from queries list, returns empty list when no queries provided), \textbf{No resources} (returns error message when no queries provided, resources missing), \textbf{E5 retriever} (single query returns a list of results, each result contains id, title, source, and score, batch query returns a list of result lists) \textbf{Mocked resources} (returns dict, each result has id, title, score, source, handles multiple queries, respects k values from config)

    \item \textbf{SPECTER-2:}

    \textbf{Get queries} (returns queries when provided as a list, filters out empty strings from queries list, returns empty list when no queries provided), \textbf{No resources} (returns error message when no queries provided, resources missing), \textbf{SPECTER-2 Retriever} (single query returns a list of results, each result contains id, title, score, source, batch query returns a list of result lists), \textbf{Mocked resources} (returns dict, each result has id, title, score, source, handles multiple queries, respects k values from config)

    \item \textbf{Aggregator:}

    \textbf{Normalize scores} (returns empty list for empty input, normalizes scores, preserves original scores separately, handles single result, handles identical scores, applies correct normalization formula, preserves all other fields), \textbf{Reciprocal rank fusion} (works with results from a single retriever, papers appearing in multiple retrievers receive higher scores, includes RRF score, includes retriever count, includes sources list, uses custom k value in RRF formula, preserves paper metadata), \textbf{Aggregator with no results} (returns empty candidate\_papers when all retrievers return empty, treats missing result keys as empty lists, treats None values as empty lists, returns appropriate AIMessage for no results), \textbf{Aggregator RRF method} (RRF is default aggregation method, removes duplicate papers, papers in multiple retrievers rank higher, stores raw results from each retriever, adds rank information to each result, uses custom rrf\_k value from config, returns AIMessage with aggregation summary), \textbf{Aggregator simple method} (uses simple method when specified, keeps highest normalized score per paper, sorts by score descending), \textbf{Aggregator integration} (correctly aggregates results from all three retrievers, handles larger result sets efficiently)

    \item \textbf{LLM Reranker:}

    \textbf{Empty candidates} (returns empty list when candidate\_papers is empty, returns empty when candidate\_papers doesn't exist) {Successful parsing} (correctly parses JSON array and rank papers, extracts JSON array even hen embedded in other text, papers not ranked by LLM should score 0, skips duplicate paper indices, skips out-of-bounds indices), \textbf{Reranker error handling} (falls back to original order on JSON error, falls back when JSON is valid but missing required keys), \textbf{Model selection} (uses OpenAI when closed\_source is True, uses Ollama when closed\_source is false) \textbf{LLMRerankerPrompt} (builds prompt with query and candidate papers, handles papers without titles, handles non-float scores)

    \item \textbf{Reranker:}

    \textbf{Get query} (returns query directly, strips whitespace, falls back to last human message, returns None when neither human message nor query exist, returns None when query is empty string, returns None when query is only whitespace) \textbf{Pairs} (creates query/title pairs, handles missing title, handles None title, handles empty papers), \textbf{Missing query} (handles missing query), \textbf{Empty candidates} (returns empty ranked\_papers for no candidates, returns empty ranked\_papers when candidates is None) \textbf{Mocked model} (ranks papers by scores in descending order, adds rerank score to each paper, uses reranker model from resources, returns AIMessage with ranked papers context, preserves original paper fields, correctly assigns scores)

    \item \textbf{DSPy:}

    \textbf{DSPy signatures} (valid dspy.Signature, valid input and output fields), {DSPy picker} (returns query when directly in state, strips whitespace, uses last HumanMessage if no query, returns None if no query or whitespace only, prefer state['query] over messages, return empty dictionary when not enabled or no config, test error handling, test for candidate building, filter papers without titles, test successful execution), \textbf{DSPy modules} (test get\_module factory function, test SimpleCitationRetriever candidate formatting, test QueryThenRetriever candidate formatting, test RerankAndSelect candidate formatting, test VerifyAndSelect configuration, verify modules inherited from dspy.Module, verify modules have forward method), \textbf{DSPy metrics} (test exact\_match\_metric function, test fuzzy\_match\_metric function, test contains\_match\_metric function, test edge case for all metrics, test using realistic paper titles), \textbf{DSPy data prep} (proper return types, handles proper data preparation, builds negative examples)

\end{itemize}

\section{Results}

\begin{figure\*}[t]
\centering

\begin{subfigure}{0.32\textwidth}
\includegraphics[width=\linewidth]{results1.png}
\caption{Performance Heatmap}
\end{subfigure}
\hfill
\begin{subfigure}{0.32\textwidth}
\includegraphics[width=\linewidth]{results2.png}
\caption{Retriever comparison}
\end{subfigure}
\hfill
\begin{subfigure}{0.32\textwidth}
\includegraphics[width=\linewidth]{results3.png}
\caption{Reranking Recall}
\end{subfigure}

\caption{Retrieval and reranking performance across system components.}
\label{fig:results}
\end{figure\*}
\section{Conclusion}

\subsection{Future Work}

While our system demonstrates strong performance using a combination of sparse and dense retrieval, aggregation, and reranking methods, several extensions of our project remain. First, incorporating stronger embedding models from the MTEB leaderboard could further improve dense retrieval quality, particularly for domain-specific or long-context scientific queries. With increased computational resources, we could also expand the training corpus used for reranking and citation selection, enabling better generalization and harder negative sampling. Beyond retrieval, expanding our LLM-as-a-Judge framework to include multiple evaluation dimensions, such as the relevance and novelty of the citation, would yield a more comprehensive and reliable assessment pipeline. Finally, an integration with generative writing assistants represents an actionable next step, which would allow for citation retrieval, verification, and selection to operate seamlessly within one continuous workflow.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
ERLEAD
