"""
FastAPI server for the Citation Retrieval system.

This API exposes the citation retrieval workflow as a REST endpoint.
"""

import logging
import os
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.workflow import RetrievalWorkflow
from src.models.state import RetrievalState

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Citation Retrieval API",
    description="Multi-baseline citation retrieval system combining BM25, E5, and SPECTER retrievers",
    version="0.1.0"
)

# Configure CORS to allow the Next.js client
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://client:3000",  # Docker internal network
        os.getenv("CLIENT_URL", "http://localhost:3000")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for resources (loaded on startup)
workflow: Optional[RetrievalWorkflow] = None
resources: Optional[Dict] = None


# Request/Response Models
class CitationRequest(BaseModel):
    """Request model for citation retrieval."""
    context: str = Field(
        ...,
        description="The citation context with [CITATION] marker",
        min_length=10
    )
    k: int = Field(
        default=10,
        description="Number of top results to return",
        ge=1,
        le=50
    )
    use_llm_reranker: bool = Field(
        default=True,
        description="Whether to use LLM-based reranking"
    )


class Citation(BaseModel):
    """Citation information."""
    title: str
    authors: List[str]
    year: Optional[int] = None
    source: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None


class CitationResult(BaseModel):
    """Response model for citation retrieval."""
    citation: Citation
    confidence: float = Field(
        description="Confidence score (0-100)",
        ge=0,
        le=100
    )
    reasoning: Optional[str] = None
    score: float = Field(
        description="Retrieval score"
    )
    formatted: Dict[str, str] = Field(
        description="Formatted citations in different styles"
    )


class CitationResponse(BaseModel):
    """Response containing the top citation results."""
    results: List[CitationResult]
    query: str
    expanded_queries: List[str]


def format_apa(paper: Dict) -> str:
    """Format a paper in APA style."""
    authors = paper.get("authors", [])
    author_str = authors[0] if authors else "Unknown"
    if len(authors) > 2:
        author_str += " et al."
    elif len(authors) == 2:
        author_str += f" & {authors[1]}"

    year = paper.get("year", "n.d.")
    title = paper.get("title", "Unknown Title")
    source = paper.get("source", "Unknown Source")
    doi = paper.get("doi", "")

    citation = f"{author_str} ({year}). {title}. {source}."
    if doi:
        citation += f" https://doi.org/{doi}"

    return citation


def format_mla(paper: Dict) -> str:
    """Format a paper in MLA style."""
    authors = paper.get("authors", [])
    if not authors:
        author_str = "Unknown"
    else:
        # MLA format: Last, First
        author_parts = authors[0].split(",")
        if len(author_parts) == 2:
            author_str = f"{author_parts[0].strip()}, {author_parts[1].strip()}"
        else:
            author_str = authors[0]

        if len(authors) > 2:
            author_str += ", et al."
        elif len(authors) == 2:
            author_str += f", and {authors[1]}"

    title = paper.get("title", "Unknown Title")
    source = paper.get("source", "Unknown Source")
    year = paper.get("year", "n.d.")

    return f'{author_str}. "{title}." {source}, {year}.'


def load_resources():
    """Load corpus and build retrieval resources."""
    global resources

    logger.info("Loading corpus and building resources...")

    # Import here to avoid circular dependencies
    from datasets.scholarcopilot import load_dataset, build_citation_corpus

    # Get dataset path from environment or use default
    dataset_path = os.getenv(
        "DATASET_DIR",
        "corpus/scholarcopilot/scholar_copilot_eval_data_1k.json"
    )

    if not Path(dataset_path).exists():
        # Try alternative paths
        alt_paths = [
            "/app/data/scholarcopilot/scholar_copilot_eval_data_1k.json",
            "server/corpus/scholarcopilot/scholar_copilot_eval_data_1k.json"
        ]
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                dataset_path = alt_path
                break

    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please set DATASET_DIR environment variable or place dataset in corpus/scholarcopilot/"
        )

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)

    logger.info(f"Building corpus from {len(dataset)} examples...")
    corpus = build_citation_corpus(dataset)

    logger.info(f"Building retrieval resources for {len(corpus)} papers...")

    # Build resources based on available method
    try:
        from src.resources.builders import build_inmemory_resources
        resources = build_inmemory_resources(corpus)
    except ImportError:
        # Fallback if builders moved
        logger.warning("Could not import from src.resources.builders, trying alternative import")
        from datasets.scholarcopilot.loader import build_resources_from_corpus
        resources = build_resources_from_corpus(corpus)

    logger.info("Resources loaded successfully!")
    return resources


@app.on_event("startup")
async def startup_event():
    """Initialize the workflow and load resources on startup."""
    global workflow, resources

    try:
        logger.info("Initializing Citation Retrieval API...")

        # Initialize workflow
        use_llm_reranker = os.getenv("USE_LLM_RERANKER", "true").lower() == "true"
        workflow = RetrievalWorkflow(use_llm_reranker=use_llm_reranker)
        logger.info(f"Workflow initialized (LLM reranker: {use_llm_reranker})")

        # Load resources
        resources = load_resources()

        logger.info("API ready to accept requests!")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}", exc_info=True)
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Citation Retrieval API",
        "version": "0.1.0",
        "status": "running" if workflow and resources else "initializing"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not workflow or not resources:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "status": "healthy",
        "corpus_size": len(resources.get("ids", [])),
        "retrievers": ["bm25", "e5", "specter"]
    }


@app.post("/api/find-citation", response_model=CitationResponse)
async def find_citation(request: CitationRequest):
    """
    Find the most relevant citation for a given context.

    This endpoint takes a citation context (with [CITATION] marker) and returns
    the top matching papers using multi-baseline retrieval.
    """
    if not workflow or not resources:
        raise HTTPException(
            status_code=503,
            detail="Service initializing, please try again in a few moments"
        )

    try:
        logger.info(f"Processing citation request (k={request.k})")

        # Clean the context (remove [CITATION] marker for query)
        context = request.context.replace("[CITATION]", "").strip()

        # Prepare initial state
        initial_state: RetrievalState = {
            "messages": [HumanMessage(content=context)],
            "query": context,
            "queries": [context],
            "resources": resources,
            "config": {
                "k": request.k,
                "use_llm_reranker": request.use_llm_reranker
            },
            "bm25_results": [],
            "e5_results": [],
            "specter_results": [],
            "retriever_results": {},
            "candidate_papers": [],
            "ranked_papers": []
        }

        # Run workflow
        logger.info("Running retrieval workflow...")
        final_state = workflow.run(initial_state)

        # Extract results
        ranked_papers = final_state.get("ranked_papers", [])
        expanded_queries = final_state.get("queries", [context])

        if not ranked_papers:
            raise HTTPException(
                status_code=404,
                detail="No matching citations found"
            )

        logger.info(f"Found {len(ranked_papers)} results")

        # Format results
        results = []
        for paper in ranked_papers[:request.k]:
            # Extract citation info
            citation = Citation(
                title=paper.get("title", "Unknown Title"),
                authors=paper.get("authors", []),
                year=paper.get("year"),
                source=paper.get("source"),
                doi=paper.get("doi"),
                abstract=paper.get("abstract")
            )

            # Calculate confidence (normalize score to 0-100)
            score = paper.get("rerank_score", paper.get("score", 0))
            # Confidence based on score and rank
            max_score = ranked_papers[0].get("rerank_score", ranked_papers[0].get("score", 1))
            confidence = min(100, max(0, (score / max_score) * 100)) if max_score > 0 else 50

            # Create result
            result = CitationResult(
                citation=citation,
                confidence=round(confidence, 1),
                reasoning=paper.get("reasoning", ""),
                score=score,
                formatted={
                    "apa": format_apa(paper),
                    "mla": format_mla(paper)
                }
            )
            results.append(result)

        return CitationResponse(
            results=results,
            query=context,
            expanded_queries=expanded_queries
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
