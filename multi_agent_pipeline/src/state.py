from typing import List, Optional, Annotated
from pydantic import BaseModel, Field

class PipelineState(BaseModel):
    query: Annotated[str, "input_query"]
    expanded_queries: Optional[List[str]] = []
    candidate_papers: Optional[List[str]] = []
    ranked_papers: Optional[List[str]] = []
    verified_paper: Optional[str] = None
