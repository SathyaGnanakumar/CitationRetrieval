from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class AgentResponse(BaseModel):
    """A single agent's response to a question"""
    agent_name: str = Field(..., description="Name of the agent (e.g., 'Pro Research Paper', 'Con Research Paper')")
    response: str = Field(..., description="The agent's full response text")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="When the response was generated")

class QuestionDebate(BaseModel):
    """Debate results for a single question"""
    question_number: int = Field(..., description="Sequential number of the question (1, 2, 3, ...)")
    question_text: str = Field(..., description="The full text of the question being debated")
    moderator_prompt: str = Field(..., description="The moderator's prompt that initiated this question's debate")
    agent_responses: List[AgentResponse] = Field(..., description="List of all agent responses to this question")
    
class PaperMetadata(BaseModel):
    """Metadata about the paper being evaluated"""
    title: str = Field(..., description="Title of the research paper")
    abstract: Optional[str] = Field(None, description="Abstract of the paper")
    authors: Optional[List[str]] = Field(None, description="List of paper authors")
    doi: Optional[str] = Field(None, description="DOI of the paper")
    publication_date: Optional[str] = Field(None, description="Publication date")
    
    @field_validator('authors', mode='before')
    @classmethod
    def parse_authors(cls, v):
        if isinstance(v, str):
            # Split string authors by common delimiters
            import re
            authors = re.split(r',\s*|\s+and\s+|\s*;\s*', v)
            return [author.strip() for author in authors if author.strip()]
        return v

class DebateSession(BaseModel):
    """Complete debate session results"""
    paper_metadata: PaperMetadata = Field(..., description="Information about the paper being debated")
    participating_agents: List[str] = Field(..., description="Names of all participating agents")
    total_questions: int = Field(..., description="Total number of questions addressed")
    question_debates: List[QuestionDebate] = Field(..., description="Debate results for each question")
    session_timestamp: datetime = Field(default_factory=datetime.now, description="When the debate session was completed")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DebateResults(BaseModel):
    """Top-level container for all debate results"""
    session: DebateSession = Field(..., description="The complete debate session")
    
    def to_json(self, **kwargs) -> str:
        """Export debate results as JSON"""
        return self.model_dump_json(indent=2, **kwargs)
    
    def get_question_responses(self, question_number: int) -> Optional[QuestionDebate]:
        """Get debate results for a specific question number"""
        for debate in self.session.question_debates:
            if debate.question_number == question_number:
                return debate
        return None
    
    def get_agent_responses(self, agent_name: str) -> List[AgentResponse]:
        """Get all responses from a specific agent across all questions"""
        responses = []
        for debate in self.session.question_debates:
            for response in debate.agent_responses:
                if response.agent_name == agent_name:
                    responses.append(response)
        return responses