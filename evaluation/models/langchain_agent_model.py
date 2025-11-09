# In evaluation/models/langchain_agent_model.py
from typing import List, Dict, Any
from .base_model import BaseRetrievalModel
from .dense_model import DenseRetrievalModel # Reuse your existing model
from langchain_community.document_transformers import CrossEncoderReranker
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class LangChainAgentModel(BaseRetrievalModel):
    
    def __init__(self, 
                 retriever_model: str = "allenai/specter2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 llm_model: str = "claude-3-sonnet-20240229",
                 stage1_k: int = 100,
                 stage2_k: int = 20):
        
        super().__init__("LangChainAgent")
        
        # --- AGENT 1: RETRIEVER ---
        # We reuse your existing DenseRetrievalModel
        self.retriever = DenseRetrievalModel(model_name=retriever_model)
        self.stage1_k = stage1_k

        # --- AGENT 2: RERANKER ---
        self.reranker = CrossEncoderReranker(model=reranker_model, top_n=stage2_k)

        # --- AGENT 3: LLM SELECTION ---
        # Define the JSON output structure
        class Selection(BaseModel):
            paper_num: int = Field(description="The paper number (1-N)")
            reasoning: str = Field(description="Brief reasoning for the selection")
            score: float = Field(description="Relevance score (0-10)")

        class LLMResponse(BaseModel):
            selections: List[Selection] = Field(description="List of selected papers")

        # Create the LLM and bind the tool
        self.llm = ChatAnthropic(model=llm_model, temperature=0)
        self.llm_with_tools = self.llm.bind_tools([LLMResponse])
        self.llm_parser = PydanticToolsParser(tools=[LLMResponse])

        # --- Create the full chain ---
        self.chain = (
            RunnablePassthrough.assign(candidates=RunnableLambda(self._agent_1_retrieve))
            | RunnablePassthrough.assign(candidates=RunnableLambda(self._agent_2_rerank))
            | RunnablePassthrough.assign(final_papers=RunnableLambda(self._agent_3_select))
        )

    # ... (Helper methods will go here) ...
    def _agent_1_retrieve(self, state: dict) -> List[Dict]:
        """Agent 1: Fetches top-k candidates from the corpus."""
        print(f"--- Agent 1: Retrieving {self.stage1_k} candidates... ---")
        return self.retriever.retrieve(
            query=state["query"],
            corpus=state["corpus"],
            k=self.stage1_k
        )
    
    def _agent_2_rerank(self, state: dict) -> List[Dict]:
        """Agent 2: Reranks candidates using a cross-encoder."""
        print(f"--- Agent 2: Reranking {len(state['candidates'])} candidates... ---")
        # Convert dicts to LangChain Document objects
        documents = [
            Document(
                page_content=(c.get('title', '') + " " + c.get('abstract', '')).strip(),
                metadata=c
            ) for c in state["candidates"]
        ]
        
        # Run the reranker
        reranked_docs = self.reranker.compress_documents(
            documents=documents,
            query=state["query"]
        )
        
        # Convert back to dicts
        return [doc.metadata for doc in reranked_docs]
    
    def _agent_3_select(self, state: dict) -> List[Dict]:
        """Agent 3: Uses Claude to select the final papers."""
        print(f"--- Agent 3: Selecting top {state['final_k']} papers... ---")
        
        # 1. Format candidates for the prompt
        candidate_strs = []
        for i, cand in enumerate(state["candidates"], 1):
            title = cand.get('title', 'No Title')
            abstract = (cand.get('abstract', '') or "")[:200]
            candidate_strs.append(f"{i}. {title}\n   Abstract: {abstract}...")
        
        prompt = f"""You are a citation expert. Given the context, select the top {state['final_k']} most relevant papers from the list.

CITATION CONTEXT:
{state["query"]}

CANDIDATE PAPERS:
{'\n'.join(candidate_strs)}

Select the best {state['final_k']} papers and provide reasoning.
"""
        
        # 2. Build the LLM chain
        selection_chain = (
            ChatPromptTemplate.from_template(prompt)
            | self.llm_with_tools
            | self.llm_parser
        )
        
        # 3. Invoke and parse
        try:
            # The parser returns a list of tools, we want the first one
            response = selection_chain.invoke({})[0]
            
            # 4. Map results back to original paper dicts
            final_papers = []
            for sel in response.selections:
                paper_index = sel.paper_num - 1 # 1-indexed to 0-indexed
                if 0 <= paper_index < len(state["candidates"]):
                    paper_data = state["candidates"][paper_index]
                    paper_data['llm_reasoning'] = sel.reasoning
                    paper_data['llm_score'] = sel.score
                    final_papers.append(paper_data)
            
            return final_papers[:state['final_k']]

        except Exception as e:
            print(f"Error in Agent 3 (LLM Selection): {e}")
            # Fallback: return the top-k from the reranker
            return state["candidates"][:state['final_k']]

    def retrieve(self, query: str, corpus: List[Dict], k: int = 10) -> List[Dict]:
        """
        Runs the full LCEL chain.
        """
        # The input "state" for the chain
        chain_input = {
            "query": query,
            "corpus": corpus,
            "final_k": k
        }
        
        # Invoke the chain
        result_state = self.chain.invoke(chain_input)
        
        # Return the final selected papers
        return result_state["final_papers"]
    
    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}