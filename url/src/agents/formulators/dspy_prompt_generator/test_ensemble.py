import dspy
import os
from dotenv import load_dotenv
from dspy_prompt_generator.modules import get_module

# Load environment variables
load_dotenv()

# Configure DSPy with OpenAI (assuming API key is in .env)
# If not, we'll use a dummy for structure testing, but it won't produce real reasoning.
try:
    lm = dspy.OpenAI(model='gpt-4o', max_tokens=1000)
    dspy.settings.configure(lm=lm)
    print("Configured DSPy with OpenAI GPT-4o")
except Exception as e:
    print(f"Warning: Could not configure OpenAI: {e}")
    class MockLM(dspy.LM):
        def __init__(self, responses):
            super().__init__(model="mock")
            self.responses = responses
            self.history = []

        def basic_request(self, prompt, **kwargs):
            self.history.append(prompt)
            return [self.responses[len(self.history) % len(self.responses)]]

        def __call__(self, *args, **kwargs):
            # Handle prompt being passed as positional or keyword arg
            prompt = kwargs.get("prompt")
            if prompt is None and args:
                prompt = args[0]
            
            return self.basic_request(prompt, **kwargs)

    print("Using MockLM for testing structure only.")
    lm = MockLM([
        "Reasoning: The context mentions ScholarCopilot benchmark. Candidate 1 is ScholarCopilot. Match.\nSelected Title: ScholarCopilot: Training large language models for academic writing with accurate citations",
        "Reasoning: Search query should be 'ScholarCopilot benchmark'.\nSearch Query: ScholarCopilot benchmark",
        "Reasoning: Candidate 1 is the exact match.\nRanked Titles: ScholarCopilot: Training large language models for academic writing with accurate citations",
        "Reasoning: Matches perfectly.\nIs Match: True\nConfidence: High"
    ])
    dspy.settings.configure(lm=lm)

def test_ensemble():
    # 1. Extract context from the paper (simulated)
    # From paper: "We evaluate our system against information retrieval baselines and single-agent models using the ScholarCopilot benchmark."
    citation_context = "We evaluate our system against information retrieval baselines and single-agent models using the ScholarCopilot benchmark [CITATION]."
    
    # 2. Define candidates (one correct, some distractors)
    candidates = [
        {
            "title": "ScholarCopilot: Training large language models for academic writing with accurate citations",
            "abstract": "We introduce ScholarCopilot, a dataset and framework for citation retrieval..."
        },
        {
            "title": "Litsearch: A retrieval benchmark for scientific literature search",
            "abstract": "We present Litsearch, a benchmark for retrieving scientific literature..."
        },
        {
            "title": "CiteME: Can Language Models Accurately Cite Scientific Claims?",
            "abstract": "We evaluate LLMs on citation tasks using the CiteME benchmark..."
        }
    ]
    
    print(f"\nTesting EnsembleRetriever with context:\n'{citation_context}'\n")
    
    # 3. Initialize EnsembleRetriever
    ensemble = get_module("ensemble")
    
    # 4. Run prediction
    prediction = ensemble(citation_context=citation_context, candidates=candidates)
    
    # 5. Output results
    print("\n--- Ensemble Prediction ---")
    print(f"Selected Title: {prediction.selected_title}")
    print(f"Votes: {prediction.votes}")
    print(f"Reasoning: {prediction.reasoning}")
    
    print("\n--- Individual Module Choices ---")
    print(f"Simple: {prediction.simple_choice}")
    print(f"Query: {prediction.query_choice}")
    print(f"Rerank: {prediction.rerank_choice}")

if __name__ == "__main__":
    test_ensemble()
