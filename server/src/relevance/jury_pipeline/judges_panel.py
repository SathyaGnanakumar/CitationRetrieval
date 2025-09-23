from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, TypedDict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random
import json
from langchain_openai import ChatOpenAI
from ar.debate_models import DebateResults, QuestionDebate, AgentResponse


# ---------- Schemas ----------
class SideScore(BaseModel):
    score: float = Field(..., ge=0, le=10)
    reasons: str
    evidence_quotes: List[str] = []


class JudgeVerdict(BaseModel):
    judge_id: str
    question_id: str
    pro_score: SideScore
    con_score: SideScore
    preference: Literal["Pro", "Con", "tie"]
    confidence: float = Field(..., ge=0, le=1)
    flags_pro: List[str] = []
    flags_con: List[str] = []


class ConsensusFinding(BaseModel):
    question_id: str
    question_text: str
    winner: Literal["Pro", "Con", "tie"]
    avg_score_pro: float
    avg_score_con: float
    kappa: float
    reasoning_summary: str
    relevance_label: Literal["relevant", "borderline", "not_relevant"]
    rationale_points: List[str]


class PaperEvaluation(BaseModel):
    """Final evaluation of the entire paper"""

    paper_title: str
    overall_relevance: Literal["relevant", "borderline", "not_relevant"]
    pro_wins: int
    con_wins: int
    ties: int
    avg_pro_score: float
    avg_con_score: float
    question_results: List[ConsensusFinding]
    summary: str


# ---------- Panel config ----------
@dataclass
class JudgeConfig:
    id: str
    role: str  # "Methods", "Regulatory", "Techno-econ", etc.
    model_name: str  # e.g., "gpt-4o", "gpt-4o-mini", "mistral-large"


# Example ensemble of 5 judges (mix models/roles as you like)
JUDGES = [
    JudgeConfig("J1", "Methods", "gpt-4.1-2025-04-14"),
    JudgeConfig("J2", "Regulatory", "gpt-5-2025-08-07"),
    JudgeConfig("J3", "TechnoEcon", "gpt-4o"),
    JudgeConfig("J4", "Applicability", "o3-2025-04-16"),
    JudgeConfig("J5", "Skeptic", "o4-mini-2025-04-16"),
]


# ---------- State for LangGraph ----------
class DebateState(TypedDict):
    question_id: str
    question_text: str
    answer_pro: str
    answer_con: str
    context_chunks: List[str]
    judge_outputs: List[dict]
    consensus: dict


# ---------- Helpers ----------
def judge_prompt(
    role: str, question: str, answer_pro: str, answer_con: str, context: str
):
    return f"""
You are a {role} judge evaluating TWO responses (Pro and Con) to a question about methane-mitigation research relevance.
Use only the provided evidence. Penalize speculation. 

Return your evaluation in this JSON format:
{{
  "judge_id": "will be filled automatically",
  "question_id": "will be filled automatically", 
  "pro_score": {{
    "score": <0-10 float>,
    "reasons": "<detailed reasoning for pro score>",
    "evidence_quotes": ["<relevant quote 1>", "<relevant quote 2>"]
  }},
  "con_score": {{
    "score": <0-10 float>,
    "reasons": "<detailed reasoning for con score>",
    "evidence_quotes": ["<relevant quote 1>", "<relevant quote 2>"]
  }},
  "preference": "<Pro|Con|tie>",
  "confidence": <0-1 float>,
  "flags_pro": ["<any issues with pro argument>"],
  "flags_con": ["<any issues with con argument>"]
}}

Scoring Rubric (10 points per side):
- Groundedness in evidence (0-3 points): How well does the answer cite specific data from the context?
- Specificity (0-2 points): Are claims precise with numbers/details vs. vague generalizations?
- Reasoning quality (0-3 points): Is the logic sound and arguments well-structured?
- CGS/G-MAST relevance (0-2 points): Does it address methane abatement technology relevance effectively?

[QUESTION]
{question}

[PRO RESEARCH PAPER RESPONSE]
{answer_pro}

[CON RESEARCH PAPER RESPONSE] 
{answer_con}

[EVIDENCE/CONTEXT ALLOWED]
{context}
""".strip()


def run_one_judge(state: DebateState, cfg: JudgeConfig):
    """Run a single judge evaluation"""
    llm = ChatOpenAI(model=cfg.model_name)
    structured = llm.with_structured_output(JudgeVerdict)

    prompt = judge_prompt(
        cfg.role,
        state["question_text"],
        state["answer_pro"],
        state["answer_con"],
        "\n".join(state["context_chunks"]),
    )

    verdict = structured.invoke([{"role": "user", "content": prompt}])

    # Convert to dict and add metadata
    verdict_dict = verdict.model_dump()
    verdict_dict["judge_id"] = cfg.id
    verdict_dict["question_id"] = state["question_id"]

    return verdict_dict


def run_panel_parallel(state: DebateState, judge_cfgs: List[JudgeConfig]):
    """Run all judges in parallel"""
    with ThreadPoolExecutor(max_workers=len(judge_cfgs)) as executor:
        futures = [executor.submit(run_one_judge, state, cfg) for cfg in judge_cfgs]
        return [future.result() for future in futures]


def aggregate_verdicts(judge_outputs: List[dict], question_text: str):
    """Aggregate judge verdicts into consensus finding"""
    avg_pro = sum(o["pro_score"]["score"] for o in judge_outputs) / len(judge_outputs)
    avg_con = sum(o["con_score"]["score"] for o in judge_outputs) / len(judge_outputs)

    votes_pro = sum(o["preference"] == "Pro" for o in judge_outputs)
    votes_con = sum(o["preference"] == "Con" for o in judge_outputs)
    votes_tie = sum(o["preference"] == "tie" for o in judge_outputs)

    # Determine winner
    if votes_pro > votes_con and votes_pro > votes_tie:
        winner = "Pro"
    elif votes_con > votes_pro and votes_con > votes_tie:
        winner = "Con"
    else:
        winner = "tie"

    # Calculate agreement (kappa proxy)
    max_votes = max(votes_pro, votes_con, votes_tie)
    kappa = round(max_votes / len(judge_outputs), 2)

    # Determine relevance
    max_score = max(avg_pro, avg_con)
    if max_score >= 7 and winner != "tie":
        relevance = "relevant"
    elif max_score >= 5:
        relevance = "borderline"
    else:
        relevance = "not_relevant"

    # Collect reasoning from top judges
    reasoning_parts = []
    for output in judge_outputs[:3]:  # Take first 3 judges' reasoning
        if winner == "Pro":
            reasoning_parts.append(output["pro_score"]["reasons"])
        elif winner == "Con":
            reasoning_parts.append(output["con_score"]["reasons"])
        else:  # tie
            reasoning_parts.append(output["pro_score"]["reasons"])

    return ConsensusFinding(
        question_id=judge_outputs[0]["question_id"],
        question_text=question_text,
        winner=winner,
        avg_score_pro=round(avg_pro, 2),
        avg_score_con=round(avg_con, 2),
        kappa=kappa,
        reasoning_summary=" | ".join(reasoning_parts)[:800],  # Limit length
        relevance_label=relevance,
        rationale_points=[
            f"Average Pro score: {avg_pro:.1f}, Con score: {avg_con:.1f}",
            f"Judge preferences: Pro={votes_pro}, Con={votes_con}, Tie={votes_tie}",
            f"Agreement level (kappa): {kappa}",
            f"Relevance assessment: {relevance}",
        ],
    )


# ---------- LangGraph nodes ----------
def panel_node(state: DebateState):
    """Execute judge panel evaluation"""
    judge_outputs = run_panel_parallel(state, JUDGES)
    return {"judge_outputs": judge_outputs}


def aggregate_node(state: DebateState):
    """Aggregate judge verdicts into consensus"""
    consensus = aggregate_verdicts(state["judge_outputs"], state["question_text"])
    return {"consensus": consensus.model_dump()}


def build_panel_graph():
    """Build the LangGraph workflow"""
    graph = StateGraph(DebateState)
    graph.add_node("panel", panel_node)
    graph.add_node("aggregate", aggregate_node)
    graph.set_entry_point("panel")
    graph.add_edge("panel", "aggregate")
    graph.set_finish_point("aggregate")
    return graph.compile()


# ---------- Integration Functions ----------
def debate_to_judge_state(
    question_debate: QuestionDebate, context_chunks: List[str]
) -> DebateState:
    """Convert QuestionDebate to DebateState for judges"""

    # Extract Pro and Con responses
    pro_response = ""
    con_response = ""

    for response in question_debate.agent_responses:
        if "Pro" in response.agent_name:
            pro_response = response.response
        elif "Con" in response.agent_name:
            con_response = response.response

    return DebateState(
        question_id=f"Q{question_debate.question_number}",
        question_text=question_debate.question_text,
        answer_pro=pro_response,
        answer_con=con_response,
        context_chunks=context_chunks,
        judge_outputs=[],
        consensus={},
    )


def evaluate_paper(
    debate_results: DebateResults, context_chunks: List[str]
) -> PaperEvaluation:
    """Evaluate entire paper using judge panel"""

    judge_graph = build_panel_graph()
    question_results = []

    print("🏛️ Starting Judge Panel Evaluation...")
    print(f"📋 Evaluating {len(debate_results.session.question_debates)} questions")

    for i, question_debate in enumerate(debate_results.session.question_debates, 1):
        print(f"⚖️  Judging Question {i}/{len(debate_results.session.question_debates)}")

        # Convert to judge state
        judge_state = debate_to_judge_state(question_debate, context_chunks)

        # Run judge panel
        result = judge_graph.invoke(judge_state)
        consensus = ConsensusFinding(**result["consensus"])
        question_results.append(consensus)

        print(f"   Winner: {consensus.winner} | Relevance: {consensus.relevance_label}")

    # Calculate overall statistics
    pro_wins = sum(1 for r in question_results if r.winner == "Pro")
    con_wins = sum(1 for r in question_results if r.winner == "Con")
    ties = sum(1 for r in question_results if r.winner == "tie")

    avg_pro_score = sum(r.avg_score_pro for r in question_results) / len(
        question_results
    )
    avg_con_score = sum(r.avg_score_con for r in question_results) / len(
        question_results
    )

    # Determine overall relevance
    relevant_count = sum(1 for r in question_results if r.relevance_label == "relevant")
    borderline_count = sum(
        1 for r in question_results if r.relevance_label == "borderline"
    )

    if relevant_count >= len(question_results) * 0.6:
        overall_relevance = "relevant"
    elif (relevant_count + borderline_count) >= len(question_results) * 0.5:
        overall_relevance = "borderline"
    else:
        overall_relevance = "not_relevant"

    summary_parts = [
        f"Paper evaluation completed with {len(question_results)} questions judged.",
        f"Pro arguments won {pro_wins} questions, Con won {con_wins}, with {ties} ties.",
        f"Average scores: Pro {avg_pro_score:.1f}/10, Con {avg_con_score:.1f}/10.",
        f"Overall relevance assessment: {overall_relevance}.",
        f"Questions rated as relevant: {relevant_count}, borderline: {borderline_count}.",
    ]

    return PaperEvaluation(
        paper_title=debate_results.session.paper_metadata.title,
        overall_relevance=overall_relevance,
        pro_wins=pro_wins,
        con_wins=con_wins,
        ties=ties,
        avg_pro_score=round(avg_pro_score, 2),
        avg_con_score=round(avg_con_score, 2),
        question_results=question_results,
        summary=" ".join(summary_parts),
    )
