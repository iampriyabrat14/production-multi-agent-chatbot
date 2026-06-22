from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tools.pinecone_retriever import PineconeRetriever

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)          # answer generation
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # grading + reflection


# ── Subgraph State ─────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """
    State that flows through every node in the RAG subgraph.
    Each node reads from and writes to this state.
    LangGraph checkpoints this state after every node.
    """
    question: str                              # original user question
    chunks: list[dict]                         # retrieved Pinecone chunks
    relevant_chunks: list[dict]                # chunks that passed grading
    answer: str                                # generated answer
    reflection_score: float                    # quality score 0.0-1.0
    retry_count: int                           # how many retries attempted
    sources: list[str]                         # source documents used


# ── Node 1: Retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: RAGState) -> RAGState:
    """
    ReAct — ACT step.
    Searches Pinecone for chunks relevant to the question.
    """
    retriever = PineconeRetriever()
    chunks = retriever.retrieve(state["question"], top_k=5)
    return {**state, "chunks": chunks}


# ── Node 2: Grade Chunks ───────────────────────────────────────────────────────

def grade_chunks_node(state: RAGState) -> RAGState:
    """
    ReAct — OBSERVE step.
    Checks if retrieved chunks are actually relevant to the question.
    Filters out low-score or off-topic chunks before sending to LLM.
    Only chunks with score > 0.75 pass through.
    """
    relevant = [c for c in state["chunks"] if c["score"] > 0.75]

    # if nothing passed grading, keep top 2 anyway
    if not relevant:
        relevant = state["chunks"][:2]

    sources = list({c["source"] for c in relevant})
    return {**state, "relevant_chunks": relevant, "sources": sources}


# ── Node 3: Generate ───────────────────────────────────────────────────────────

def generate_node(state: RAGState) -> RAGState:
    """
    ReAct — ACT step.
    LLM generates an answer grounded in retrieved chunks.
    System prompt enforces staying within retrieved context only.
    """
    context = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(state["relevant_chunks"])
    )

    system_prompt = """You are a helpful AI assistant that summarizes and explains company documents.

Using the provided context, give a clear and complete answer to the user's question.
- Summarize the key points in a friendly, readable format
- Use bullet points or sections when the content has multiple topics
- Always cite which document number [1], [2] etc. you used
- If the context covers the topic partially, summarize what is available and note what may not be covered
- Only say you cannot answer if the context is completely unrelated to the question"""

    messages = [
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}")
    ]

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        *messages,
    ])

    return {**state, "answer": response.content, "retry_count": state.get("retry_count", 0) + 1}


# ── Node 4: Reflect ────────────────────────────────────────────────────────────

def reflect_node(state: RAGState) -> RAGState:
    """
    Reflection step.
    LLM critiques its own answer and assigns a quality score 0.0-1.0.
    If score < 0.7 and retry_count < 2, graph loops back to generate_node.
    """
    critique_prompt = f"""Rate the quality of this answer on a scale of 0.0 to 1.0.

Question: {state['question']}
Answer: {state['answer']}

Criteria:
- Is the answer grounded in the context? (not hallucinated)
- Does it fully address the question?
- Is it concise and clear?

Return ONLY a number between 0.0 and 1.0. Nothing else."""

    response = llm_fast.invoke([{"role": "user", "content": critique_prompt}])

    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))   # clamp between 0 and 1
    except ValueError:
        score = 0.8   # default if parsing fails

    return {**state, "reflection_score": score}


# ── Conditional Edge: Retry or End ────────────────────────────────────────────

def should_retry(state: RAGState) -> str:
    """
    Skip reflection on first attempt (retry_count==1) to save latency.
    Only reflect if we already retried once.
    """
    if state.get("retry_count", 0) < 1:
        return "end"
    if state["reflection_score"] < 0.7 and state.get("retry_count", 0) < 2:
        return "retry"
    return "end"


# ── Build RAG Subgraph ─────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Assembles all RAG nodes into a LangGraph subgraph.

    Graph flow:
      retrieve → grade_chunks → generate → reflect
                                    ↑           │
                                    └── retry ──┘ (max 2 retries)
                                                │
                                               end
    """
    graph = StateGraph(RAGState)

    # add all nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_chunks", grade_chunks_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)

    # define edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_edge("grade_chunks", "generate")
    graph.add_edge("generate", "reflect")

    # conditional edge — retry or end based on reflection score
    graph.add_conditional_edges(
        "reflect",
        should_retry,
        {
            "retry": "generate",   # loop back to generate with same chunks
            "end": END,
        },
    )

    return graph.compile()


# ── Public Entry Point ─────────────────────────────────────────────────────────

_rag_graph = None

def _get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


def run_rag_agent(question: str) -> dict:
    """
    Run the RAG subgraph for a given question.
    Returns answer + sources + reflection score.

    Called by the Supervisor Agent in Step 16.
    """
    rag_graph = _get_rag_graph()

    initial_state: RAGState = {
        "question": question,
        "chunks": [],
        "relevant_chunks": [],
        "answer": "",
        "reflection_score": 0.0,
        "retry_count": 0,
        "sources": [],
    }

    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "reflection_score": final_state["reflection_score"],
        "agent": "rag_agent",
    }
