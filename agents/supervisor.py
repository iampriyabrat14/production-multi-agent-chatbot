# Author: Priyabrat Dalbehera | github.com/iampriyabrat14 | Production Multi-Agent Chatbot
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import json

from agents.rag_agent import run_rag_agent
from agents.sql_agent import run_sql_agent
from agents.realtime_agent import run_realtime_agent
from agents.memory_agent import retrieve_memory, save_memory

# gpt-4o-mini for routing — 10x faster, routing doesn't need full GPT-4o
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# checkpointer — saves full graph state after every node
# enables conversation resumption, human-in-the-loop, failure recovery
checkpointer = MemorySaver()


# ── Supervisor State ───────────────────────────────────────────────────────────

class SupervisorState(TypedDict):
    """
    Master state flowing through the entire supervisor graph.
    Every node reads from and writes to this state.
    LangGraph checkpoints this after every node via MemorySaver.
    """
    user_id: str
    session_id: str
    question: str                   # original user question
    memory_context: str             # loaded from Memory Agent before routing
    plan: list[str]                 # list of agents to call e.g. ["rag", "sql"]
    agent_results: dict             # results from each agent
    final_answer: str               # merged + polished final answer
    sources: list[str]              # source documents used
    tokens_used: int                # total tokens across all LLM calls
    requires_human_approval: bool   # human-in-the-loop flag


# ── Node 1: Memory Retrieve ────────────────────────────────────────────────────

def memory_retrieve_node(state: SupervisorState) -> SupervisorState:
    """
    First node — always runs before anything else.
    Loads user facts + conversation history from Memory Agent.
    Injects context into state so all downstream nodes can use it.
    """
    context = retrieve_memory(
        user_id=state["user_id"],
        session_id=state["session_id"],
        question=state["question"],
    )
    return {**state, "memory_context": context}


# ── Node 2: Plan ───────────────────────────────────────────────────────────────

def plan_node(state: SupervisorState) -> SupervisorState:
    """
    Supervisor reasoning — decides which agent(s) to call.
    This is the PLANNING step of agentic AI.
    Can plan multiple agents for complex multi-part questions.

    Returns a plan like:
      ["rag"]                     → document question only
      ["sql"]                     → CSV data question only
      ["realtime"]                → live data question only
      ["memory"]                  → facts about user only
      ["realtime", "sql"]         → needs both live + CSV data
      ["rag", "sql", "realtime"]  → complex multi-source question
    """
    prompt = f"""You are a supervisor AI. Analyze the user question and decide which specialist agents to call.

Available agents:
  - rag      → answers questions from PDF documents, policies, manuals, knowledge base
  - sql      → answers questions from CSV/tabular data: sales, revenue, customers, employees, products
  - realtime → answers questions needing live data: currency rates, exchange rates, stock prices
  - memory   → retrieves facts about the user

User question: {state['question']}

Memory context: {state['memory_context']}

Examples:
  "What is the refund policy?"              → ["rag"]
  "What is Q3 total revenue?"               → ["sql"]
  "Show me top 5 products by sales"         → ["sql"]
  "How many customers are in North region?" → ["sql"]
  "Convert 100 USD to EUR"                  → ["realtime"]
  "What is EUR/USD rate?"                   → ["realtime"]
  "What is our refund policy and Q3 sales?" → ["rag", "sql"]
  "What is the warranty policy?"            → ["rag"]
  "List all employees in Engineering"       → ["sql"]

Rules:
  - Use "sql" for ANY question about sales, revenue, customers, employees, products, CSV data
  - Use "rag" ONLY for policy/document/manual questions
  - Use "realtime" for live currency or exchange rate questions
  - Include ALL agents needed to fully answer multi-part questions

Return a JSON array only. No explanation. Example: ["sql"] or ["rag", "sql"]
"""
    response = llm.invoke([{"role": "user", "content": prompt}])

    # strip markdown code blocks if LLM wraps response in ```json ... ```
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        plan = json.loads(content)
        valid = {"rag", "sql", "realtime", "memory"}
        plan = [p for p in plan if p in valid]
        if not plan:
            plan = ["sql"]  # better default than rag for business questions
    except json.JSONDecodeError:
        # guess from keywords if JSON fails
        q = state["question"].lower()
        if any(w in q for w in ["revenue", "sales", "customer", "employee", "product", "csv", "total", "how many", "list"]):
            plan = ["sql"]
        elif any(w in q for w in ["usd", "eur", "currency", "rate", "convert", "exchange"]):
            plan = ["realtime"]
        else:
            plan = ["rag"]

    return {**state, "plan": plan}


# ── Node 3: Route (Map Phase) ──────────────────────────────────────────────────

def route_node(state: SupervisorState) -> SupervisorState:
    """
    Map-Reduce pattern — MAP phase.
    Executes all planned agents.
    Runs multiple agents in PARALLEL using ThreadPoolExecutor.
    Single agent runs directly without parallelism overhead.
    """
    plan = state["plan"]
    question = state["question"]
    results = {}

    def run_agent(agent_name: str) -> tuple[str, dict]:
        """Run a single agent and return its result."""
        if agent_name == "rag":
            return agent_name, run_rag_agent(question)
        elif agent_name == "sql":
            return agent_name, run_sql_agent(question)
        elif agent_name == "realtime":
            return agent_name, run_realtime_agent(question)
        elif agent_name == "memory":
            context = retrieve_memory(state["user_id"], state["session_id"], question)
            return agent_name, {"answer": context, "agent": "memory"}
        return agent_name, {"answer": "", "agent": agent_name}

    if len(plan) == 1:
        # single agent — run directly
        name, result = run_agent(plan[0])
        results[name] = result
    else:
        # multiple agents — run in parallel (Map phase)
        with ThreadPoolExecutor(max_workers=len(plan)) as executor:
            futures = {executor.submit(run_agent, agent): agent for agent in plan}
            for future in futures:
                name, result = future.result()
                results[name] = result

    return {**state, "agent_results": results}


# ── Node 4: Merge (Reduce Phase) ───────────────────────────────────────────────

def merge_node(state: SupervisorState) -> SupervisorState:
    """
    Map-Reduce pattern — REDUCE phase.
    Combines results from all agents into one coherent answer.
    Uses LLM to merge intelligently — not just concatenation.
    """
    results = state["agent_results"]

    if len(results) == 1:
        # single agent — no merge needed
        agent_name = list(results.keys())[0]
        answer = results[agent_name].get("answer", "")
        sources = results[agent_name].get("sources", [])
        return {**state, "final_answer": answer, "sources": sources}

    # multiple agents — merge with LLM
    results_text = "\n\n".join(
        f"[{name.upper()} AGENT]:\n{result.get('answer', '')}"
        for name, result in results.items()
    )

    prompt = f"""You are synthesizing results from multiple specialist AI agents.

Original question: {state['question']}

Agent results:
{results_text}

Create ONE clear, coherent answer that:
- Combines all relevant information
- Does not repeat information
- Flows naturally as a single response
- Cites sources where relevant
"""
    response = llm.invoke([{"role": "user", "content": prompt}])

    # collect all sources
    all_sources = []
    for result in results.values():
        all_sources.extend(result.get("sources", []))

    return {
        **state,
        "final_answer": response.content,
        "sources": list(set(all_sources)),
    }


# ── Node 5: Memory Save ────────────────────────────────────────────────────────

def memory_save_node(state: SupervisorState) -> SupervisorState:
    """
    Last node — always runs after answer is generated.
    Passes full conversation to Memory Agent to extract + save new facts.
    Runs silently — user never sees this step.
    """
    conversation = (
        f"User: {state['question']}\n"
        f"Assistant: {state['final_answer']}"
    )
    save_memory(
        user_id=state["user_id"],
        session_id=state["session_id"],
        conversation=conversation,
    )
    return state


# ── Conditional Edge: Check Human Approval ────────────────────────────────────

def needs_human_approval(state: SupervisorState) -> str:
    """
    Human-in-the-loop check.
    Flags sensitive operations for human review before execution.
    Example: delete requests, bulk data changes, financial transactions.
    """
    sensitive_keywords = ["delete", "remove", "drop", "truncate", "transfer", "send money"]
    question_lower = state["question"].lower()

    if any(kw in question_lower for kw in sensitive_keywords):
        return "human_review"
    return "route"


# ── Build Supervisor Graph ─────────────────────────────────────────────────────

def build_supervisor_graph() -> StateGraph:
    """
    Assembles the full supervisor graph with all nodes and edges.

    Graph flow:
      memory_retrieve → plan → needs_human_approval?
                                    │              │
                              human_review       route
                                    │              │
                                   END           merge → memory_save → END

    Checkpointing: MemorySaver saves state after every node.
    Thread ID: each user session gets its own thread for isolated state.
    """
    graph = StateGraph(SupervisorState)

    graph.add_node("memory_retrieve", memory_retrieve_node)
    graph.add_node("planner", plan_node)
    graph.add_node("route", route_node)
    graph.add_node("merge", merge_node)
    graph.add_node("memory_save", memory_save_node)

    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "planner")

    # human-in-the-loop check after planning
    graph.add_conditional_edges(
        "planner",
        needs_human_approval,
        {
            "human_review": END,   # pause — wait for human approval
            "route": "route",      # proceed automatically
        },
    )

    graph.add_edge("route", "merge")
    graph.add_edge("merge", "memory_save")
    graph.add_edge("memory_save", END)

    # compile with checkpointer — enables state persistence per thread
    return graph.compile(checkpointer=checkpointer)


# ── Public Entry Point ─────────────────────────────────────────────────────────

# lazy singleton — built on first call, not at import time
# node names must not clash with SupervisorState keys (plan, route, merge etc.)
_supervisor_graph = None


def _get_graph():
    global _supervisor_graph
    if _supervisor_graph is None:
        _supervisor_graph = build_supervisor_graph()
    return _supervisor_graph


def run_supervisor(
    question: str,
    user_id: str,
    session_id: str,
) -> dict:
    """
    Main entry point — called by FastAPI /chat endpoint.
    Each session_id gets its own checkpoint thread in MemorySaver.

    Returns:
      answer       → final answer to user
      sources      → documents used
      plan         → which agents were called
      agent_results → individual agent outputs
    """
    initial_state: SupervisorState = {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "memory_context": "",
        "plan": [],
        "agent_results": {},
        "final_answer": "",
        "sources": [],
        "tokens_used": 0,
        "requires_human_approval": False,
    }

    # thread_id = session_id ensures each session has isolated checkpoint state
    config = {"configurable": {"thread_id": session_id}}

    final_state = _get_graph().invoke(initial_state, config=config)

    return {
        "answer": final_state["final_answer"],
        "sources": final_state["sources"],
        "plan": final_state["plan"],
        "agent_results": final_state["agent_results"],
    }
