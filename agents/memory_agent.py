from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tools.memory_store import MemoryStore
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # memory extraction is simple classification


# ── Subgraph State ─────────────────────────────────────────────────────────────

class MemoryAgentState(TypedDict):
    """
    State flowing through the Memory subgraph.
    Checkpointed by LangGraph after every node.
    """
    user_id: str                # which user
    session_id: str             # which session
    question: str               # current user question
    conversation: str           # full conversation text for fact extraction
    existing_facts: dict        # facts already known about user
    extracted_facts: dict       # new facts found in this conversation
    context: str                # memory context injected into agent prompts
    facts_saved: int            # number of new facts saved


# ── Node 1: Retrieve ───────────────────────────────────────────────────────────

def retrieve_node(state: MemoryAgentState) -> MemoryAgentState:
    """
    Runs BEFORE other agents.
    Loads all known facts + recent history for this user.
    Returns context string to inject into supervisor + agent prompts.
    """
    store = MemoryStore(user_id=state["user_id"], session_id=state["session_id"])
    store.load_session(last_n=10)

    existing_facts = store.get_facts()
    context = store.get_context()

    return {
        **state,
        "existing_facts": existing_facts,
        "context": context,
    }


# ── Node 2: Extract Facts ──────────────────────────────────────────────────────

def extract_facts_node(state: MemoryAgentState) -> MemoryAgentState:
    """
    Runs AFTER other agents answer the user.
    LLM reads the full conversation and extracts new important facts.

    Only extracts facts that:
      - Are not already known (not in existing_facts)
      - Are genuinely useful for future conversations
      - Are stated clearly by the user (not guessed)
    """
    if not state["conversation"]:
        return {**state, "extracted_facts": {}}

    existing_str = json.dumps(state["existing_facts"], indent=2)

    prompt = f"""You are a memory extraction specialist.
Read this conversation and extract NEW important facts about the user.

Already known facts (do NOT re-extract these):
{existing_str}

Conversation:
{state['conversation']}

Extract facts that are:
- Explicitly stated by the user (not assumed)
- Useful for future conversations (preferences, role, company, habits)
- NOT already in the known facts list

Return JSON object with fact key-value pairs.
If no new facts found, return empty object {{}}.
Return ONLY the JSON, nothing else.

Examples of good facts:
  {{"preferred_currency": "USD", "company": "Acme Corp", "role": "Finance Manager"}}
"""

    response = llm.invoke([{"role": "user", "content": prompt}])

    try:
        extracted = json.loads(response.content.strip())
        if not isinstance(extracted, dict):
            extracted = {}
    except json.JSONDecodeError:
        extracted = {}

    return {**state, "extracted_facts": extracted}


# ── Node 3: Save Facts ─────────────────────────────────────────────────────────

def save_facts_node(state: MemoryAgentState) -> MemoryAgentState:
    """
    Saves newly extracted facts to PostgreSQL long-term memory.
    Only saves if there are new facts to save.
    """
    if not state["extracted_facts"]:
        return {**state, "facts_saved": 0}

    store = MemoryStore(user_id=state["user_id"], session_id=state["session_id"])

    count = 0
    for key, value in state["extracted_facts"].items():
        store.save_fact(key, str(value))
        count += 1

    return {**state, "facts_saved": count}


# ── Conditional Edge: Save or Skip ────────────────────────────────────────────

def has_new_facts(state: MemoryAgentState) -> str:
    """Only proceed to save if new facts were extracted."""
    return "save" if state["extracted_facts"] else "skip"


# ── Build Memory Subgraph ──────────────────────────────────────────────────────

def build_memory_graph() -> StateGraph:
    """
    Assembles memory nodes into a LangGraph subgraph.

    Two modes depending on how it's called:

    BEFORE query (retrieve mode):
      retrieve → END
      Returns context to inject into other agents

    AFTER query (extract + save mode):
      extract_facts → has_new_facts?
                           │
                    ┌──────┴──────┐
                  save          skip
                    │              │
                   END            END
    """
    graph = StateGraph(MemoryAgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("save_facts", save_facts_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "extract_facts")

    graph.add_conditional_edges(
        "extract_facts",
        has_new_facts,
        {
            "save": "save_facts",
            "skip": END,
        },
    )

    graph.add_edge("save_facts", END)

    return graph.compile()


# ── Public Entry Points ────────────────────────────────────────────────────────

def retrieve_memory(user_id: str, session_id: str, question: str) -> str:
    """
    Called by Supervisor BEFORE routing to specialist agents.
    Returns memory context string to inject into agent prompts.
    """
    memory_graph = build_memory_graph()

    initial_state: MemoryAgentState = {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "conversation": "",
        "existing_facts": {},
        "extracted_facts": {},
        "context": "",
        "facts_saved": 0,
    }

    final_state = memory_graph.invoke(initial_state)
    return final_state["context"]


def save_memory(user_id: str, session_id: str, conversation: str) -> int:
    """
    Called by Supervisor AFTER generating answer.
    1. Saves both messages to conversation_history
    2. Extracts and saves new facts to user_facts
    Returns number of new facts saved.
    """
    store = MemoryStore(user_id=user_id, session_id=session_id)

    # parse "User: ...\nAssistant: ..." and save each message to conversation_history
    lines = conversation.strip().split("\n")
    for line in lines:
        if line.startswith("User: "):
            store.add_message("human", line[len("User: "):].strip())
        elif line.startswith("Assistant: "):
            store.add_message("ai", line[len("Assistant: "):].strip())

    # extract + save new user facts from conversation
    existing_facts = store.get_facts()

    existing_str = json.dumps(existing_facts, indent=2)
    prompt = f"""You are a memory extraction specialist.
Read this conversation and extract NEW important facts about the user.

Already known facts (do NOT re-extract these):
{existing_str}

Conversation:
{conversation}

Extract facts that are:
- Explicitly stated by the user (not assumed)
- Useful for future conversations (preferences, role, company, habits)
- NOT already in the known facts list

Return JSON object with fact key-value pairs.
If no new facts found, return empty object {{}}.
Return ONLY the JSON, nothing else.

Examples of good facts:
  {{"preferred_currency": "USD", "company": "Acme Corp", "role": "Finance Manager"}}
"""
    response = llm.invoke([{"role": "user", "content": prompt}])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        extracted = json.loads(content)
        if not isinstance(extracted, dict):
            extracted = {}
    except json.JSONDecodeError:
        extracted = {}

    count = 0
    for key, value in extracted.items():
        store.save_fact(key, str(value))
        count += 1

    return count
