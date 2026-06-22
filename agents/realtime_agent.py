from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tools.currency_tool import CurrencyTool
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0)
currency_tool = CurrencyTool()


# ── Subgraph State ─────────────────────────────────────────────────────────────

class RealtimeState(TypedDict):
    """
    State flowing through the Realtime subgraph.
    Checkpointed by LangGraph after every node.
    """
    question: str           # original user question
    base_currency: str      # extracted source currency e.g. "EUR"
    target_currency: str    # extracted target currency e.g. "USD"
    amount: float           # amount to convert (default 1.0 for rate only)
    tool_result: dict       # raw result from CurrencyTool
    answer: str             # final formatted answer
    error: str              # error message if API call fails


# ── Node 1: Parse Intent ───────────────────────────────────────────────────────

def parse_intent_node(state: RealtimeState) -> RealtimeState:
    """
    ReAct — REASON step.
    Extracts currencies and amount from natural language question.

    Examples:
      "What is EUR/USD rate?"         → base=EUR, target=USD, amount=1.0
      "Convert 500 EUR to USD"        → base=EUR, target=USD, amount=500.0
      "How much is 1000 GBP in INR?"  → base=GBP, target=INR, amount=1000.0
    """
    prompt = f"""Extract currency information from this question.

Question: {state['question']}

Return JSON with these exact keys:
{{
  "base_currency": "3-letter currency code e.g. EUR",
  "target_currency": "3-letter currency code e.g. USD",
  "amount": number (default 1.0 if not specified)
}}

Return ONLY the JSON, nothing else."""

    response = llm.invoke([{"role": "user", "content": prompt}])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
        return {
            **state,
            "base_currency": parsed.get("base_currency", "EUR").upper(),
            "target_currency": parsed.get("target_currency", "USD").upper(),
            "amount": float(parsed.get("amount", 1.0)),
            "error": "",
        }
    except (json.JSONDecodeError, ValueError):
        # fallback: extract currencies directly from question text
        import re
        codes = re.findall(r'\b([A-Z]{3})\b', state["question"].upper())
        base = codes[0] if len(codes) >= 1 else "EUR"
        target = codes[1] if len(codes) >= 2 else "USD"
        return {
            **state,
            "base_currency": base,
            "target_currency": target,
            "amount": 1.0,
            "error": "",
        }


# ── Node 2: Call Tool ──────────────────────────────────────────────────────────

def call_tool_node(state: RealtimeState) -> RealtimeState:
    """
    ReAct — ACT step.
    Calls CurrencyTool to fetch live exchange rate.
    This is where the real-time API call happens.
    Uses asyncio.run() to call async tool from sync LangGraph node.
    """
    if state["error"]:
        return state

    try:
        if state["amount"] == 1.0:
            result = currency_tool.get_rate(state["base_currency"], state["target_currency"])
        else:
            result = currency_tool.convert(
                state["amount"],
                state["base_currency"],
                state["target_currency"],
            )
        return {**state, "tool_result": result, "error": ""}

    except Exception as e:
        return {**state, "tool_result": {}, "error": str(e)}


# ── Node 3: Handle Error ───────────────────────────────────────────────────────

def handle_error_node(state: RealtimeState) -> RealtimeState:
    """
    Graceful fallback if currency API is down or returns error.
    Returns a helpful message instead of crashing.
    """
    if not state["error"]:
        return state

    fallback = (
        f"I was unable to fetch the live {state['base_currency']}/{state['target_currency']} "
        f"rate at this moment. The currency API may be temporarily unavailable. "
        f"Please try again in a few seconds."
    )
    return {**state, "answer": fallback}


# ── Node 4: Format Response ────────────────────────────────────────────────────

def format_response_node(state: RealtimeState) -> RealtimeState:
    """
    ReAct — OBSERVE + final REASON step.
    Converts raw tool result into a clean, friendly answer.
    """
    if state["error"] or not state["tool_result"]:
        return state

    result = state["tool_result"]

    # rate only query
    if state["amount"] == 1.0:
        answer = (
            f"Live exchange rate (as of {result['timestamp']}):\n"
            f"1 {result['base']} = {result['rate']:.4f} {result['target']}"
        )
    else:
        # conversion query
        answer = (
            f"Currency conversion (as of {result['timestamp']}):\n"
            f"{result['amount']:,.2f} {result['base']} = "
            f"{result['converted']:,.2f} {result['target']}\n"
            f"Exchange rate: 1 {result['base']} = {result['rate']:.4f} {result['target']}"
        )

    return {**state, "answer": answer}


# ── Conditional Edge: Error or Format ─────────────────────────────────────────

def has_error(state: RealtimeState) -> str:
    """Route to error handler if API failed, otherwise format response."""
    return "error" if state["error"] else "format"


# ── Build Realtime Subgraph ────────────────────────────────────────────────────

def build_realtime_graph() -> StateGraph:
    """
    Assembles all Realtime nodes into a LangGraph subgraph.

    Graph flow:
      parse_intent → call_tool
                         │
              ┌──────────┴──────────┐
           error?               success?
              │                     │
        handle_error        format_response
              │                     │
              └──────────┬──────────┘
                        END
    """
    graph = StateGraph(RealtimeState)

    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("call_tool", call_tool_node)
    graph.add_node("handle_error", handle_error_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "call_tool")

    # conditional edge — error handling or format response
    graph.add_conditional_edges(
        "call_tool",
        has_error,
        {
            "error": "handle_error",
            "format": "format_response",
        },
    )

    graph.add_edge("handle_error", END)
    graph.add_edge("format_response", END)

    return graph.compile()


# ── Public Entry Point ─────────────────────────────────────────────────────────

_realtime_graph = None

def _get_realtime_graph():
    global _realtime_graph
    if _realtime_graph is None:
        _realtime_graph = build_realtime_graph()
    return _realtime_graph


def run_realtime_agent(question: str) -> dict:
    """
    Run the Realtime subgraph for a given question.
    Returns live answer + tool result.
    Called by the Supervisor Agent in Step 16.
    """
    realtime_graph = _get_realtime_graph()

    initial_state: RealtimeState = {
        "question": question,
        "base_currency": "",
        "target_currency": "",
        "amount": 1.0,
        "tool_result": {},
        "answer": "",
        "error": "",
    }

    final_state = realtime_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "tool_result": final_state["tool_result"],
        "agent": "realtime_agent",
    }
