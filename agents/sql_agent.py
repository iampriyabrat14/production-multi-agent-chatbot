from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from tools.nl2sql_engine import NL2SQLEngine
import json
import os

llm = ChatOpenAI(model="gpt-4o", temperature=0)            # SQL generation + format
llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # intent detection + reflection
engine = NL2SQLEngine()


# ── Subgraph State ─────────────────────────────────────────────────────────────

class SQLState(TypedDict):
    """
    State that flows through every node in the SQL subgraph.
    LangGraph checkpoints this after every node.
    """
    question: str               # original user question
    csv_files: list[str]        # detected CSV files needed
    schema: str                 # combined schema of all loaded tables
    sql: str                    # generated SQL query
    reflection_passed: bool     # did SQL pass reflection check?
    result: str                 # final formatted result
    error: str                  # error message if execution fails
    retry_count: int            # number of retries attempted


# ── Node 1: Understand Intent ──────────────────────────────────────────────────

def understand_intent_node(state: SQLState) -> SQLState:
    """
    Detects which CSV file(s) the question needs.
    Lists available CSVs from data/csv_files/ and asks LLM to pick.
    Supports multi-CSV detection for JOIN queries.
    """
    import os
    from pathlib import Path

    csv_dir = Path(os.getenv("CSV_DIR", "data/csv_files"))
    available = [f.name for f in csv_dir.glob("*.csv")] if csv_dir.exists() else []

    if not available:
        return {**state, "csv_files": [], "error": "No CSV files found in data/csv_files/"}

    prompt = f"""You are a data analyst. Given a user question, identify which CSV files are needed.

Available CSV files:
{json.dumps(available, indent=2)}

User question: {state['question']}

Rules:
- Return a JSON array of CSV filenames needed
- Include multiple files if a JOIN is needed
- Return ONLY the JSON array, nothing else
- Example: ["sales.csv"] or ["sales.csv", "customers.csv"]
"""
    response = llm_fast.invoke([{"role": "user", "content": prompt}])

    try:
        csv_files = json.loads(response.content.strip())
        if not isinstance(csv_files, list):
            csv_files = [csv_files]
    except json.JSONDecodeError:
        csv_files = available[:1]   # fallback to first CSV

    return {**state, "csv_files": csv_files, "error": ""}


# ── Node 2: Generate SQL ───────────────────────────────────────────────────────

def generate_sql_node(state: SQLState) -> SQLState:
    """
    Loads CSV(s) into DuckDB and generates SQL using NL2SQLEngine.
    Schema is extracted from loaded tables for accurate SQL generation.
    """
    if not state["csv_files"]:
        return {**state, "sql": "", "error": "No CSV files identified"}

    # load all CSVs and get combined schema
    table_names = engine.load_multiple_csvs(state["csv_files"])
    schema = (
        engine.get_schema(table_names[0])
        if len(table_names) == 1
        else engine.get_multi_schema(table_names)
    )

    # generate SQL from question + schema
    sql = engine.generate_sql(state["question"], schema)

    return {**state, "sql": sql, "schema": schema}


# ── Node 3: Reflect on SQL ─────────────────────────────────────────────────────

def reflect_sql_node(state: SQLState) -> SQLState:
    """
    Reflection step — LLM validates generated SQL before execution.
    Critical for SQL because wrong queries return wrong data silently.

    Checks:
      - Correct column names (from schema)
      - Proper GROUP BY for aggregations
      - Valid JOIN conditions
      - No SQL injection patterns
    """
    if not state["sql"]:
        return {**state, "reflection_passed": False}

    prompt = f"""You are a SQL reviewer. Validate this SQL query against the schema.

Schema:
{state['schema']}

SQL query:
{state['sql']}

Check for:
1. Correct column names matching the schema exactly
2. Proper GROUP BY clauses for aggregations
3. Valid JOIN conditions if multiple tables used
4. No syntax errors

Return JSON: {{"valid": true/false, "corrected_sql": "...", "reason": "..."}}
If valid, corrected_sql = original sql unchanged.
Return ONLY the JSON, nothing else.
"""
    response = llm_fast.invoke([{"role": "user", "content": prompt}])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        result = json.loads(content)
        passed = result.get("valid", True)
        corrected_sql = result.get("corrected_sql", state["sql"]) or state["sql"]
    except json.JSONDecodeError:
        passed = True
        corrected_sql = state["sql"]

    return {
        **state,
        "reflection_passed": passed,
        "sql": corrected_sql,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# ── Node 4: Execute SQL ────────────────────────────────────────────────────────

def execute_sql_node(state: SQLState) -> SQLState:
    """
    Executes the validated SQL on DuckDB.
    Returns formatted result string or error message.
    """
    if not state["sql"]:
        return {**state, "result": "", "error": "No SQL generated"}

    try:
        result_df = engine.execute_query(state["sql"])

        if result_df.empty:
            return {**state, "result": "No results found for your query.", "error": ""}

        result = (
            f"Query: {state['sql']}\n\n"
            f"Result:\n{result_df.to_string(index=False)}"
        )
        return {**state, "result": result, "error": ""}

    except Exception as e:
        return {**state, "result": "", "error": str(e)}


# ── Node 5: Format Result ──────────────────────────────────────────────────────

def format_result_node(state: SQLState) -> SQLState:
    """
    Converts raw SQL result into a natural language answer using LLM.
    Makes the output readable for non-technical users.
    """
    if state["error"]:
        return {**state, "result": f"I could not answer that query: {state['error']}"}

    prompt = f"""Convert this SQL query result into a clear, natural language answer.

Original question: {state['question']}
SQL result:
{state['result']}

Write a concise, friendly answer. Include key numbers and insights.
Do not mention SQL or technical details."""

    response = llm.invoke([{"role": "user", "content": prompt}])
    return {**state, "result": response.content}


# ── Conditional Edge: Retry or Execute ────────────────────────────────────────

def reflection_passed(state: SQLState) -> str:
    """
    After reflection — if SQL is invalid and retries remain, regenerate.
    Otherwise proceed to execution.
    """
    if not state["reflection_passed"] and state.get("retry_count", 0) < 2:
        return "retry"
    return "execute"


# ── Build SQL Subgraph ─────────────────────────────────────────────────────────

def build_sql_graph() -> StateGraph:
    """
    Assembles all SQL nodes into a LangGraph subgraph.

    Graph flow:
      understand_intent → generate_sql → reflect_sql
                               ↑               │
                               └─── retry ─────┤ (invalid SQL, max 2x)
                                               │
                                          execute_sql → format_result → END
    """
    graph = StateGraph(SQLState)

    graph.add_node("understand_intent", understand_intent_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("reflect_sql", reflect_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("format_result", format_result_node)

    graph.set_entry_point("understand_intent")
    graph.add_edge("understand_intent", "generate_sql")
    graph.add_edge("generate_sql", "reflect_sql")

    # conditional edge — retry SQL generation or proceed to execution
    graph.add_conditional_edges(
        "reflect_sql",
        reflection_passed,
        {
            "retry": "generate_sql",
            "execute": "execute_sql",
        },
    )

    graph.add_edge("execute_sql", "format_result")
    graph.add_edge("format_result", END)

    return graph.compile()


# ── Public Entry Point ─────────────────────────────────────────────────────────

_sql_graph = None

def _get_sql_graph():
    global _sql_graph
    if _sql_graph is None:
        _sql_graph = build_sql_graph()
    return _sql_graph


def run_sql_agent(question: str) -> dict:
    """
    Run the SQL subgraph for a given question.
    Returns natural language answer + SQL used.
    Called by the Supervisor Agent in Step 16.
    """
    sql_graph = _get_sql_graph()

    initial_state: SQLState = {
        "question": question,
        "csv_files": [],
        "schema": "",
        "sql": "",
        "reflection_passed": False,
        "result": "",
        "error": "",
        "retry_count": 0,
    }

    final_state = sql_graph.invoke(initial_state)

    return {
        "answer": final_state["result"],
        "sql_used": final_state["sql"],
        "agent": "sql_agent",
    }
