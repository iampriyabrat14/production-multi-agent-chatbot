from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate
from langchain_openai import ChatOpenAI
from datetime import datetime
import os

# ── Setup ──────────────────────────────────────────────────────────────────────

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "production-chatbot")

client = Client(api_key=LANGCHAIN_API_KEY)
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# ── Dataset Management ─────────────────────────────────────────────────────────

def create_dataset(name: str, description: str = "") -> str:
    """
    Create a named evaluation dataset in LangSmith.
    Datasets store Q&A pairs used for automated evaluation.

    Returns dataset ID.

    Usage:
      dataset_id = create_dataset("RAG eval set - v1")
      add_examples(dataset_id, [...])
      run_evaluation(dataset_id)
    """
    dataset = client.create_dataset(
        dataset_name=name,
        description=description or f"Evaluation dataset created {datetime.utcnow().date()}",
    )
    return str(dataset.id)


def add_examples(dataset_id: str, examples: list[dict]) -> int:
    """
    Add Q&A pairs to a LangSmith dataset.

    Args:
      dataset_id : ID returned by create_dataset()
      examples   : list of {"question": "...", "answer": "..."} dicts

    Returns number of examples added.
    """
    client.create_examples(
        inputs=[{"question": e["question"]} for e in examples],
        outputs=[{"answer": e["answer"]} for e in examples],
        dataset_id=dataset_id,
    )
    return len(examples)


# ── Run Logging ────────────────────────────────────────────────────────────────

def log_run(
    question: str,
    answer: str,
    agent_used: str,
    latency_ms: float,
    tokens_used: int,
    sources: list[str],
    ragas_scores: dict = None,
) -> None:
    """
    Log a single agent run to LangSmith for observability.
    Every /chat request logs here — full traceability.

    LangSmith dashboard shows:
      - Which agents were called
      - Latency per agent
      - Token usage
      - RAGAS scores if available
      - Full prompt + response
    """
    metadata = {
        "agent_used": agent_used,
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if ragas_scores:
        metadata["ragas_faithfulness"] = ragas_scores.get("faithfulness", 0)
        metadata["ragas_relevancy"] = ragas_scores.get("answer_relevancy", 0)
        metadata["ragas_passed"] = ragas_scores.get("passed", False)

    client.create_run(
        name=f"chat_{agent_used}",
        run_type="chain",
        inputs={"question": question},
        outputs={"answer": answer},
        extra={"metadata": metadata},
        project_name=LANGCHAIN_PROJECT,
    )


# ── LLM-as-Judge Evaluators ────────────────────────────────────────────────────

def correctness_evaluator(run, example) -> dict:
    """
    LLM-as-judge evaluator — scores answer correctness.
    Used by LangSmith's evaluate() function during batch evaluation.

    Compares generated answer against ground truth answer.
    Returns score 0-1 and reasoning.
    """
    question = example.inputs.get("question", "")
    generated = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")

    prompt = f"""Score how correct this answer is compared to the expected answer.

Question: {question}
Expected answer: {expected}
Generated answer: {generated}

Score from 0.0 to 1.0:
  1.0 = completely correct
  0.5 = partially correct
  0.0 = completely wrong

Return ONLY a number between 0.0 and 1.0."""

    response = llm.invoke([{"role": "user", "content": prompt}])

    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.0

    return {"key": "correctness", "score": score}


def conciseness_evaluator(run, example) -> dict:
    """
    LLM-as-judge evaluator — scores answer conciseness.
    Penalizes overly verbose or padded answers.
    """
    generated = run.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    prompt = f"""Score how concise this answer is for the question asked.

Question: {question}
Answer: {generated}

Score from 0.0 to 1.0:
  1.0 = perfectly concise, no filler
  0.5 = somewhat verbose but acceptable
  0.0 = extremely verbose or padded

Return ONLY a number between 0.0 and 1.0."""

    response = llm.invoke([{"role": "user", "content": prompt}])

    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.5

    return {"key": "conciseness", "score": score}


# ── Full Evaluation Pipeline ───────────────────────────────────────────────────

def run_evaluation(dataset_name: str, pipeline_fn) -> dict:
    """
    Run full LangSmith evaluation against a named dataset.
    Uses LLM-as-judge evaluators for correctness + conciseness.

    Args:
      dataset_name : name of dataset in LangSmith
      pipeline_fn  : function that takes {"question": str} and returns {"answer": str}

    Returns summary of evaluation results.

    Usage:
      from agents.supervisor import run_supervisor

      def pipeline(inputs):
          result = run_supervisor(inputs["question"], "eval_user", "eval_session")
          return {"answer": result["answer"]}

      results = run_evaluation("RAG eval set - v1", pipeline)
    """
    results = ls_evaluate(
        pipeline_fn,
        data=dataset_name,
        evaluators=[correctness_evaluator, conciseness_evaluator],
        experiment_prefix="chatbot_eval",
        metadata={"project": LANGCHAIN_PROJECT},
    )

    return {
        "experiment_name": results.experiment_name,
        "results_url": f"https://smith.langchain.com/projects/{LANGCHAIN_PROJECT}",
        "total_examples": len(results),
    }


# ── Summary Report ─────────────────────────────────────────────────────────────

def get_project_summary() -> dict:
    """
    Fetch summary stats from LangSmith for the current project.
    Shows total runs, average latency, error rate.
    """
    runs = list(client.list_runs(
        project_name=LANGCHAIN_PROJECT,
        limit=100,
    ))

    if not runs:
        return {"message": "No runs found in LangSmith project"}

    total = len(runs)
    errors = sum(1 for r in runs if r.error is not None)
    latencies = [
        (r.end_time - r.start_time).total_seconds() * 1000
        for r in runs
        if r.end_time and r.start_time
    ]

    return {
        "project": LANGCHAIN_PROJECT,
        "total_runs": total,
        "error_rate": round(errors / total, 4) if total else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
    }
