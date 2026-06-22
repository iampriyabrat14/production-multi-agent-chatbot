from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

# ── Setup ──────────────────────────────────────────────────────────────────────

# RAGAS uses LLM + embeddings internally to score answers
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# target scores — alert if any metric drops below these
SCORE_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.75,
    "context_precision": 0.75,
}


# ── Single Response Evaluation ─────────────────────────────────────────────────

def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = "",
) -> dict:
    """
    Score a single RAG response on 4 RAGAS metrics.

    Args:
      question    : the user's original question
      answer      : the RAG agent's generated answer
      contexts    : list of retrieved chunk texts used to generate answer
      ground_truth: optional correct answer for context_recall scoring

    Returns:
      {
        "faithfulness": 0.92,
        "answer_relevancy": 0.88,
        "context_recall": 0.79,
        "context_precision": 0.81,
        "passed": True,
        "failed_metrics": []
      }

    Metrics explained:
      faithfulness      → answer is grounded in retrieved context (not hallucinated)
      answer_relevancy  → answer actually addresses the question asked
      context_recall    → retrieved chunks contain the correct answer
      context_precision → retrieved chunks are relevant (no noise)
    """
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth if ground_truth else answer],
    }

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores = {
        "faithfulness": round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_recall": round(float(results["context_recall"]), 4),
        "context_precision": round(float(results["context_precision"]), 4),
    }

    # check which metrics failed thresholds
    failed = [
        metric for metric, score in scores.items()
        if score < SCORE_THRESHOLDS[metric]
    ]

    scores["passed"] = len(failed) == 0
    scores["failed_metrics"] = failed

    return scores


# ── Batch Evaluation ───────────────────────────────────────────────────────────

def run_batch_evaluation(samples: list[dict]) -> dict:
    """
    Evaluate a dataset of Q&A pairs in batch.
    Used for regression testing after changes to RAG pipeline.

    Args:
      samples: list of dicts with keys:
        - question
        - answer
        - contexts (list of strings)
        - ground_truth (optional)

    Returns:
      {
        "average_scores": {...},
        "passed_count": 8,
        "failed_count": 2,
        "total": 10,
        "pass_rate": 0.80
      }
    """
    data = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s.get("ground_truth", s["answer"]) for s in samples],
    }

    dataset = Dataset.from_dict(data)
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    avg_scores = {
        "faithfulness": round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_recall": round(float(results["context_recall"]), 4),
        "context_precision": round(float(results["context_precision"]), 4),
    }

    passed = sum(
        1 for metric, score in avg_scores.items()
        if score >= SCORE_THRESHOLDS[metric]
    )

    return {
        "average_scores": avg_scores,
        "passed_metrics": passed,
        "failed_metrics": len(avg_scores) - passed,
        "total_samples": len(samples),
    }


# ── Format Scores ──────────────────────────────────────────────────────────────

def format_scores(scores: dict) -> str:
    """
    Format evaluation scores as a readable report string.

    Example output:
      RAGAS Evaluation Report
      ─────────────────────────────────
      Faithfulness      : 0.9200  ✅ (threshold: 0.85)
      Answer Relevancy  : 0.8800  ✅ (threshold: 0.80)
      Context Recall    : 0.7200  ❌ (threshold: 0.75)
      Context Precision : 0.8100  ✅ (threshold: 0.75)
      ─────────────────────────────────
      Overall: FAILED — context_recall below threshold
    """
    lines = ["RAGAS Evaluation Report", "─" * 40]

    for metric, threshold in SCORE_THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        status = "✅" if score >= threshold else "❌"
        lines.append(
            f"{metric.replace('_', ' ').title():<22}: {score:.4f}  {status} (threshold: {threshold})"
        )

    lines.append("─" * 40)

    if scores.get("passed"):
        lines.append("Overall: PASSED ✅")
    else:
        failed = ", ".join(scores.get("failed_metrics", []))
        lines.append(f"Overall: FAILED ❌ — {failed} below threshold")

    return "\n".join(lines)
