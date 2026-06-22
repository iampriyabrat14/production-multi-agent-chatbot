from unittest.mock import patch, MagicMock


# ── Test individual RAG node functions (no full graph needed) ─────────────────

class TestRAGNodes:

    @patch("tools.pinecone_retriever.PineconeRetriever.retrieve")
    def test_retrieve_node_populates_chunks(self, mock_retrieve):
        mock_retrieve.return_value = [
            {"text": "Refunds take 7 days.", "source": "policy.pdf", "score": 0.92}
        ]

        from agents.rag_agent import retrieve_node, RAGState
        state: RAGState = {
            "question": "What is the refund policy?",
            "chunks": [], "relevant_chunks": [], "answer": "",
            "reflection_score": 0.0, "retry_count": 0, "sources": []
        }
        result = retrieve_node(state)

        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["text"] == "Refunds take 7 days."

    @patch("tools.pinecone_retriever.PineconeRetriever.retrieve")
    def test_retrieve_node_empty_when_no_results(self, mock_retrieve):
        mock_retrieve.return_value = []

        from agents.rag_agent import retrieve_node, RAGState
        state: RAGState = {
            "question": "Unknown topic",
            "chunks": [], "relevant_chunks": [], "answer": "",
            "reflection_score": 0.0, "retry_count": 0, "sources": []
        }
        result = retrieve_node(state)

        assert result["chunks"] == []

    def test_grade_chunks_node_filters_low_score(self):
        from agents.rag_agent import grade_chunks_node, RAGState
        state: RAGState = {
            "question": "refund policy",
            "chunks": [
                {"text": "Relevant chunk", "source": "doc.pdf", "score": 0.85},
                {"text": "Irrelevant chunk", "source": "other.pdf", "score": 0.40},
            ],
            "relevant_chunks": [], "answer": "",
            "reflection_score": 0.0, "retry_count": 0, "sources": []
        }
        result = grade_chunks_node(state)

        # Only high-score chunk should pass
        assert len(result["relevant_chunks"]) == 1
        assert result["relevant_chunks"][0]["score"] == 0.85

    @patch("agents.rag_agent.llm")
    def test_generate_node_produces_answer(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="Refunds are processed in 7 days.")

        from agents.rag_agent import generate_node, RAGState
        state: RAGState = {
            "question": "What is the refund policy?",
            "chunks": [],
            "relevant_chunks": [{"text": "Refunds take 7 days.", "source": "policy.pdf", "score": 0.92}],
            "answer": "",
            "reflection_score": 0.0, "retry_count": 0, "sources": []
        }
        result = generate_node(state)

        assert result["answer"] != ""
        assert "sources" in result

    @patch("agents.rag_agent.llm")
    def test_reflect_node_scores_answer(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='{"score": 0.9, "reason": "well grounded in context"}'
        )

        from agents.rag_agent import reflect_node, RAGState
        state: RAGState = {
            "question": "refund?",
            "chunks": [],
            "relevant_chunks": [{"text": "7 day refund.", "source": "doc.pdf", "score": 0.9}],
            "answer": "Refunds take 7 days.",
            "reflection_score": 0.0, "retry_count": 0, "sources": []
        }
        result = reflect_node(state)

        assert result["reflection_score"] >= 0.0
