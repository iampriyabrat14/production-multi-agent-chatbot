from unittest.mock import patch, MagicMock


def make_state(**overrides):
    base = {
        "user_id": "user_1",
        "session_id": "session_1",
        "question": "test question",
        "memory_context": "",
        "plan": [],
        "agent_results": {},
        "final_answer": "",
        "sources": [],
        "tokens_used": 0,
        "requires_human_approval": False,
    }
    base.update(overrides)
    return base


class TestSupervisorNodes:

    @patch("agents.supervisor.retrieve_memory")
    def test_memory_retrieve_node_loads_context(self, mock_mem):
        mock_mem.return_value = "User prefers USD. Past: asked about Q3 sales."

        from agents.supervisor import memory_retrieve_node
        state = make_state(question="What is EUR/USD?")
        result = memory_retrieve_node(state)

        assert "memory_context" in result

    @patch("agents.supervisor.llm")
    def test_plan_node_returns_agent_list(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='["rag"]'
        )

        from agents.supervisor import plan_node
        state = make_state(question="What is the refund policy?", memory_context="")
        result = plan_node(state)

        assert "plan" in result
        assert isinstance(result["plan"], list)

    @patch("agents.supervisor.run_rag_agent")
    def test_route_node_calls_rag_for_rag_plan(self, mock_rag):
        mock_rag.return_value = {"answer": "Policy: 7 day refunds.", "sources": ["policy.pdf"]}

        from agents.supervisor import route_node
        state = make_state(question="What is refund policy?", plan=["rag"])
        result = route_node(state)

        assert "agent_results" in result
        mock_rag.assert_called_once()

    @patch("agents.supervisor.run_sql_agent")
    def test_route_node_calls_sql_for_sql_plan(self, mock_sql):
        mock_sql.return_value = {"answer": "Q3 revenue: $2.4M", "sources": ["sales.csv"]}

        from agents.supervisor import route_node
        state = make_state(question="Show Q3 revenue", plan=["sql"])
        result = route_node(state)

        assert "agent_results" in result
        mock_sql.assert_called_once()

    @patch("agents.supervisor.run_rag_agent")
    @patch("agents.supervisor.run_sql_agent")
    def test_route_node_runs_parallel_agents(self, mock_sql, mock_rag):
        mock_rag.return_value = {"answer": "Policy answer.", "sources": []}
        mock_sql.return_value = {"answer": "SQL answer.", "sources": []}

        from agents.supervisor import route_node
        state = make_state(question="Policy and sales?", plan=["rag", "sql"])
        result = route_node(state)

        assert "rag" in result["agent_results"] or "sql" in result["agent_results"]

    @patch("agents.supervisor.llm")
    def test_merge_node_combines_answers(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content="EUR/USD is 1.085. Q3 revenue was $2.4M."
        )

        from agents.supervisor import merge_node
        state = make_state(
            question="Rate and revenue?",
            agent_results={
                "realtime": {"answer": "EUR/USD = 1.085", "sources": []},
                "sql": {"answer": "Q3 = $2.4M", "sources": ["sales.csv"]}
            }
        )
        result = merge_node(state)

        assert result["final_answer"] != ""

    @patch("agents.supervisor.save_memory")
    def test_memory_save_node_stores_facts(self, mock_save):
        mock_save.return_value = None

        from agents.supervisor import memory_save_node
        state = make_state(
            final_answer="EUR/USD is 1.085.",
            question="What is EUR/USD?",
        )
        result = memory_save_node(state)

        assert "final_answer" in result
