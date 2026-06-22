from unittest.mock import patch, MagicMock


# ── Test individual SQL node functions (no full graph needed) ─────────────────

class TestSQLNodes:

    @patch("agents.sql_agent.llm")
    def test_understand_intent_extracts_csv_files(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='{"csv_files": ["sales.csv"], "intent": "aggregate revenue"}'
        )

        from agents.sql_agent import understand_intent_node, SQLState
        state: SQLState = {
            "question": "What is Q3 total revenue?",
            "csv_files": [], "schema": "", "sql": "",
            "reflection_passed": False, "result": "", "error": "", "retry_count": 0
        }
        result = understand_intent_node(state)
        assert "csv_files" in result

    @patch("agents.sql_agent.engine.generate_sql")
    @patch("agents.sql_agent.engine.load_multiple_csvs")
    @patch("agents.sql_agent.engine.get_schema")
    def test_generate_sql_produces_query(self, mock_schema, mock_load, mock_gen_sql):
        mock_load.return_value = ["sales"]
        mock_schema.return_value = "sales(order_id, total_revenue, quarter)"
        mock_gen_sql.return_value = "SELECT SUM(total_revenue) as total FROM sales WHERE quarter = 'Q3'"

        from agents.sql_agent import generate_sql_node, SQLState
        state: SQLState = {
            "question": "What is Q3 total revenue?",
            "csv_files": ["sales.csv"],
            "schema": "sales(order_id, total_revenue, quarter)",
            "sql": "", "reflection_passed": False,
            "result": "", "error": "", "retry_count": 0
        }
        result = generate_sql_node(state)

        assert result["sql"] != ""
        assert "SELECT" in result["sql"].upper()

    @patch("agents.sql_agent.llm")
    def test_reflect_sql_passes_valid_query(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='{"valid": true, "feedback": "Query looks correct"}'
        )

        from agents.sql_agent import reflect_sql_node, SQLState
        state: SQLState = {
            "question": "Total revenue?",
            "csv_files": ["sales.csv"], "schema": "sales(total_revenue)",
            "sql": "SELECT SUM(total_revenue) FROM sales",
            "reflection_passed": False,
            "result": "", "error": "", "retry_count": 0
        }
        result = reflect_sql_node(state)

        assert result["reflection_passed"] is True

    @patch("agents.sql_agent.llm")
    def test_reflect_sql_fails_invalid_query(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='{"valid": false, "feedback": "FORM is a typo, should be FROM"}'
        )

        from agents.sql_agent import reflect_sql_node, SQLState
        state: SQLState = {
            "question": "Total?",
            "csv_files": ["sales.csv"], "schema": "sales(total_revenue)",
            "sql": "SELECT SUM(total_revenue) FORM sales",
            "reflection_passed": False,
            "result": "", "error": "", "retry_count": 0
        }
        result = reflect_sql_node(state)

        assert result["reflection_passed"] is False


# ── Test NL2SQL engine directly (pure DuckDB, no LLM) ────────────────────────

class TestNL2SQLEngine:

    def test_direct_query_returns_results(self):
        from tools.nl2sql_engine import NL2SQLEngine
        engine = NL2SQLEngine()
        engine.load_csv("sales.csv")
        result = engine.execute_query("SELECT COUNT(*) as total FROM sales")
        assert result is not None
        assert len(result) > 0

    def test_schema_extraction_has_columns(self):
        from tools.nl2sql_engine import NL2SQLEngine
        engine = NL2SQLEngine()
        engine.load_csv("sales.csv")
        schema = engine.get_schema("sales")
        assert "order_id" in schema
        assert "total_revenue" in schema

    def test_multi_csv_join_works(self):
        from tools.nl2sql_engine import NL2SQLEngine
        engine = NL2SQLEngine()
        engine.load_multiple_csvs(["sales.csv", "products.csv"])
        result = engine.execute_query(
            "SELECT s.product, p.category FROM sales s JOIN products p ON s.product = p.name LIMIT 3"
        )
        assert result is not None

    def test_aggregate_query(self):
        from tools.nl2sql_engine import NL2SQLEngine
        engine = NL2SQLEngine()
        engine.load_csv("sales.csv")
        result = engine.execute_query(
            "SELECT quarter, SUM(total_revenue) as total FROM sales GROUP BY quarter ORDER BY quarter"
        )
        assert len(result) == 4  # Q1, Q2, Q3, Q4
