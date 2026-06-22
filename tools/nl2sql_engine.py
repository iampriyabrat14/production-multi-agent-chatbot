import duckdb
import pandas as pd
from openai import OpenAI
from pathlib import Path
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_DIR = Path(os.getenv("CSV_DIR", "data/csv_files"))


class NL2SQLEngine:
    """
    Converts natural language questions into SQL queries
    and executes them on CSV files using DuckDB.

    Flow:
      1. Load CSV into DuckDB as an in-memory table
      2. Extract schema (columns + sample rows) for LLM context
      3. LLM generates SQL from question + schema
      4. Reflect on generated SQL (validate before running)
      5. Execute SQL on DuckDB
      6. Return result as readable text
    """

    def __init__(self):
        # in-memory DuckDB connection — fast, no setup needed
        self.conn = duckdb.connect(database=":memory:")
        self.loaded_tables: dict[str, str] = {}  # table_name → csv_path

    def load_csv(self, csv_filename: str) -> str:
        """
        Load a CSV file into DuckDB as a table.
        Table name = filename without extension.
        Returns table name.
        """
        csv_path = CSV_DIR / csv_filename
        table_name = Path(csv_filename).stem.replace("-", "_").replace(" ", "_")

        if table_name not in self.loaded_tables:
            # read via pandas first — avoids file lock errors when Excel has the file open
            df = pd.read_csv(csv_path)
            self.conn.register(table_name, df)
            self.loaded_tables[table_name] = str(csv_path)

        return table_name

    def load_multiple_csvs(self, csv_filenames: list[str]) -> list[str]:
        """
        Load multiple CSV files into DuckDB as separate tables.
        Returns list of table names.
        Used when a question needs JOIN across multiple CSVs.

        Example:
          load_multiple_csvs(["sales.csv", "customers.csv", "products.csv"])
          → loads 3 tables: sales, customers, products
          → LLM can now JOIN across all 3
        """
        return [self.load_csv(filename) for filename in csv_filenames]

    def get_schema(self, table_name: str, sample_rows: int = 3) -> str:
        """
        Get column names, types, and sample rows for LLM context.
        This tells the LLM what the table looks like before generating SQL.

        Example output:
          Table: sales
          Columns: quarter (VARCHAR), revenue (DOUBLE), region (VARCHAR)
          Sample rows:
            Q1 | 1200000.0 | North
            Q2 | 1450000.0 | South
            Q3 | 2400000.0 | North
        """
        # get column info
        columns = self.conn.execute(f"DESCRIBE {table_name}").fetchdf()
        col_info = ", ".join(f"{row['column_name']} ({row['column_type']})"
                             for _, row in columns.iterrows())

        # get sample rows
        sample = self.conn.execute(
            f"SELECT * FROM {table_name} LIMIT {sample_rows}"
        ).fetchdf()
        sample_str = sample.to_string(index=False)

        return (
            f"Table: {table_name}\n"
            f"Columns: {col_info}\n"
            f"Sample rows:\n{sample_str}"
        )

    def get_multi_schema(self, table_names: list[str], sample_rows: int = 3) -> str:
        """
        Get combined schema for multiple tables — used for JOIN queries.
        Gives LLM full picture of all tables before generating JOIN SQL.

        Example output:
          Table: sales
          Columns: id (VARCHAR), customer_id (VARCHAR), revenue (DOUBLE)
          Sample rows: ...

          Table: customers
          Columns: id (VARCHAR), name (VARCHAR), region (VARCHAR)
          Sample rows: ...
        """
        return "\n\n".join(
            self.get_schema(table_name, sample_rows)
            for table_name in table_names
        )

    def generate_sql(self, question: str, schema: str) -> str:
        """
        Ask LLM to generate a SQL query from a natural language question.
        Schema is injected so LLM knows exact column names and types.
        """
        prompt = f"""You are a SQL expert. Generate a single valid DuckDB SQL query.

Table schema:
{schema}

User question: {question}

Rules:
- Return ONLY the SQL query, nothing else
- No markdown, no explanation, no backticks
- Use exact column names from the schema
- For aggregations use proper GROUP BY
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,      # deterministic SQL generation
        )
        return response.choices[0].message.content.strip()

    def reflect_on_sql(self, sql: str, schema: str) -> str:
        """
        Reflection step — LLM reviews its own generated SQL before execution.
        Catches issues like wrong column names, missing GROUP BY, syntax errors.
        Returns corrected SQL or the same SQL if it's valid.
        """
        prompt = f"""Review this SQL query for correctness against the schema.

Schema:
{schema}

SQL query:
{sql}

If the SQL is correct, return it unchanged.
If there are issues, return the corrected SQL only.
Return ONLY the SQL query, nothing else.
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL on DuckDB and return result as a DataFrame."""
        return self.conn.execute(sql).fetchdf()

    def query(self, question: str, csv_files: str | list[str]) -> str:
        """
        Full NL2SQL pipeline — question in, answer out.
        Supports single CSV or multiple CSVs (for JOINs).

        Single table:
          engine.query("Total Q3 revenue?", "sales.csv")

        Multiple tables (JOIN):
          engine.query(
            "Show sales with customer names",
            ["sales.csv", "customers.csv"]
          )

        Steps:
          1. Load CSV(s) into DuckDB
          2. Get schema(s) for LLM context
          3. Generate SQL
          4. Reflect + correct SQL
          5. Execute
          6. Format result
        """
        # step 1 — load one or multiple CSVs
        if isinstance(csv_files, str):
            table_names = [self.load_csv(csv_files)]
        else:
            table_names = self.load_multiple_csvs(csv_files)

        # step 2 — get schema for all loaded tables
        schema = (
            self.get_schema(table_names[0])
            if len(table_names) == 1
            else self.get_multi_schema(table_names)
        )

        # step 3 — generate SQL from question + schema
        sql = self.generate_sql(question, schema)

        # step 4 — reflect and correct if needed
        sql = self.reflect_on_sql(sql, schema)

        # step 5 — execute on DuckDB
        result_df = self.execute_query(sql)

        # step 6 — format as readable string
        if result_df.empty:
            return "No results found for your query."

        return (
            f"Query: {sql}\n\n"
            f"Result:\n{result_df.to_string(index=False)}"
        )
