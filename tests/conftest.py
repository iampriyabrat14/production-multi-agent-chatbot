import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Set test env vars BEFORE any project imports
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["PINECONE_API_KEY"] = "test-pinecone-key"
os.environ["LANGCHAIN_API_KEY"] = "test-langsmith-key"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-minimum-32-characters-long"
os.environ["POSTGRES_URL"] = "postgresql://chatbot_user:chatbot_pass@localhost:5432/chatbot"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["CSV_DIR"] = "data/csv_files"
os.environ["APP_ENV"] = "test"

# ── Mock all DB + external connections globally for unit tests ────────────────
# Prevents PostgreSQL / Redis / Pinecone connections during unit tests

_mock_session = MagicMock()
_mock_engine = MagicMock()
_mock_session_maker = MagicMock(return_value=_mock_session)

# Patch SQLAlchemy engine creation so no real DB connection is attempted
patch("sqlalchemy.create_engine", return_value=_mock_engine).start()
patch("sqlalchemy.orm.sessionmaker", return_value=_mock_session_maker).start()

# Patch ChatOpenAI so no API key validation happens at import time
_mock_llm_instance = MagicMock()
_mock_llm_instance.invoke.return_value = MagicMock(content="mocked response")
patch("langchain_openai.ChatOpenAI", return_value=_mock_llm_instance).start()

# Patch memory modules that connect to DB at import time
patch("memory.long_term.LongTermMemory.save_fact", return_value=None).start()
patch("memory.long_term.LongTermMemory.get_facts", return_value=[]).start()
patch("memory.conversation_history.ConversationHistory.save_message", return_value=None).start()
patch("memory.conversation_history.ConversationHistory.get_history", return_value=[]).start()

# Patch Pinecone so it never dials out
patch("pinecone.Pinecone", return_value=MagicMock()).start()



@pytest.fixture(scope="session")
def test_client():
    from api.main import app
    return TestClient(app)


@pytest.fixture
def auth_token(test_client):
    response = test_client.post("/token", data={"username": "testuser", "password": "testpass"})
    if response.status_code == 200:
        return response.json()["access_token"]
    # fallback: create token directly
    from security.auth import create_token
    return create_token({"sub": "testuser"})


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock:
        instance = MagicMock()
        instance.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Mocked LLM response"))],
            usage=MagicMock(total_tokens=50)
        )
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_pinecone():
    with patch("pinecone.Pinecone") as mock:
        instance = MagicMock()
        index = MagicMock()
        index.query.return_value = MagicMock(matches=[
            MagicMock(score=0.92, metadata={"text": "Refunds are processed within 7 days.", "source": "policy.pdf"}),
            MagicMock(score=0.85, metadata={"text": "Shipping takes 3-5 business days.", "source": "policy.pdf"}),
        ])
        instance.Index.return_value = index
        mock.return_value = instance
        yield index


@pytest.fixture
def sample_user_id():
    return "test_user_123"


@pytest.fixture
def sample_session_id():
    return "test_session_abc"
