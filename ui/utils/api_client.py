import httpx
import streamlit as st
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── Auth ───────────────────────────────────────────────────────────────────────

def login(username: str, password: str) -> str | None:
    """
    Call POST /auth/token and return JWT token.
    Stores token in Streamlit session state on success.
    Returns token string or None if login failed.
    """
    try:
        response = httpx.post(
            f"{API_BASE_URL}/auth/token",
            json={"username": username, "password": password},
            timeout=10.0,
        )
        if response.status_code == 200:
            token = response.json()["access_token"]
            st.session_state["token"] = token
            st.session_state["username"] = username
            return token
        return None
    except httpx.RequestError:
        return None


def get_token() -> str | None:
    """Return stored JWT token from session state."""
    return st.session_state.get("token")


def is_logged_in() -> bool:
    """Check if user has a valid token in session."""
    return bool(get_token())


# ── Chat ───────────────────────────────────────────────────────────────────────

def send_message(message: str, session_id: str) -> dict:
    """
    Call POST /chat with user message.
    Returns full ChatResponse dict including answer, agent_used, sources.

    Returns:
      {
        "answer": "...",
        "agent_used": "rag, sql",
        "session_id": "...",
        "sources": [...],
        "tokens_used": 342,
        "latency_ms": 1243.5,
        "timestamp": "..."
      }

    On error returns:
      {"answer": "Error: ...", "agent_used": "error", ...}
    """
    token = get_token()
    if not token:
        return _error_response("Not authenticated. Please log in.")

    username = st.session_state.get("username", "user")

    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": message,
                "session_id": session_id,
                "user_id": username,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0,   # agents can take time — generous timeout
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            return _error_response("Rate limit reached. Please wait a moment.")
        elif response.status_code == 401:
            st.session_state.pop("token", None)
            return _error_response("Session expired. Please log in again.")
        elif response.status_code == 400:
            detail = response.json().get("detail", "Invalid request")
            return _error_response(f"Request blocked: {detail}")
        else:
            return _error_response(f"Server error ({response.status_code})")

    except httpx.TimeoutException:
        return _error_response("Request timed out. The agents are taking too long.")
    except httpx.RequestError as e:
        return _error_response(f"Could not connect to API: {str(e)}")


# ── Health ─────────────────────────────────────────────────────────────────────

def check_health() -> bool:
    """
    Ping /health endpoint to check if API is running.
    Used by Streamlit UI to show connection status in sidebar.
    """
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        return response.status_code == 200
    except httpx.RequestError:
        return False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _error_response(message: str) -> dict:
    """Return a consistent error response dict."""
    return {
        "answer": f"⚠️ {message}",
        "agent_used": "error",
        "session_id": "",
        "sources": [],
        "tokens_used": 0,
        "latency_ms": 0,
        "timestamp": "",
    }
