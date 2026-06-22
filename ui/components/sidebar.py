import streamlit as st
from ui.utils.api_client import check_health, is_logged_in
from uuid import uuid4


def render_sidebar() -> dict:
    """
    Renders the left sidebar panel.
    Returns sidebar config dict used by main app.

    Sidebar sections:
      1. API connection status
      2. Login form (if not logged in)
      3. User info + logout (if logged in)
      4. Memory facts panel
      5. Session management
      6. Settings
    """
    config = {}

    with st.sidebar:
        st.title("🤖 Multi-Agent Chatbot")
        st.divider()

        # ── Section 1: API Status ──────────────────────────────────────────────
        api_healthy = check_health()
        if api_healthy:
            st.success("🟢 API Connected", icon="✅")
        else:
            st.error("🔴 API Offline", icon="❌")
            st.caption("Make sure FastAPI is running on port 8000")

        st.divider()

        # ── Section 2/3: Auth ──────────────────────────────────────────────────
        if not is_logged_in():
            config = _render_login_form(config)
        else:
            config = _render_user_panel(config)

    return config


def _render_login_form(config: dict) -> dict:
    """Render login form when user is not authenticated."""
    st.subheader("🔑 Login")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        from ui.utils.api_client import login
        token = login(username, password)
        if token:
            st.session_state["session_id"] = str(uuid4())
            st.success("Logged in!")
            st.rerun()
        else:
            st.error("Invalid credentials")

    config["logged_in"] = False
    return config


def _render_user_panel(config: dict) -> dict:
    """Render user info, memory facts, and settings when logged in."""
    username = st.session_state.get("username", "user")

    # ── User Info ──────────────────────────────────────────────────────────────
    st.subheader(f"👤 {username}")
    if st.button("Logout", use_container_width=True):
        for key in ["token", "username", "session_id", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()

    # ── Memory Facts ───────────────────────────────────────────────────────────
    st.subheader("🧠 Memory")
    st.caption("Facts the agent remembers about you")

    facts = st.session_state.get("memory_facts", {})
    if facts:
        for key, value in facts.items():
            st.markdown(f"• **{key}**: {value}")
    else:
        st.caption("No facts saved yet. Start chatting!")

    st.divider()

    # ── Session Management ─────────────────────────────────────────────────────
    st.subheader("💬 Session")

    session_id = st.session_state.get("session_id", str(uuid4()))
    st.caption(f"Session: `{session_id[:8]}...`")

    if st.button("New Session", use_container_width=True):
        st.session_state["session_id"] = str(uuid4())
        st.session_state["messages"] = []
        st.rerun()

    msg_count = len(st.session_state.get("messages", []))
    st.caption(f"Messages this session: {msg_count}")

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")

    show_agent_steps = st.toggle(
        "Show agent reasoning steps",
        value=st.session_state.get("show_agent_steps", True),
    )
    st.session_state["show_agent_steps"] = show_agent_steps

    show_sources = st.toggle(
        "Show source documents",
        value=st.session_state.get("show_sources", True),
    )
    st.session_state["show_sources"] = show_sources

    show_tokens = st.toggle(
        "Show token usage",
        value=st.session_state.get("show_tokens", False),
    )
    st.session_state["show_tokens"] = show_tokens

    config["logged_in"] = True
    config["show_agent_steps"] = show_agent_steps
    config["show_sources"] = show_sources
    config["show_tokens"] = show_tokens
    config["session_id"] = session_id

    return config
