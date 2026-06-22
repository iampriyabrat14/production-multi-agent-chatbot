import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv

from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat_history, render_chat_input, render_welcome_message

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Agent Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Hide Streamlit default header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Chat container max width */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Assistant message styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 8px;
        margin-bottom: 4px;
    }

    /* Input box styling */
    [data-testid="stChatInput"] {
        border-radius: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────

def init_session_state() -> None:
    """
    Initialize all session state variables on first load.
    Streamlit reruns entire script on every interaction —
    session_state persists values across reruns.
    """
    defaults = {
        "messages": [],
        "token": None,
        "username": None,
        "session_id": str(uuid4()),
        "memory_facts": {},
        "show_agent_steps": True,
        "show_sources": True,
        "show_tokens": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── Main App ───────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main Streamlit app entry point.

    Layout:
      Left sidebar  → login, memory facts, session, settings
      Main area     → chat history + input box

    Flow per rerun:
      1. init session state
      2. render sidebar → get config (logged_in, session_id, settings)
      3. if not logged in → show login prompt in main area
      4. if logged in → render chat
    """
    init_session_state()

    # render sidebar — returns config with login status + settings
    config = render_sidebar()

    # ── Main Area ──────────────────────────────────────────────────────────────

    if not config.get("logged_in", False):
        # not logged in — show prompt
        st.title("🤖 Production Multi-Agent Chatbot")
        st.info("👈 Please log in from the sidebar to start chatting.")

        st.markdown("### What this chatbot can do:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - 📚 **RAG** — answers from your documents
            - 🗃️ **NL2SQL** — queries your CSV data
            """)
        with col2:
            st.markdown("""
            - 💱 **Realtime** — live currency rates
            - 🧬 **Memory** — remembers you across sessions
            """)
        return

    # logged in — render chat
    session_id = config.get("session_id", st.session_state["session_id"])

    st.title("🤖 Multi-Agent Chatbot")
    st.caption(
        f"Session `{session_id[:8]}...` • "
        f"Agents: RAG · SQL · Realtime · Memory"
    )
    st.divider()

    # show welcome message if no messages yet
    messages = st.session_state.get("messages", [])
    if not messages:
        render_welcome_message()

    # render all past messages
    render_chat_history()

    # render input box — handles send + response
    render_chat_input(session_id)


if __name__ == "__main__":
    main()
