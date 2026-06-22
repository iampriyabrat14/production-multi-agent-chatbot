import streamlit as st


# agent name → emoji + label
AGENT_LABELS = {
    "rag":       ("📚", "RAG Agent",      "Searched knowledge base documents"),
    "sql":       ("🗃️",  "SQL Agent",      "Queried internal CSV data"),
    "realtime":  ("💱",  "Realtime Agent", "Fetched live external data"),
    "memory":    ("🧬",  "Memory Agent",   "Retrieved user memory and facts"),
    "supervisor":("🧠",  "Supervisor",     "Planned and routed the request"),
    "error":     ("❌",  "Error",          "An error occurred"),
}


def render_agent_steps(response: dict) -> None:
    """
    Renders an expandable section showing which agents ran and what they did.
    Collapsed by default so it does not clutter the chat.

    Shows:
      - Which agents were called (from plan)
      - What each agent returned
      - Token usage + latency
      - Source documents used

    Args:
      response: ChatResponse dict from FastAPI
        {
          "answer": "...",
          "agent_used": "rag, sql",
          "sources": [...],
          "tokens_used": 342,
          "latency_ms": 1243.5,
        }
    """
    agent_used = response.get("agent_used", "")
    sources = response.get("sources", [])
    tokens = response.get("tokens_used", 0)
    latency = response.get("latency_ms", 0)

    agents_called = [a.strip() for a in agent_used.split(",") if a.strip()]

    with st.expander("🔍 Agent reasoning steps", expanded=False):

        # ── Agents Called ──────────────────────────────────────────────────────
        st.markdown("**Agents called:**")
        for agent in agents_called:
            emoji, label, description = AGENT_LABELS.get(
                agent, ("🤖", agent.title(), "")
            )
            st.markdown(f"{emoji} **{label}** — {description}")

        # ── Agent Flow ─────────────────────────────────────────────────────────
        if len(agents_called) > 1:
            st.divider()
            st.markdown("**Execution flow:**")
            flow = " → ".join(
                AGENT_LABELS.get(a, ("🤖", a, ""))[1]
                for a in ["supervisor"] + agents_called
            )
            st.code(flow, language=None)

        # ── Sources ────────────────────────────────────────────────────────────
        if sources and st.session_state.get("show_sources", True):
            st.divider()
            st.markdown("**Source documents used:**")
            for source in sources:
                st.markdown(f"📄 `{source}`")

        # ── Performance Metrics ────────────────────────────────────────────────
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            latency_color = (
                "green" if latency < 2000
                else "orange" if latency < 5000
                else "red"
            )
            st.markdown(
                f"⚡ **Latency:** "
                f":{latency_color}[{latency:.0f}ms]"
            )

        with col2:
            if st.session_state.get("show_tokens", False):
                st.markdown(f"🪙 **Tokens:** {tokens:,}")


def render_metadata_badge(response: dict) -> None:
    """
    Renders a small inline badge under the assistant message showing
    agent used + latency. More compact than the full expander.

    Example:
      🧠 rag, sql  •  1243ms  •  342 tokens
    """
    agent_used = response.get("agent_used", "")
    latency = response.get("latency_ms", 0)
    tokens = response.get("tokens_used", 0)

    parts = [f"🧠 `{agent_used}`", f"⚡ `{latency:.0f}ms`"]

    if st.session_state.get("show_tokens", False) and tokens:
        parts.append(f"🪙 `{tokens:,} tokens`")

    st.caption("  •  ".join(parts))
