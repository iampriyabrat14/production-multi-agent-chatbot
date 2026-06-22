import streamlit as st
from ui.components.agent_steps import render_agent_steps, render_metadata_badge


def render_chat_history() -> None:
    """
    Renders all past messages in the current session.
    Called on every Streamlit rerun to display full chat history.

    Each message pair shows:
      - Human message (right-aligned, user avatar)
      - Assistant response (left-aligned, bot avatar)
      - Agent steps expander (if enabled in settings)
      - Metadata badge (agent used, latency)
    """
    messages = st.session_state.get("messages", [])

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        response = msg.get("response", {})

        if role == "human":
            with st.chat_message("user"):
                st.markdown(content)

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

                # show agent steps if enabled
                if response and st.session_state.get("show_agent_steps", True):
                    render_agent_steps(response)
                else:
                    # compact badge only
                    if response:
                        render_metadata_badge(response)


def render_chat_input(session_id: str) -> None:
    """
    Renders the chat input box at the bottom of the screen.
    Handles message submission, API call, and state update.

    Flow:
      User types message → hits Enter or Send
      → add human message to session state
      → call FastAPI /chat
      → add assistant response to session state
      → rerun to display new messages
    """
    from ui.utils.api_client import send_message

    user_input = st.chat_input("Ask anything — documents, CSV data, live rates...")

    if user_input:
        # immediately show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # add to session state
        st.session_state.setdefault("messages", []).append({
            "role": "human",
            "content": user_input,
        })

        # call API and show spinner while waiting
        with st.chat_message("assistant"):
            with st.spinner("Agents working..."):
                response = send_message(user_input, session_id)

            answer = response.get("answer", "Sorry, something went wrong.")
            st.markdown(answer)

            # show agent steps if enabled
            if st.session_state.get("show_agent_steps", True):
                render_agent_steps(response)
            else:
                render_metadata_badge(response)

        # save assistant response to session state
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "response": response,
        })

        # update memory facts in sidebar from response metadata
        _update_memory_display(response)


def render_welcome_message() -> None:
    """
    Renders a welcome message when chat is empty.
    Shows example questions to help user get started.
    """
    st.markdown("### 👋 Welcome! What would you like to know?")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📚 From documents:**")
        st.caption("• What is our refund policy?")
        st.caption("• Summarize the product manual")
        st.caption("• What are the compliance rules?")

        st.markdown("**💱 Live data:**")
        st.caption("• What is EUR/USD rate right now?")
        st.caption("• Convert 500 GBP to INR")

    with col2:
        st.markdown("**🗃️ From your CSV data:**")
        st.caption("• What was Q3 total revenue?")
        st.caption("• Show sales by region")
        st.caption("• Which product sold most in 2024?")

        st.markdown("**🧬 Memory:**")
        st.caption("• What do you remember about me?")
        st.caption("• My preferred currency is USD")


def _update_memory_display(response: dict) -> None:
    """
    Updates the memory facts displayed in the sidebar
    after each response, if new facts were saved.
    """
    agent_used = response.get("agent_used", "")
    if "memory" in agent_used:
        # trigger sidebar memory refresh on next rerun
        st.session_state["memory_refresh"] = True
