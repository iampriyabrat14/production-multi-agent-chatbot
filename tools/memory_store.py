from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.conversation_history import ConversationHistory
from langchain_core.messages import HumanMessage, AIMessage


class MemoryStore:
    """
    Single interface for all memory operations.
    Agents interact with this class only — not with individual memory classes.

    Internally manages:
      - ShortTermMemory     → current session messages (in RAM)
      - LongTermMemory      → user facts (PostgreSQL, forever)
      - ConversationHistory → all messages ever (PostgreSQL, forever)

    Usage:
      store = MemoryStore(user_id="user_123", session_id="sess_001")
      store.load_session()           # load past context on new session
      store.add_message("human", "Hello")
      store.add_message("ai", "Hi there!")
      context = store.get_context()  # pass to agent system prompt
    """

    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id

        # initialize all 3 memory layers
        self.short_term = ShortTermMemory(max_messages=20)
        self.long_term = LongTermMemory()
        self.history = ConversationHistory()

    def load_session(self, last_n: int = 10) -> None:
        """
        Called at the start of every new session.
        Loads last N messages from history into short-term memory
        so agent has context from previous conversations.
        """
        past_messages = self.history.get_history(self.user_id, last_n)
        for msg in past_messages:
            self.short_term.add_message(msg["role"], msg["content"])

    def add_message(self, role: str, content: str) -> None:
        """
        Save a message to both short-term memory and conversation history.
        Called after every human message and every AI response.

        role: "human" | "ai"
        """
        # save to in-session short-term memory
        self.short_term.add_message(role, content)

        # save permanently to PostgreSQL history
        self.history.save_message(
            user_id=self.user_id,
            session_id=self.session_id,
            role=role,
            content=content,
        )

    def get_context(self) -> str:
        """
        Returns full context string to inject into agent system prompt.
        Combines:
          - Long-term user facts
          - Recent conversation history

        Example output:
          "Known facts about this user:
             - preferred_currency: USD
             - company: Acme Corp

           Previous conversation:
             human: What is EUR/USD?
             assistant: EUR/USD is 1.085"
        """
        facts = self.long_term.format_for_prompt(self.user_id)
        history = self.history.format_for_context(self.user_id, last_n=10)
        return f"{facts}\n\n{history}"

    def get_messages(self) -> list[HumanMessage | AIMessage]:
        """
        Returns current session messages for LangGraph state.
        Passed directly into the LangGraph graph as message history.
        """
        return self.short_term.get_messages()

    def save_fact(self, key: str, value: str) -> None:
        """
        Save a key fact about the user to long-term memory.
        Called by Memory Agent when it detects important user information.

        Example:
          store.save_fact("preferred_currency", "USD")
          store.save_fact("company", "Acme Corp")
        """
        self.long_term.save_fact(self.user_id, key, value)

    def get_facts(self) -> dict[str, str]:
        """Return all known facts about the user."""
        return self.long_term.get_facts(self.user_id)

    def clear_session(self) -> None:
        """
        Clear short-term memory when user logs out.
        History and long-term facts are NOT cleared — they persist forever.
        """
        self.short_term.clear()
