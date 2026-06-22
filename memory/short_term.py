from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Union

# type alias for all message types
Message = Union[HumanMessage, AIMessage, SystemMessage]


class ShortTermMemory:
    """
    Holds the current session's conversation messages in memory.
    Lives only for the duration of the session — cleared when session ends.
    Passed directly into LangGraph state on every agent call.
    """

    def __init__(self, max_messages: int = 20):
        # max_messages prevents context window overflow
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the session.
        role: "human" | "ai" | "system"
        """
        if role == "human":
            self._messages.append(HumanMessage(content=content))
        elif role == "ai":
            self._messages.append(AIMessage(content=content))
        elif role == "system":
            self._messages.append(SystemMessage(content=content))

        # keep only the last max_messages to avoid context overflow
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def get_messages(self) -> list[Message]:
        """Return all current session messages for LangGraph state."""
        return self._messages

    def get_last_n(self, n: int) -> list[Message]:
        """Return only the last N messages."""
        return self._messages[-n:]

    def clear(self) -> None:
        """Clear all messages — called when session ends."""
        self._messages = []

    def count(self) -> int:
        """Return number of messages in current session."""
        return len(self._messages)
