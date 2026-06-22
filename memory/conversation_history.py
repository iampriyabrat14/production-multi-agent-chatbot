from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from uuid import uuid4
import os

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://localhost/chatbot")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ── Database Table ─────────────────────────────────────────────────────────────

class ConversationMessage(Base):
    """
    Stores every single message ever sent by a user.
    This is the full chat log — like WhatsApp history.

    Columns:
      id         → unique message ID
      user_id    → which user sent/received this
      session_id → which session this belongs to
      role       → "human" or "assistant"
      content    → the actual message text
      timestamp  → when it was sent
    """
    __tablename__ = "conversation_history"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ── Conversation History Class ─────────────────────────────────────────────────

class ConversationHistory:
    """
    Saves and retrieves full conversation history from PostgreSQL.
    Every message is stored — nothing is discarded.
    """

    def save_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        """
        Save one message to the history table.
        Called after every human message and every AI response.
        """
        with SessionLocal() as db:
            message = ConversationMessage(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                timestamp=datetime.utcnow(),
            )
            db.add(message)
            db.commit()

    def get_history(self, user_id: str, last_n: int = 10) -> list[dict]:
        """
        Get the last N messages across all sessions for this user.
        Used to inject past context into a new session.
        Returns list of {"role": ..., "content": ...} dicts.
        """
        with SessionLocal() as db:
            messages = (
                db.query(ConversationMessage)
                .filter_by(user_id=user_id)
                .order_by(ConversationMessage.timestamp.desc())
                .limit(last_n)
                .all()
            )
            # reverse so oldest is first (chronological order)
            return [
                {"role": m.role, "content": m.content, "timestamp": str(m.timestamp)}
                for m in reversed(messages)
            ]

    def get_session_history(self, session_id: str) -> list[dict]:
        """
        Get all messages from one specific session.
        Used for session replay or debugging.
        """
        with SessionLocal() as db:
            messages = (
                db.query(ConversationMessage)
                .filter_by(session_id=session_id)
                .order_by(ConversationMessage.timestamp.asc())
                .all()
            )
            return [
                {"role": m.role, "content": m.content, "timestamp": str(m.timestamp)}
                for m in messages
            ]

    def format_for_context(self, user_id: str, last_n: int = 10) -> str:
        """
        Format last N messages as a string to inject into short-term memory.
        Example output:
          "Previous conversation:
           human: What is EUR/USD rate?
           assistant: The EUR/USD rate is 1.085"
        """
        messages = self.get_history(user_id, last_n)
        if not messages:
            return "No previous conversation history."
        lines = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return f"Previous conversation:\n{lines}"

    def count_messages(self, user_id: str) -> int:
        """Return total number of messages ever sent by this user."""
        with SessionLocal() as db:
            return db.query(ConversationMessage).filter_by(user_id=user_id).count()
