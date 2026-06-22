from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://localhost/chatbot")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ── Database Table ─────────────────────────────────────────────────────────────

class UserFact(Base):
    """
    Stores key facts about a user extracted from conversations.
    Example rows:
      user_id=user_123  key=preferred_currency   value=USD
      user_id=user_123  key=company              value=Acme Corp
      user_id=user_123  key=role                 value=Finance Manager
    """
    __tablename__ = "user_facts"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    key = Column(String(255), nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ── Long-term Memory Class ─────────────────────────────────────────────────────

class LongTermMemory:
    """
    Reads and writes user facts to PostgreSQL.
    Facts persist forever across all sessions.
    """

    def save_fact(self, user_id: str, key: str, value: str) -> None:
        """
        Save or update a fact about the user.
        If the key already exists for this user, it gets updated.
        """
        with SessionLocal() as db:
            existing = db.query(UserFact).filter_by(user_id=user_id, key=key).first()
            if existing:
                existing.value = value
                existing.updated_at = datetime.utcnow()
            else:
                fact = UserFact(
                    id=f"{user_id}_{key}",
                    user_id=user_id,
                    key=key,
                    value=value,
                )
                db.add(fact)
            db.commit()

    def get_facts(self, user_id: str) -> dict[str, str]:
        """
        Retrieve all facts for a user as a key-value dictionary.
        Returns: {"preferred_currency": "USD", "company": "Acme Corp"}
        """
        with SessionLocal() as db:
            facts = db.query(UserFact).filter_by(user_id=user_id).all()
            return {f.key: f.value for f in facts}

    def get_fact(self, user_id: str, key: str) -> str | None:
        """Retrieve a single fact by key."""
        with SessionLocal() as db:
            fact = db.query(UserFact).filter_by(user_id=user_id, key=key).first()
            return fact.value if fact else None

    def delete_fact(self, user_id: str, key: str) -> None:
        """Delete an outdated or incorrect fact."""
        with SessionLocal() as db:
            db.query(UserFact).filter_by(user_id=user_id, key=key).delete()
            db.commit()

    def format_for_prompt(self, user_id: str) -> str:
        """
        Format all user facts as a string to inject into agent system prompt.
        Example output:
          "Known facts about this user:
           - preferred_currency: USD
           - company: Acme Corp"
        """
        facts = self.get_facts(user_id)
        if not facts:
            return "No known facts about this user yet."
        lines = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
        return f"Known facts about this user:\n{lines}"
