from slowapi import Limiter
from slowapi.util import get_remote_address
import os

# Redis URL from .env — shared across all server instances
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# limiter uses Redis to count requests per user IP
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=REDIS_URL,
    default_limits=["60/minute"],
)


def get_chat_limit() -> str:
    """
    Returns the rate limit for the /chat endpoint.
    Kept as a function so it can be overridden per user tier later.
    Example:
        Free tier  → 20/minute
        Pro tier   → 60/minute
        Enterprise → 200/minute
    """
    return os.getenv("CHAT_RATE_LIMIT", "60/minute")
