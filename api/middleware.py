from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from langchain_openai import ChatOpenAI
import tiktoken
import time
import logging

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# GPT-4o context window = 128,000 tokens
# We use 75% as safe limit — leaves room for response + system prompt
MODEL_MAX_TOKENS = 128_000
SAFE_LIMIT = int(MODEL_MAX_TOKENS * 0.75)   # 96,000 tokens
KEEP_LAST_N = 10                             # always keep last N messages intact


# ── Token Counting ─────────────────────────────────────────────────────────────

def count_tokens(messages: list[dict]) -> int:
    """
    Count tokens in a list of messages using tiktoken.
    tiktoken is OpenAI's official tokenizer — same one GPT-4o uses.

    Args:
      messages: list of {"role": "human"/"ai", "content": "..."}

    Returns total token count.
    """
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        # 4 tokens per message overhead (OpenAI format)
        total += 4
        total += len(enc.encode(str(msg.get("content", ""))))
        total += len(enc.encode(str(msg.get("role", ""))))

    return total


# ── Summarization ──────────────────────────────────────────────────────────────

def summarize_old_messages(messages: list[dict]) -> str:
    """
    LLM summarizes a list of old messages into one short paragraph.
    Called when conversation history exceeds the safe token limit.

    Returns a summary string that replaces the old messages.
    Preserves key facts, decisions, and context from old messages.
    """
    conversation_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )

    prompt = f"""Summarize this conversation history into one concise paragraph.
Preserve:
  - Key facts mentioned by the user
  - Important decisions or preferences
  - Topics discussed
  - Any specific numbers or data points

Conversation to summarize:
{conversation_text}

Write a 3-5 sentence summary. Be factual and concise."""

    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content


# ── Context Window Manager ─────────────────────────────────────────────────────

def manage_context_window(messages: list[dict]) -> list[dict]:
    """
    Main context window management function.
    Called before every LLM interaction to prevent token limit errors.

    Strategy:
      1. Count tokens in full history
      2. If under safe limit → return as-is
      3. If over safe limit:
         a. Keep last KEEP_LAST_N messages intact
         b. Summarize everything older
         c. Replace old messages with summary message
         d. Return trimmed history

    Example:
      Input:  100 messages (150,000 tokens) — over limit
      Output: 1 summary message + last 10 messages (~5,000 tokens) — safe
    """
    token_count = count_tokens(messages)

    # under safe limit — no action needed
    if token_count <= SAFE_LIMIT:
        return messages

    logger.warning(
        f"Context window at {token_count} tokens (limit: {SAFE_LIMIT}). "
        f"Summarizing old messages."
    )

    # split messages — old ones to summarize, recent ones to keep
    if len(messages) <= KEEP_LAST_N:
        # not enough messages to split — return as-is
        return messages

    old_messages = messages[:-KEEP_LAST_N]
    recent_messages = messages[-KEEP_LAST_N:]

    # summarize old messages
    summary_text = summarize_old_messages(old_messages)

    # replace old messages with a single system summary message
    summary_message = {
        "role": "system",
        "content": f"[Earlier conversation summary]: {summary_text}",
    }

    trimmed = [summary_message] + recent_messages

    new_token_count = count_tokens(trimmed)
    logger.info(
        f"Context reduced from {token_count} to {new_token_count} tokens "
        f"({len(old_messages)} messages summarized)."
    )

    return trimmed


# ── Logging Middleware ─────────────────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that logs every request.
    Runs automatically on every HTTP request before/after route handler.

    Logs:
      - HTTP method + path
      - Status code
      - Latency in ms
      - User agent

    Example log:
      POST /chat → 200 OK in 1243ms
      POST /chat → 429 Too Many Requests in 2ms
    """

    async def dispatch(self, request: Request, call_next):
        start = time.time()

        response = await call_next(request)

        latency_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            f"{request.method} {request.url.path} → "
            f"{response.status_code} in {latency_ms}ms | "
            f"UA: {request.headers.get('user-agent', 'unknown')[:50]}"
        )

        # attach latency to response headers for debugging
        response.headers["X-Latency-Ms"] = str(latency_ms)

        return response
