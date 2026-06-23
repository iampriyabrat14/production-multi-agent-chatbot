# Author: Priyabrat Dalbehera | github.com/iampriyabrat14 | Production Multi-Agent Chatbot
from dotenv import load_dotenv
load_dotenv()  # must be first — loads .env before any agent/LLM imports

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import time

from api.schemas import (
    ChatRequest,
    ChatResponse,
    TokenRequest,
    TokenResponse,
    HealthResponse,
)
from security.auth import create_token, get_current_user
from security.guardrails import check_input_safety, anonymize_pii_input, anonymize_pii_output
from security.rate_limiter import limiter
from agents.supervisor import run_supervisor
from api.middleware import LoggingMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once on startup — connect DB, load models, warm cache
    print("Production chatbot starting...")
    yield
    # runs once on shutdown — close connections cleanly
    print("Production chatbot shutting down...")


app = FastAPI(
    title="Production Multi-Agent Chatbot",
    description="LangGraph multi-agent chatbot with RAG, NL2SQL, memory and real-time tools",
    version="1.0.0",
    lifespan=lifespan,
)

# rate limiter wired to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# log every request with method, path, status, latency
app.add_middleware(LoggingMiddleware)

# allow frontend (Streamlit) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check if the API is running."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@app.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
async def login(data: TokenRequest):
    """Exchange username + password for a JWT token."""
    token = create_token(data.username)
    return TokenResponse(access_token=token)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
@limiter.limit("60/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: str = Depends(get_current_user),
):
    """
    Main chat endpoint — full production pipeline:
    1. Rate limit check (decorator)
    2. JWT auth check (Depends)
    3. Input guardrails — injection + PII
    4. Supervisor routes to correct agent(s)
    5. Output guardrails — PII scrub
    6. Return clean response
    """
    start = time.time()

    # step 1 — block prompt injection + jailbreak attempts
    check_input_safety(body.message)

    # step 2 — anonymize PII in user message before LLM sees it
    clean_message = anonymize_pii_input(body.message)

    # step 3 — run supervisor (routes to RAG / SQL / Realtime / Memory agents)
    result = run_supervisor(
        question=clean_message,
        user_id=body.user_id,
        session_id=body.session_id,
    )

    # step 4 — scrub PII from LLM response before sending to user
    clean_answer = anonymize_pii_output(result["answer"])

    # step 5 — determine which agent(s) were used
    agent_used = ", ".join(result.get("plan", ["supervisor"]))

    return ChatResponse(
        answer=clean_answer,
        agent_used=agent_used,
        session_id=body.session_id,
        sources=result.get("sources", []),
        tokens_used=result.get("tokens_used", 0),
        latency_ms=round((time.time() - start) * 1000, 2),
        timestamp=datetime.utcnow(),
    )
