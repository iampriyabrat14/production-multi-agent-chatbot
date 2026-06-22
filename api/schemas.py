from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="Authenticated user ID")


class ChatResponse(BaseModel):
    answer: str
    agent_used: str
    session_id: str
    sources: Optional[list[str]] = []
    tokens_used: int
    latency_ms: float
    timestamp: datetime


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
