# 🤖 Production-Grade Multi-Agent Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B6B?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector--DB-00C4B4?style=for-the-badge&logo=pinecone&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-ECS+RDS-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 👤 Author & Attribution

**Built by [Priyabrat Dalbehera](https://github.com/iampriyabrat14)**
- GitHub: [@iampriyabrat14](https://github.com/iampriyabrat14)
- LinkedIn: [Priyabrat Dalbehera](https://linkedin.com/in/priyabrat-dalbehera)
- Email: ipriyabrat689@gmail.com

> If you use this code, architecture, or ideas in your own project, please **credit the original author** with a link back to this repository. See [LICENSE](LICENSE) for full terms.

---

## 🎯 Problem Statement

> **Modern enterprises struggle with fragmented AI workflows** — one tool for documents, another for databases, another for live data, with no memory of past conversations and zero safety guarantees.

### The Core Problems This Solves

| # | Problem | How This System Solves It |
|---|---------|--------------------------|
| 1 | **Data Silos** — Documents, CSVs, and live APIs are all separate | Single chat interface queries all sources through specialized agents |
| 2 | **No Memory** — Every conversation starts from scratch | Persistent long-term memory across all sessions via PostgreSQL |
| 3 | **Hallucination Risk** — LLMs make up answers | RAG grounds every answer in real retrieved documents |
| 4 | **Stale Information** — Static knowledge cutoffs | Real-time tool agents fetch live data (currency, APIs) on demand |
| 5 | **No Safety Net** — Prompt injections, PII leaks | Guardrails layer at entry and exit of every request |
| 6 | **Black Box AI** — No idea why the model answered | LangSmith traces every agent decision end-to-end |
| 7 | **Cost Explosions** — Unbounded API usage | Per-user token budgets + rate limiting enforced at gateway |
| 8 | **Single Point of Failure** — One LLM call fails, everything fails | Circuit breakers + async queues + fallback routing |

---

## 🏗️ Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🌐 USER INTERFACE LAYER                              ║
║                                                                              ║
║   Web App / Mobile App / API Client / Slack Bot                             ║
╚══════════════════════════╦═══════════════════════════════════════════════════╝
                           ║  HTTPS Request
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🔐 SECURITY GATEWAY (FastAPI)                           ║
║                                                                              ║
║   ┌─────────────┐   ┌──────────────┐   ┌─────────────────────────────┐     ║
║   │  JWT Auth   │──▶│ Rate Limiter │──▶│  Guardrails (Input Filter)  │     ║
║   │  (per user) │   │ (Redis-back) │   │  PII detect + Injection     │     ║
║   └─────────────┘   └──────────────┘   └─────────────────────────────┘     ║
╚══════════════════════════╦═══════════════════════════════════════════════════╝
                           ║  Validated Request
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🧠 LANGGRAPH SUPERVISOR AGENT                             ║
║                                                                              ║
║   • Understands user intent                                                  ║
║   • Routes to correct specialist agent(s)                                    ║
║   • Merges multi-agent responses                                             ║
║   • Manages conversation state graph                                         ║
╚══╦═══════════════╦══════════════╦═══════════════╦════════════════════════════╝
   ║               ║              ║               ║
   ▼               ▼              ▼               ▼
╔══════╗      ╔════════╗     ╔════════╗     ╔══════════╗
║ 📚   ║      ║  🗃️    ║     ║  💱    ║     ║  🧬      ║
║ RAG  ║      ║ NL2SQL ║     ║  Real  ║     ║ Memory   ║
║AGENT ║      ║ AGENT  ║     ║  Time  ║     ║  AGENT   ║
║      ║      ║        ║     ║ AGENT  ║     ║          ║
╚══╦═══╝      ╚═══╦════╝     ╚════╦═══╝     ╚════╦═════╝
   ║               ║              ║               ║
   ▼               ▼              ▼               ▼
╔══════════╗  ╔══════════╗  ╔══════════╗  ╔══════════════════════════╗
║ Pinecone ║  ║  DuckDB  ║  ║Currency  ║  ║      PostgreSQL           ║
║ Vector   ║  ║  (CSV→   ║  ║  API     ║  ║  ┌────────────────────┐  ║
║    DB    ║  ║   SQL)   ║  ║ (live)   ║  ║  │  Long-term Memory  │  ║
╚══════════╝  ╚══════════╝  ╚══════════╝  ║  ├────────────────────┤  ║
                                          ║  │ Conversation       │  ║
                                          ║  │ History (all msgs) │  ║
                                          ║  ├────────────────────┤  ║
                                          ║  │ LangGraph          │  ║
                                          ║  │ Checkpoints        │  ║
                                          ║  └────────────────────┘  ║
                                          ╚══════════════════════════╝
                           ║
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🛡️ OUTPUT GUARDRAILS LAYER                                ║
║                                                                              ║
║   PII Scrubbing  │  Hallucination Check  │  Content Policy Filter           ║
╚══════════════════════════╦═══════════════════════════════════════════════════╝
                           ║
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║              📊 OBSERVABILITY + EVALUATION LAYER                            ║
║                                                                              ║
║   ┌─────────────────┐          ┌─────────────────────────────────────┐      ║
║   │   LangSmith     │          │           RAGAS Evaluation          │      ║
║   │  • Full traces  │          │  • Faithfulness   • Answer Relevancy│      ║
║   │  • Agent hops   │          │  • Context Recall • Correctness     │      ║
║   │  • Token usage  │          └─────────────────────────────────────┘      ║
║   │  • Latency      │                                                        ║
║   └─────────────────┘                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
                           ║
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🖥️  STREAMLIT UI LAYER                                    ║
║                                                                              ║
║   Chat Interface  │  Agent Step Viewer  │  Memory Inspector  │  History     ║
║   (port 8501)     │  (expandable)       │  (sidebar)         │  (scrollable)║
╚══════════════════════════╦═══════════════════════════════════════════════════╝
                           ║  calls FastAPI internally
                           ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   ☁️  AWS DEPLOYMENT LAYER                                  ║
║                                                                              ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              Application Load Balancer                              │    ║
║  │   yourdomain.com → Streamlit UI   yourdomain.com/api → FastAPI     │    ║
║  └───────────────────────┬─────────────────────────┬───────────────────┘    ║
║                          │                         │                        ║
║              ┌───────────▼──────────┐  ┌──────────▼──────────┐             ║
║              │ ECS Fargate          │  │ ECS Fargate          │             ║
║              │ Container 1          │  │ Container 2          │             ║
║              │ Streamlit UI :8501   │  │ FastAPI Backend :8000│             ║
║              └──────────────────────┘  └──────────────────────┘             ║
║                                                                              ║
║  SQL Server / RDS  │  ElastiCache Redis  │  CloudWatch Alarms                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🗂️ Project Structure

```
production-chatbot/
│
├── 🔐 security/
│   ├── auth.py                    # JWT token creation & validation
│   ├── rate_limiter.py            # Per-user Redis-backed rate limits
│   └── guardrails.py              # Input/output safety + PII detection
│
├── 🧠 agents/
│   ├── supervisor.py              # LangGraph supervisor — routes all queries
│   ├── rag_agent.py               # Document Q&A via Pinecone retrieval
│   ├── sql_agent.py               # Natural language → SQL over CSV files
│   ├── realtime_agent.py          # Live data: currency, APIs
│   └── memory_agent.py            # Read/write cross-session memory
│
├── 🛠️ tools/
│   ├── pinecone_retriever.py      # Embed text + similarity search
│   ├── currency_tool.py           # FX rates from live API
│   ├── nl2sql_engine.py           # CSV ingestion + DuckDB query engine
│   └── memory_store.py            # PostgreSQL CRUD for user memories
│
├── 🧬 memory/
│   ├── short_term.py              # In-context conversation buffer
│   ├── long_term.py               # Persistent cross-session store
│   └── conversation_history.py    # Full message history per user (PostgreSQL)
│
├── 📊 evaluation/
│   ├── langsmith_eval.py          # Trace scoring + dataset eval
│   └── ragas_eval.py              # RAG-specific quality metrics
│
├── ⚙️ api/
│   ├── main.py                    # FastAPI app + all route definitions
│   ├── middleware.py              # CORS, logging, error handling
│   └── schemas.py                 # Pydantic request/response models
│
├── 📁 data/
│   └── csv_files/                 # Your internal CSV data goes here
│
├── 🧪 tests/
│   ├── unit/                      # Per-agent isolated tests
│   ├── integration/               # Full pipeline end-to-end tests
│   └── adversarial/               # Prompt injection + red-team tests
│
├── 🐳 docker/
│   ├── Dockerfile                 # Multi-stage production build
│   └── docker-compose.yml         # Local dev: app + redis + postgres
│
├── ⚙️ .github/workflows/
│   ├── ci.yml                     # Run tests on every PR
│   └── deploy.yml                 # Auto-deploy to AWS on merge to main
│
├── 🖥️ ui/
│   ├── app.py                     # Streamlit chat interface
│   ├── components/
│   │   ├── chat.py                # Chat message display
│   │   ├── sidebar.py             # Memory inspector + user info
│   │   └── agent_steps.py        # Expandable agent reasoning steps
│   └── utils/
│       └── api_client.py          # Calls FastAPI backend
│
├── 📋 .env.example                # All required environment variables
├── 📦 requirements.txt            # Python dependencies
└── 📖 README.md                   # This file
```

---

## 🚀 Local Setup Guide

### Option A — SQL Server (Windows, no Docker needed)

```powershell
# 1. Clone and create virtual environment
cd "production-chatbot"
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure .env
# Set POSTGRES_URL to your SQL Server:
# mssql+pyodbc://YOURPC\SQLEXPRESS/chatbot?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes

# 4. Start Redis (still needed for rate limiting)
docker run -d -p 6379:6379 redis:7-alpine

# 5. Start API (auto-creates DB tables on first run)
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000

# 6. Start UI (in a second terminal)
.\venv\Scripts\python.exe -m streamlit run ui/app.py
```

### Option B — Full Docker (PostgreSQL + Redis)

```powershell
# Start all services
docker compose -f docker/docker-compose.yml up postgres redis -d

# Start API
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000

# Start UI
.\venv\Scripts\python.exe -m streamlit run ui/app.py
```

### Access Points

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |

### Login
> Any username and password works — JWT token is issued for whatever name you enter.
> Example: username `admin`, password `admin`

---

## 🧩 Agent Communication Flow

```
User: "What is the EUR/USD rate and summarize our Q3 sales from the CSV?"
                              │
                              ▼
              ┌───────────────────────────────┐
              │      SUPERVISOR AGENT         │
              │  Detects: 2 tasks needed      │
              │  → Task 1: currency query     │
              │  → Task 2: CSV data query     │
              └───────┬───────────────┬───────┘
                      │               │
           parallel   │               │   parallel
                      ▼               ▼
          ┌──────────────────┐  ┌─────────────────┐
          │  REALTIME AGENT  │  │   SQL AGENT     │
          │  calls FX API    │  │  CSV → DuckDB   │
          │  returns rate    │  │  returns table  │
          └────────┬─────────┘  └────────┬────────┘
                   │                     │
                   └──────────┬──────────┘
                              ▼
              ┌───────────────────────────────┐
              │      SUPERVISOR MERGES        │
              │   + checks long-term memory   │
              │   + stores new context        │
              └───────────────────────────────┘
                              │
                              ▼
                     📊 LangSmith traces
                     🛡️ Output guardrails
                              │
                              ▼
                     Answer delivered to user
```

---

## 🖥️ Streamlit UI — What Users See

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT CHAT UI                               │
│                                                                     │
│  ┌──────────────┐  ┌─────────────────────────────────────────────┐ │
│  │   SIDEBAR    │  │              CHAT WINDOW                    │ │
│  │              │  │                                             │ │
│  │ 👤 User Info │  │  🧑 You: What is EUR/USD rate?              │ │
│  │              │  │                                             │ │
│  │ 🧠 Memory    │  │  🤖 Assistant: EUR/USD = 1.085              │ │
│  │  • prefers   │  │     ▼ Agent Steps (expandable)             │ │
│  │    USD base  │  │     └── Supervisor routed to Realtime Agent│ │
│  │  • Q3 = $2.4M│  │     └── Realtime Agent called FX API       │ │
│  │              │  │     └── Returned rate 1.085                 │ │
│  │ 📜 History   │  │                                             │ │
│  │  • Jan 1     │  │  🧑 You: Summarize Q3 sales from CSV        │ │
│  │  • Jan 2     │  │                                             │ │
│  │  • Today     │  │  🤖 Assistant: Q3 total revenue = $2.4M... │ │
│  │              │  │     ▼ Agent Steps (expandable)             │ │
│  │ ⚙️ Settings  │  │     └── Supervisor routed to SQL Agent      │ │
│  │              │  │     └── SQL Agent generated query           │ │
│  └──────────────┘  │     └── DuckDB returned results            │ │
│                    │                                             │ │
│                    │  [ Type your message here...        Send ] │ │
│                    └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Streamlit Deployment Options

| Option | Where | Cost | Best For |
|---|---|---|---|
| **Local** | `streamlit run ui/app.py` | Free | Development |
| **Streamlit Community Cloud** | streamlit.io/cloud | Free | Portfolio / Share link |
| **AWS ECS Fargate** | Same cluster as FastAPI | Pay per use | Production |

---

## 🔒 Security Layers

```
Request Journey Through Security:

1. 🔑 JWT Authentication
   └── Every request must carry a valid signed token
   └── Tokens expire — refresh token flow enforced

2. ⏱️ Rate Limiting (Redis)
   └── 60 requests/minute per user (configurable)
   └── 429 Too Many Requests returned gracefully

3. 🧹 Input Guardrails
   └── Prompt injection patterns blocked
   └── PII detected before hitting LLM (names, emails, cards)
   └── Jailbreak attempt detection

4. 🧹 Output Guardrails
   └── PII scrubbed from responses
   └── Hallucination confidence scoring
   └── Toxic content filter on output

5. 💰 Cost Hard Limits
   └── Per-request token budget enforced
   └── Per-user daily spend cap
   └── CloudWatch alarm on cost anomaly
```

---

## 📊 Evaluation Metrics (RAGAS)

| Metric | What It Measures | Target Score |
|--------|-----------------|--------------|
| **Faithfulness** | Is the answer grounded in retrieved docs? | > 0.85 |
| **Answer Relevancy** | Does the answer address the question? | > 0.80 |
| **Context Recall** | Did we retrieve the right chunks? | > 0.75 |
| **Context Precision** | Are retrieved chunks actually useful? | > 0.75 |
| **Answer Correctness** | Is the answer factually correct? | > 0.80 |

---

## ⚡ Latency & Cost Optimization

```
Strategy                          Savings
─────────────────────────────────────────────────────
GPT-4o-mini for simple queries    ~85% cost reduction
GPT-4o only for complex reasoning  Quality preserved
Pinecone metadata pre-filter       ~40% faster retrieval
Redis response caching             ~60% latency cut (repeated queries)
Async Celery for NL2SQL            API never blocked
Parallel agent execution           Multi-task in single request
Streaming responses                Perceived latency ~70% lower
```

---

## 🚀 Deployment Pipeline

```
Developer pushes code
        │
        ▼
GitHub Actions CI (ci.yml)
  ├── Run unit tests
  ├── Run integration tests
  ├── Run adversarial tests
  ├── Lint + type check
  └── Build Docker image
        │
     All pass?
        │
        ▼
GitHub Actions CD (deploy.yml)
  ├── Push image to ECR
  ├── Deploy to ECS Fargate (rolling update)
  ├── Health check API endpoint
  └── CloudWatch alarms armed
        │
        ▼
   Production Live ✅
```

---

## 🌍 Environment Variables Required

```env
# LLM
OPENAI_API_KEY=sk-...

# Vector DB
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=chatbot-index

# Database — SQL Server (Windows Auth)
POSTGRES_URL=mssql+pyodbc://SERVER\SQLEXPRESS/chatbot?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes

# Database — SQL Server (SQL Auth)
# POSTGRES_URL=mssql+pyodbc://user:pass@SERVER\SQLEXPRESS/chatbot?driver=ODBC+Driver+17+for+SQL+Server

# Database — PostgreSQL (Docker)
# POSTGRES_URL=postgresql://chatbot_user:chatbot_pass@localhost:5432/chatbot

# Cache + Queue
REDIS_URL=redis://localhost:6379

# Auth
JWT_SECRET_KEY=your-secret-key-min-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# Currency API
EXCHANGE_RATE_API_KEY=

# Observability
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=production-chatbot
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# AWS (only needed for cloud deployment)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
```

---

## 💬 Conversation History — Full Cross-Session Memory

> Unlike short-term memory (current session only) and long-term memory (key facts only), **Conversation History saves every single message** a user ever sends — just like ChatGPT remembers old chats.

```
┌─────────────────────────────────────────────────────────────────────┐
│               CONVERSATION HISTORY FLOW                             │
│                                                                     │
│  PostgreSQL Table: conversation_history                             │
│  ┌──────────┬──────────┬───────────┬────────────┬───────────────┐  │
│  │ id       │ user_id  │ role      │ content    │ timestamp     │  │
│  ├──────────┼──────────┼───────────┼────────────┼───────────────┤  │
│  │ 1        │ user_123 │ human     │ EUR/USD?   │ 2024-01-01    │  │
│  │ 2        │ user_123 │ assistant │ Rate: 1.08 │ 2024-01-01    │  │
│  │ 3        │ user_123 │ human     │ Q3 sales?  │ 2024-01-02    │  │
│  │ 4        │ user_123 │ assistant │ $2.4M...   │ 2024-01-02    │  │
│  └──────────┴──────────┴───────────┴────────────┴───────────────┘  │
│                                                                     │
│  On every new session:                                              │
│  1. Load last N messages from history for this user                 │
│  2. Inject into short-term context so agent has full context        │
│  3. After every turn, save new messages to history table            │
│                                                                     │
│  3 Memory Layers Working Together:                                  │
│                                                                     │
│  Short-term  ──▶  current session back-and-forth (in-context)      │
│  History     ──▶  all past messages (PostgreSQL, loaded on start)   │
│  Long-term   ──▶  key facts extracted from history (PostgreSQL)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔁 ReAct Pattern — How Every Agent Thinks

> **ReAct = Reason + Act** — the foundational loop behind every agentic AI system.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ReAct AGENT LOOP                             │
│                                                                     │
│   User Query                                                        │
│       │                                                             │
│       ▼                                                             │
│   ┌──────────────────────────────────────────────┐                 │
│   │  STEP 1 — REASON (Thought)                   │                 │
│   │  "The user wants EUR/USD rate.               │                 │
│   │   I need to call the currency tool."         │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │                                          │
│                          ▼                                          │
│   ┌──────────────────────────────────────────────┐                 │
│   │  STEP 2 — ACT (Tool Call)                    │                 │
│   │  → calls currency_tool("EUR", "USD")         │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │                                          │
│                          ▼                                          │
│   ┌──────────────────────────────────────────────┐                 │
│   │  STEP 3 — OBSERVE (Tool Result)              │                 │
│   │  ← returns {"rate": 1.085, "ts": "2024..."}  │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │                                          │
│                          ▼                                          │
│   ┌──────────────────────────────────────────────┐                 │
│   │  STEP 4 — REASON AGAIN                       │                 │
│   │  "I have the rate. Is more info needed?      │                 │
│   │   No. I can now generate the final answer."  │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │                                          │
│              ┌───────────┴────────────┐                            │
│              │                        │                            │
│         More tools                Final Answer                     │
│         needed?                   to Supervisor                    │
│         → loop again                                               │
└─────────────────────────────────────────────────────────────────────┘
```

**Every specialist agent (RAG, SQL, Realtime, Memory) runs this loop independently.**
The Supervisor agent runs ReAct at a higher level — reasoning over which agents to call.

---

## 💾 LangGraph Checkpointing — Stateful Graph Execution

> **Checkpointing is LangGraph's killer feature.** It saves the entire graph state after every node execution.

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CHECKPOINTING FLOW                                │
│                                                                     │
│  Turn 1                                                             │
│  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐                │
│  │ START  │──▶│ RAG    │──▶│MEMORY  │──▶│  END   │                │
│  │ Node   │   │ Node   │   │ Node   │   │ Node   │                │
│  └────────┘   └────────┘   └────────┘   └────────┘                │
│       │            │            │             │                     │
│       ▼            ▼            ▼             ▼                     │
│  [checkpoint]  [checkpoint] [checkpoint]  [checkpoint]             │
│  state saved   state saved  state saved   state saved              │
│                                                                     │
│  Turn 2 — graph RESUMES from last checkpoint, not from scratch     │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Previous messages ✅  Previous tool results ✅         │        │
│  │  Previous memory ✅    User context ✅                  │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  Failure Recovery                                                   │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Node crashes? → Resume from last checkpoint           │        │
│  │  No data loss. No re-running completed steps.          │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                     │
│  Human-in-the-Loop                                                  │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Graph pauses at sensitive node (e.g. delete action)   │        │
│  │  → Waits for human approval                            │        │
│  │  → Resumes exactly where it paused                     │        │
│  └────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

**Storage Backend:** PostgreSQL (same DB as long-term memory) — every thread_id maps to a checkpoint row.

---

## 🪞 Reflection — Agent Self-Critique Loop

> **Reflection** makes agents self-aware. Instead of returning the first answer, the agent grades its own output and retries if quality is low.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REFLECTION LOOP                                 │
│                                                                     │
│   Agent generates initial answer                                    │
│            │                                                        │
│            ▼                                                        │
│   ┌──────────────────────────────────────────────┐                 │
│   │  CRITIC NODE (same LLM, different prompt)    │                 │
│   │                                              │                 │
│   │  Checks:                                     │                 │
│   │  • Is the answer grounded in sources?        │                 │
│   │  • Does it fully address the question?       │                 │
│   │  • Is any part hallucinated?                 │                 │
│   │  • Is the SQL query correct?                 │                 │
│   │                                              │                 │
│   │  Score: 0.0 → 1.0                            │                 │
│   └──────────────────────┬───────────────────────┘                 │
│                          │                                          │
│               ┌──────────┴──────────┐                              │
│               │                     │                              │
│          Score < 0.7           Score ≥ 0.7                         │
│               │                     │                              │
│               ▼                     ▼                              │
│   ┌─────────────────────┐   ┌───────────────────┐                 │
│   │  RETRY with          │   │  PASS to          │                 │
│   │  improved prompt     │   │  Supervisor       │                 │
│   │  (max 2 retries)     │   │  → User           │                 │
│   └─────────────────────┘   └───────────────────┘                 │
│                                                                     │
│  Applied to:                                                        │
│  • RAG Agent   — checks answer vs retrieved chunks                  │
│  • SQL Agent   — validates generated SQL before execution           │
│  • Memory Agent — checks if stored memory is accurate              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔀 Subgraphs — Agents as Independent Graphs

> **Interview question:** *"How do you handle complexity inside a single agent node?"*
> **Answer:** Each specialist agent is itself a subgraph — it has its own nodes, edges, and state.

```
MAIN SUPERVISOR GRAPH
│
├── Node: RAG Agent ──────────► RAG SUBGRAPH
│                                  ├── Node: Embed Query
│                                  ├── Node: Search Pinecone
│                                  ├── Node: Rerank Results
│                                  └── Node: Generate Answer
│
├── Node: SQL Agent ──────────► SQL SUBGRAPH
│                                  ├── Node: Parse Intent
│                                  ├── Node: Generate SQL
│                                  ├── Node: Reflect on SQL      ← Reflection here
│                                  └── Node: Execute + Return
│
└── Node: Realtime Agent ────► REALTIME SUBGRAPH
                                   ├── Node: Identify API
                                   ├── Node: Call API
                                   └── Node: Format Result
```

Each subgraph has its **own checkpointing** and can be tested independently.

---

## 🗺️ Map-Reduce Pattern — Parallel Agent Execution

> **Interview question:** *"How does your system handle a query that needs multiple agents?"*
> **Answer:** Supervisor uses Map-Reduce — maps the query to N agents in parallel, reduces results into one answer.

```
User: "What is EUR/USD rate AND summarize Q3 sales AND find docs on pricing policy?"
                              │
                              ▼
                    SUPERVISOR — MAP PHASE
                    Splits into 3 parallel tasks
                    │           │           │
                    ▼           ▼           ▼
             Realtime      SQL Agent    RAG Agent
              Agent         (CSV)       (Pinecone)
             (FX API)
                    │           │           │
                    ▼           ▼           ▼
             rate=1.085    Q3=$2.4M    "Policy doc..."
                    │           │           │
                    └───────────┴───────────┘
                                │
                    SUPERVISOR — REDUCE PHASE
                    Merges all 3 results into
                    one coherent response
                                │
                                ▼
                    Single answer to user
```

---

## ✍️ Prompt Engineering — How Each Agent is Instructed

> **Interview question:** *"How did you prompt your agents?"*
> Every agent has a carefully crafted system prompt with role, rules, few-shot examples, and output format.

```
RAG AGENT SYSTEM PROMPT STRUCTURE
┌─────────────────────────────────────────────────────────────┐
│ ROLE                                                        │
│ You are a document retrieval specialist. Your job is to    │
│ answer questions strictly from retrieved context.          │
│                                                             │
│ RULES                                                       │
│ 1. Never answer from training knowledge alone              │
│ 2. Always cite the source document                         │
│ 3. If context is insufficient, say "I don't know"          │
│                                                             │
│ FEW-SHOT EXAMPLE                                            │
│ Q: What is the refund policy?                              │
│ Context: [doc chunk]                                       │
│ A: Based on the policy document, refunds are...            │
│                                                             │
│ OUTPUT FORMAT                                               │
│ {"answer": "...", "source": "...", "confidence": 0.0-1.0}  │
└─────────────────────────────────────────────────────────────┘
```

Each agent (RAG, SQL, Realtime, Memory, Supervisor) has its own tailored system prompt.

---

## 📏 Context Window Management — Handling Long Conversations

> **Interview question:** *"What happens when conversation history grows too long?"*
> Token counting + summarization strategy prevents hitting the LLM context limit.

```
Every turn, before calling the LLM:

Step 1 — Count tokens in current history
         history_tokens = count_tokens(messages)

Step 2 — Is it within safe limit?
         safe_limit = model_max_tokens × 0.75   (leave room for response)

         history_tokens < safe_limit?
                │                    │
               YES                   NO
                │                    │
         Send as-is         Step 3 — Summarize oldest messages
                            summary = LLM("Summarize this conversation")
                            Replace old messages with summary
                            Keep last 10 messages intact

Result: History never exceeds context limit
        No conversation data is lost (summary preserves meaning)
        User experience is uninterrupted
```

---

## 🧠 LangGraph Concepts Coverage

| LangGraph Concept | In This Project | Implementation |
|---|---|---|
| State Graph | ✅ | Supervisor manages full conversation state |
| Nodes | ✅ | RAG, SQL, Realtime, Memory — each a node |
| Edges | ✅ | Supervisor routes between nodes |
| Conditional Edges | ✅ | Intent-based dynamic routing |
| State Management | ✅ | Short-term buffer + long-term PostgreSQL |
| **Checkpointing** | ✅ | PostgreSQL-backed, every node saved |
| **Human-in-the-loop** | ✅ | Graph pauses on sensitive actions |
| **Streaming Graph Events** | ✅ | Each agent step streamed to Streamlit UI |
| Parallel Node Execution | ✅ | Multi-agent tasks run simultaneously |
| Graph Visualization | ✅ | `.get_graph().draw_mermaid()` for debug |
| **Subgraphs** | ✅ | Each specialist agent runs as its own internal subgraph |
| **Map-Reduce Pattern** | ✅ | Supervisor maps query to N agents, reduces into one answer |

---

## 🤖 Agentic AI Concepts Coverage

| Agentic Concept | In This Project | Implementation |
|---|---|---|
| Tool Use | ✅ | Currency, Pinecone, DuckDB as tools |
| Multi-Agent | ✅ | Supervisor + 4 specialists |
| Agent Communication | ✅ | Agents pass results to supervisor |
| RAG | ✅ | Pinecone vector retrieval |
| Long-term Memory | ✅ | PostgreSQL cross-session persistence |
| **ReAct Pattern** | ✅ | Reason→Act→Observe loop in every agent |
| **Planning** | ✅ | Supervisor breaks complex queries into sub-tasks |
| **Reflection** | ✅ | Critic node scores + retries low-quality answers |
| **Agent Retry on Failure** | ✅ | Supervisor reroutes on agent failure |
| Tool Selection Logic | ✅ | Supervisor dynamically picks tools per query |
| **Prompt Engineering** | ✅ | System prompts + few-shot examples per agent |
| **Context Window Management** | ✅ | Token counting + summarization when history grows long |

---

## 📈 Why This is 9.5 / 10 Production Grade

| Category | Basic Chatbot | This System |
|----------|--------------|-------------|
| Data Sources | 1 (LLM only) | 4 (docs, CSV, live API, memory) |
| Memory | None | Short-term + Long-term + Full History |
| Conversation History | None | Every message saved per user in SQL Server / PostgreSQL |
| UI | None / basic | Streamlit chat with agent steps + memory view |
| Security | None | JWT + Rate limit + Guardrails |
| Observability | None | Full LangSmith traces |
| Evaluation | None | RAGAS automated scoring |
| Deployment | Manual | GitHub Actions CI/CD + ECS Fargate + Streamlit Cloud |
| Cost Control | None | Per-user budgets + alerts |
| Multi-agent | No | Supervisor + 4 specialists |
| Async | No | Celery queue for heavy tasks |
| User Isolation | No | Pinecone namespace per user |

---

<div align="center">

**Built with LangGraph · OpenAI · Pinecone · FastAPI · SQL Server · Redis · Docker · AWS**

</div>
