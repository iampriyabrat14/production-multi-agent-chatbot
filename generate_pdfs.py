from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

W, H = A4
MARGIN = 2 * cm

# ── Color Palette ──────────────────────────────────────────────────────────────
DARK_BLUE   = colors.HexColor("#1a237e")
MID_BLUE    = colors.HexColor("#1565c0")
LIGHT_BLUE  = colors.HexColor("#e3f2fd")
ACCENT      = colors.HexColor("#00acc1")
GREEN       = colors.HexColor("#2e7d32")
ORANGE      = colors.HexColor("#e65100")
PURPLE      = colors.HexColor("#6a1b9a")
GREY_BG     = colors.HexColor("#f5f5f5")
GREY_TEXT   = colors.HexColor("#546e7a")
WHITE       = colors.white
BLACK       = colors.black


def make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle("cover_title",
        fontSize=28, textColor=WHITE, alignment=TA_CENTER,
        fontName="Helvetica-Bold", leading=36, spaceAfter=12)

    styles["cover_sub"] = ParagraphStyle("cover_sub",
        fontSize=14, textColor=colors.HexColor("#bbdefb"),
        alignment=TA_CENTER, fontName="Helvetica", leading=20, spaceAfter=6)

    styles["cover_meta"] = ParagraphStyle("cover_meta",
        fontSize=11, textColor=colors.HexColor("#90caf9"),
        alignment=TA_CENTER, fontName="Helvetica", leading=16)

    styles["h1"] = ParagraphStyle("h1",
        fontSize=20, textColor=DARK_BLUE, fontName="Helvetica-Bold",
        spaceBefore=18, spaceAfter=8, leading=26,
        borderPad=4, leftIndent=0)

    styles["h2"] = ParagraphStyle("h2",
        fontSize=15, textColor=MID_BLUE, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6, leading=20)

    styles["h3"] = ParagraphStyle("h3",
        fontSize=12, textColor=ACCENT, fontName="Helvetica-Bold",
        spaceBefore=10, spaceAfter=4, leading=16)

    styles["body"] = ParagraphStyle("body",
        fontSize=10, textColor=colors.HexColor("#212121"),
        fontName="Helvetica", leading=16, spaceAfter=6,
        alignment=TA_JUSTIFY)

    styles["bullet"] = ParagraphStyle("bullet",
        fontSize=10, textColor=colors.HexColor("#212121"),
        fontName="Helvetica", leading=15, spaceAfter=3,
        leftIndent=16, bulletIndent=6)

    styles["code"] = ParagraphStyle("code",
        fontSize=9, textColor=colors.HexColor("#1a237e"),
        fontName="Courier", leading=13, spaceAfter=4,
        leftIndent=12, backColor=GREY_BG,
        borderColor=colors.HexColor("#90caf9"),
        borderWidth=0.5, borderPad=6)

    styles["label"] = ParagraphStyle("label",
        fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
        alignment=TA_CENTER, leading=12)

    styles["table_header"] = ParagraphStyle("table_header",
        fontSize=9, textColor=WHITE, fontName="Helvetica-Bold",
        alignment=TA_CENTER, leading=12)

    styles["table_cell"] = ParagraphStyle("table_cell",
        fontSize=9, textColor=BLACK, fontName="Helvetica",
        alignment=TA_LEFT, leading=12)

    styles["highlight"] = ParagraphStyle("highlight",
        fontSize=10, textColor=DARK_BLUE, fontName="Helvetica-Bold",
        backColor=LIGHT_BLUE, leading=15, spaceAfter=4,
        leftIndent=10, borderColor=MID_BLUE, borderWidth=1, borderPad=6)

    styles["answer"] = ParagraphStyle("answer",
        fontSize=10, textColor=colors.HexColor("#1b5e20"),
        fontName="Helvetica", leading=15, spaceAfter=4,
        leftIndent=10, backColor=colors.HexColor("#e8f5e9"),
        borderColor=GREEN, borderWidth=0.5, borderPad=5)

    styles["question"] = ParagraphStyle("question",
        fontSize=11, textColor=DARK_BLUE, fontName="Helvetica-Bold",
        spaceBefore=12, spaceAfter=4, leading=16,
        leftIndent=0)

    return styles


def cover_page(title, subtitle, doc_num, styles):
    elems = []
    # dark header block via table
    data = [[Paragraph(title, styles["cover_title"])]]
    t = Table(data, colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), DARK_BLUE),
        ("TOPPADDING", (0,0), (-1,-1), 30),
        ("BOTTOMPADDING", (0,0), (-1,-1), 30),
        ("LEFTPADDING", (0,0), (-1,-1), 20),
        ("RIGHTPADDING", (0,0), (-1,-1), 20),
        ("ROUNDEDCORNERS", [8]),
    ]))
    elems.append(Spacer(1, 2*cm))
    elems.append(t)
    elems.append(Spacer(1, 0.6*cm))
    elems.append(Paragraph(subtitle, styles["cover_sub"]))
    elems.append(Spacer(1, 0.3*cm))
    elems.append(Paragraph(f"Document {doc_num} of 3  •  Production Multi-Agent Chatbot  •  {datetime.now().strftime('%B %Y')}", styles["cover_meta"]))
    elems.append(Spacer(1, 0.5*cm))
    elems.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
    elems.append(Spacer(1, 0.4*cm))

    # tag badges
    tags = ["LangGraph", "GPT-4o", "Pinecone", "DuckDB", "FastAPI", "Streamlit", "SQL Server", "Docker"]
    badge_data = [[Paragraph(t, styles["label"]) for t in tags]]
    bt = Table(badge_data, colWidths=[(W-2*MARGIN)/len(tags)]*len(tags))
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), MID_BLUE),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("GRID", (0,0), (-1,-1), 0.5, WHITE),
        ("ROUNDEDCORNERS", [4]),
    ]))
    elems.append(bt)
    elems.append(PageBreak())
    return elems


def section_header(title, styles):
    data = [[Paragraph(title, ParagraphStyle("sh", fontSize=16, textColor=WHITE,
        fontName="Helvetica-Bold", leading=22, alignment=TA_LEFT))]]
    t = Table(data, colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), MID_BLUE),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 14),
        ("RIGHTPADDING", (0,0), (-1,-1), 14),
    ]))
    return [Spacer(1, 0.3*cm), t, Spacer(1, 0.3*cm)]


def info_box(text, styles, color=LIGHT_BLUE, border=MID_BLUE):
    data = [[Paragraph(text, ParagraphStyle("ib", fontSize=10,
        fontName="Helvetica", leading=15, textColor=DARK_BLUE))]]
    t = Table(data, colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("BOX", (0,0), (-1,-1), 1, border),
    ]))
    return [t, Spacer(1, 0.2*cm)]


def make_table(headers, rows, styles, col_widths=None):
    if col_widths is None:
        col_widths = [(W - 2*MARGIN) / len(headers)] * len(headers)
    data = [[Paragraph(h, styles["table_header"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), styles["table_cell"]) for c in row])
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), DARK_BLUE),
        ("BACKGROUND", (0,1), (-1,-1), WHITE),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, GREY_BG]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bdbdbd")),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    return [t, Spacer(1, 0.3*cm)]


# ══════════════════════════════════════════════════════════════════════════════
# PDF 1 — Application Flow & Architecture
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf1(path, styles):
    doc = SimpleDocTemplate(path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN)
    elems = []

    elems += cover_page(
        "Application Flow & Architecture",
        "End-to-End System Design of the Production Multi-Agent Chatbot",
        "1", styles)

    # ── Section 1: System Overview ──────────────────────────────────────────
    elems += section_header("1. System Overview", styles)
    elems.append(Paragraph(
        "The Production Multi-Agent Chatbot is a LangGraph-powered AI system that routes user questions "
        "to specialist agents — RAG (documents), SQL (CSV data), Realtime (live rates), and Memory. "
        "It is built with FastAPI + Streamlit, secured with JWT, and persists data to SQL Server.",
        styles["body"]))
    elems.append(Spacer(1, 0.3*cm))

    # Architecture layers table
    elems += make_table(
        ["Layer", "Technology", "Purpose"],
        [
            ["UI Layer", "Streamlit", "Chat interface, login, session management"],
            ["API Layer", "FastAPI + JWT", "Auth, rate limiting, guardrails, routing"],
            ["Orchestration", "LangGraph StateGraph", "Multi-agent supervisor with checkpointing"],
            ["LLM", "OpenAI GPT-4o / GPT-4o-mini", "Answer generation, planning, reflection"],
            ["RAG", "Pinecone + OpenAI Embeddings", "Document retrieval and Q&A"],
            ["NL2SQL", "DuckDB + GPT-4o", "Natural language to SQL over CSV files"],
            ["Realtime", "ExchangeRate-API + httpx", "Live currency conversion"],
            ["Memory", "SQL Server + SQLAlchemy", "User facts + conversation history"],
            ["Security", "Presidio + python-jose", "PII scrubbing + JWT tokens"],
            ["Observability", "LangSmith", "LLM tracing and evaluation"],
            ["Infra", "Docker + GitHub Actions", "Containerization and CI/CD"],
        ],
        styles,
        col_widths=[4.5*cm, 5.5*cm, 8.5*cm])

    # ── Section 2: Request Flow ─────────────────────────────────────────────
    elems += section_header("2. Complete Request Flow", styles)

    steps = [
        ("Step 1 — User Login", "User enters credentials in Streamlit sidebar → POST /auth/token → FastAPI creates signed JWT (HS256, 30 min expiry) → token stored in Streamlit session state."),
        ("Step 2 — User Sends Message", "User types question → Streamlit calls POST /chat with Bearer token + message + session_id."),
        ("Step 3 — Security Layer", "FastAPI checks JWT (Depends(get_current_user)) → checks rate limit (60/min via slowapi+Redis) → runs injection detection (20+ regex patterns) → anonymizes PII in input (Presidio)."),
        ("Step 4 — Supervisor Entry", "run_supervisor() invoked → LangGraph StateGraph starts with initial SupervisorState (question, user_id, session_id)."),
        ("Step 5 — Memory Retrieve Node", "memory_retrieve_node() loads user facts + last 10 conversation turns from SQL Server → injects as context string into state."),
        ("Step 6 — Plan Node (Routing)", "plan_node() calls GPT-4o-mini with routing prompt + examples → returns JSON array e.g. ['sql'] or ['rag', 'realtime'] → keyword fallback if JSON parse fails."),
        ("Step 7 — Human-in-the-Loop Check", "needs_human_approval() checks for sensitive keywords (delete, drop, transfer) → if detected, routes to END for human review; otherwise continues to route node."),
        ("Step 8 — Route Node (Map Phase)", "route_node() executes planned agents. Single agent: direct call. Multiple agents: ThreadPoolExecutor runs them in parallel (Map-Reduce pattern)."),
        ("Step 9 — Agent Execution", "Each agent runs its own LangGraph subgraph with internal nodes (retrieve/generate/reflect for RAG; understand/generate/reflect/execute/format for SQL; parse/call/format for Realtime)."),
        ("Step 10 — Merge Node (Reduce Phase)", "merge_node() combines results. Single agent: pass-through. Multiple agents: GPT-4o merges into one coherent answer."),
        ("Step 11 — Memory Save Node", "memory_save_node() saves conversation to SQL Server (conversation_history table) + extracts new user facts via GPT-4o-mini (user_facts table)."),
        ("Step 12 — Output Security", "anonymize_pii_output() scrubs emails, phone numbers, credit cards from response before returning to user."),
        ("Step 13 — Response", "ChatResponse returned with: answer, agent_used, sources, latency_ms, session_id, timestamp."),
    ]

    for title, desc in steps:
        elems.append(Paragraph(f"<b>{title}</b>", styles["h3"]))
        elems.append(Paragraph(desc, styles["body"]))

    # ── Section 3: LangGraph State Flow ────────────────────────────────────
    elems += section_header("3. LangGraph Graph Structure", styles)

    elems.append(Paragraph("<b>Supervisor Graph (Main Graph):</b>", styles["h3"]))
    flow = [
        ["Node", "Type", "Function"],
        ["memory_retrieve", "Standard Node", "Load user context from SQL Server"],
        ["planner", "Standard Node", "GPT-4o-mini decides which agents to call"],
        ["needs_human_approval", "Conditional Edge", "Check for sensitive operations"],
        ["route", "Standard Node", "Execute agents (parallel via ThreadPoolExecutor)"],
        ["merge", "Standard Node", "Combine agent results into final answer"],
        ["memory_save", "Standard Node", "Persist conversation + extract facts"],
    ]
    elems += make_table(flow[0], flow[1:], styles, col_widths=[4.5*cm, 4.5*cm, 9.5*cm])

    elems.append(Paragraph("<b>RAG Subgraph:</b>", styles["h3"]))
    rag_flow = [
        ["Node", "Model", "Function"],
        ["retrieve", "Pinecone", "Embed query → search top-5 chunks"],
        ["grade_chunks", "GPT-4o-mini", "Filter chunks with score > 0.75"],
        ["generate", "GPT-4o", "Answer from grounded context"],
        ["reflect", "GPT-4o-mini", "Score answer quality 0.0-1.0, retry if < 0.7"],
    ]
    elems += make_table(rag_flow[0], rag_flow[1:], styles, col_widths=[4*cm, 4*cm, 10.5*cm])

    elems.append(Paragraph("<b>SQL Subgraph:</b>", styles["h3"]))
    sql_flow = [
        ["Node", "Model", "Function"],
        ["understand_intent", "GPT-4o-mini", "Detect which CSV files are needed"],
        ["generate_sql", "GPT-4o", "Generate DuckDB SQL from question + schema"],
        ["reflect_sql", "GPT-4o-mini", "Validate SQL columns/GROUP BY/syntax"],
        ["execute_sql", "DuckDB", "Run SQL and return DataFrame"],
        ["format_result", "GPT-4o", "Convert DataFrame to natural language"],
    ]
    elems += make_table(sql_flow[0], sql_flow[1:], styles, col_widths=[4.5*cm, 4*cm, 10*cm])

    elems.append(Paragraph("<b>Realtime Subgraph:</b>", styles["h3"]))
    rt_flow = [
        ["Node", "Tool", "Function"],
        ["parse_intent", "GPT-4o-mini", "Extract base/target currency + amount"],
        ["call_tool", "ExchangeRate-API", "Sync HTTP GET for live exchange rate"],
        ["handle_error / format_response", "—", "Error fallback or format rate as readable text"],
    ]
    elems += make_table(rt_flow[0], rt_flow[1:], styles, col_widths=[5.5*cm, 4*cm, 9*cm])

    # ── Section 4: Data Flow ────────────────────────────────────────────────
    elems += section_header("4. Data Storage Architecture", styles)
    elems += make_table(
        ["Store", "Technology", "Table/Index", "What is Stored"],
        [
            ["Vector DB", "Pinecone (serverless)", "chatbot-index", "Document chunks as 1536-dim vectors (text-embedding-3-small)"],
            ["Conversation", "SQL Server", "conversation_history", "Every human/AI message with user_id, session_id, timestamp"],
            ["User Facts", "SQL Server", "user_facts", "Extracted key-value facts per user (name, role, preferences)"],
            ["Short-term", "RAM (LangGraph)", "MemorySaver", "Full graph state per thread_id, last 20 messages"],
            ["CSV Data", "DuckDB (in-memory)", "Dynamic tables", "CSV files loaded as DuckDB tables for SQL queries"],
            ["Rate Limit", "Redis", "Key-value", "Request counters per IP/user for 60/min rate limiting"],
        ],
        styles,
        col_widths=[3*cm, 3.5*cm, 4*cm, 8*cm])

    # ── Section 5: Security Architecture ────────────────────────────────────
    elems += section_header("5. Security Architecture", styles)
    elems += make_table(
        ["Layer", "What It Protects Against", "Implementation"],
        [
            ["JWT Auth", "Unauthorized access", "HS256 signed tokens, 30-min expiry, OAuth2PasswordBearer"],
            ["Rate Limiting", "DDoS / API abuse", "slowapi + Redis, 60 requests/minute per user"],
            ["Input Guardrails", "Prompt injection, jailbreaks", "20+ regex patterns: ignore instructions, act as, jailbreak, SQL injection, XSS, SSTI"],
            ["PII Input Scrub", "User PII reaching LLM", "Presidio anonymizes PERSON, EMAIL, PHONE, CREDIT_CARD in input"],
            ["PII Output Scrub", "Sensitive data in responses", "Presidio anonymizes EMAIL, PHONE, CREDIT_CARD in output"],
            ["CORS", "Cross-origin attacks", "FastAPI CORSMiddleware, configurable origins"],
        ],
        styles,
        col_widths=[3.5*cm, 5*cm, 10*cm])

    doc.build(elems)
    print(f"PDF 1 saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF 2 — Complete Technical Documentation
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf2(path, styles):
    doc = SimpleDocTemplate(path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN)
    elems = []

    elems += cover_page(
        "Complete Technical Documentation",
        "All Files, APIs, Configurations & Deployment Guide",
        "2", styles)

    # ── Section 1: Project Structure ────────────────────────────────────────
    elems += section_header("1. Project Structure", styles)
    structure = [
        "production-chatbot/",
        "├── api/",
        "│   ├── main.py          # FastAPI app, /chat, /auth/token, /health",
        "│   ├── schemas.py       # Pydantic request/response models",
        "│   └── middleware.py    # Request logging middleware",
        "├── agents/",
        "│   ├── supervisor.py    # LangGraph supervisor — routing + orchestration",
        "│   ├── rag_agent.py     # RAG subgraph (Pinecone + GPT-4o)",
        "│   ├── sql_agent.py     # NL2SQL subgraph (DuckDB + GPT-4o)",
        "│   ├── realtime_agent.py# Currency subgraph (ExchangeRate-API)",
        "│   └── memory_agent.py  # Memory subgraph (retrieve + extract facts)",
        "├── tools/",
        "│   ├── pinecone_retriever.py  # Pinecone upsert + query",
        "│   ├── nl2sql_engine.py       # DuckDB CSV loader + SQL executor",
        "│   ├── currency_tool.py       # Sync HTTP currency API client",
        "│   └── memory_store.py        # Unified memory interface",
        "├── memory/",
        "│   ├── short_term.py          # In-memory LangChain message buffer",
        "│   ├── long_term.py           # SQL Server user_facts table",
        "│   └── conversation_history.py# SQL Server conversation_history table",
        "├── security/",
        "│   ├── auth.py                # JWT create/verify, OAuth2",
        "│   ├── guardrails.py          # Injection detection + Presidio PII",
        "│   └── rate_limiter.py        # slowapi limiter instance",
        "├── ui/",
        "│   ├── app.py                 # Streamlit main entry point",
        "│   ├── components/",
        "│   │   ├── chat.py            # Chat history + input rendering",
        "│   │   ├── sidebar.py         # Login + settings sidebar",
        "│   │   └── agent_steps.py     # Agent reasoning steps display",
        "│   └── utils/api_client.py    # HTTP client to call FastAPI",
        "├── data/",
        "│   ├── company_policy.pdf     # Ingested into Pinecone",
        "│   └── csv_files/             # sales.csv, customers.csv, employees.csv, products.csv",
        "├── tests/",
        "│   ├── conftest.py            # Global mocks (SQLAlchemy, OpenAI, Pinecone)",
        "│   ├── unit/                  # 41 unit tests",
        "│   └── adversarial/           # Injection + PII leakage tests",
        "├── docker/docker-compose.yml  # Redis + Postgres containers",
        "├── ingest.py                  # PDF → Pinecone ingestion script",
        "├── .env                       # All secrets and configuration",
        "└── requirements.txt           # Python dependencies",
    ]
    for line in structure:
        elems.append(Paragraph(line, styles["code"]))
    elems.append(Spacer(1, 0.4*cm))

    # ── Section 2: API Reference ─────────────────────────────────────────────
    elems += section_header("2. API Reference", styles)

    apis = [
        ("GET /health", "No auth", "Returns API status, version, timestamp"),
        ("POST /auth/token", "No auth", "Body: {username, password} → Returns JWT access_token"),
        ("POST /chat", "Bearer JWT", "Body: {message, user_id, session_id} → Returns answer, agent_used, sources, latency_ms"),
    ]
    elems += make_table(["Endpoint", "Auth", "Description"], apis, styles,
        col_widths=[4*cm, 3*cm, 11.5*cm])

    elems.append(Paragraph("<b>Chat Request Schema:</b>", styles["h3"]))
    req_fields = [
        ("message", "str", "Required", "The user's question"),
        ("user_id", "str", "Required", "Unique user identifier"),
        ("session_id", "str", "Required", "Unique session identifier (UUID)"),
    ]
    elems += make_table(["Field", "Type", "Required", "Description"], req_fields, styles,
        col_widths=[3.5*cm, 2.5*cm, 3*cm, 9.5*cm])

    elems.append(Paragraph("<b>Chat Response Schema:</b>", styles["h3"]))
    res_fields = [
        ("answer", "str", "Final answer to the user"),
        ("agent_used", "str", "Which agents were called e.g. 'rag, sql'"),
        ("session_id", "str", "Echo of the session ID"),
        ("sources", "list[str]", "Source documents used by RAG agent"),
        ("tokens_used", "int", "Total tokens consumed"),
        ("latency_ms", "float", "End-to-end latency in milliseconds"),
        ("timestamp", "datetime", "UTC timestamp of the response"),
    ]
    elems += make_table(["Field", "Type", "Description"], res_fields, styles,
        col_widths=[4*cm, 3.5*cm, 11*cm])

    # ── Section 3: Environment Variables ────────────────────────────────────
    elems += section_header("3. Environment Variables (.env)", styles)
    env_vars = [
        ("OPENAI_API_KEY", "OpenAI API key for GPT-4o and embeddings"),
        ("PINECONE_API_KEY", "Pinecone vector DB API key"),
        ("PINECONE_INDEX_NAME", "Index name (default: chatbot-index)"),
        ("PINECONE_ENVIRONMENT", "AWS region (default: us-east-1)"),
        ("POSTGRES_URL", "SQLAlchemy connection string for SQL Server"),
        ("REDIS_URL", "Redis URL for rate limiting (default: redis://localhost:6379)"),
        ("JWT_SECRET_KEY", "HS256 signing secret (min 32 chars)"),
        ("JWT_ALGORITHM", "JWT algorithm (default: HS256)"),
        ("JWT_EXPIRE_MINUTES", "Token expiry in minutes (default: 30)"),
        ("EXCHANGE_RATE_API_KEY", "ExchangeRate-API key for live currency rates"),
        ("CSV_DIR", "Directory for CSV files (default: data/csv_files)"),
        ("LANGCHAIN_API_KEY", "LangSmith API key for tracing"),
        ("LANGCHAIN_PROJECT", "LangSmith project name"),
        ("LANGCHAIN_TRACING_V2", "Enable LangSmith tracing (true/false)"),
        ("CHAT_RATE_LIMIT", "Rate limit (default: 60/minute)"),
    ]
    elems += make_table(["Variable", "Description"], env_vars, styles,
        col_widths=[6*cm, 12.5*cm])

    # ── Section 4: Key Dependencies ──────────────────────────────────────────
    elems += section_header("4. Key Dependencies", styles)
    deps = [
        ("langgraph", "≥0.2", "Multi-agent graph orchestration framework"),
        ("langchain-openai", "≥0.1", "OpenAI LLM + embeddings integration"),
        ("langchain-community", "≥0.2", "PDF loaders, document utilities"),
        ("pinecone", "≥3.0", "Vector database client"),
        ("openai", "≥1.0", "Direct OpenAI API client"),
        ("fastapi", "≥0.110", "Async REST API framework"),
        ("uvicorn", "≥0.29", "ASGI server for FastAPI"),
        ("streamlit", "≥1.32", "Web UI framework"),
        ("duckdb", "≥0.10", "In-memory SQL engine for CSV queries"),
        ("sqlalchemy", "≥2.0", "ORM for SQL Server"),
        ("pyodbc", "≥5.0", "SQL Server ODBC driver"),
        ("python-jose", "≥3.3", "JWT token creation and verification"),
        ("presidio-analyzer", "≥2.2", "PII detection engine"),
        ("presidio-anonymizer", "≥2.2", "PII anonymization engine"),
        ("slowapi", "≥0.1", "Rate limiting for FastAPI"),
        ("httpx", "≥0.27", "Sync/async HTTP client for currency API"),
        ("python-dotenv", "≥1.0", "Load .env files into environment"),
        ("pypdf", "≥4.0", "PDF text extraction for ingestion"),
    ]
    elems += make_table(["Package", "Version", "Purpose"], deps, styles,
        col_widths=[4.5*cm, 2.5*cm, 11.5*cm])

    # ── Section 5: Database Schema ───────────────────────────────────────────
    elems += section_header("5. Database Schema (SQL Server)", styles)

    elems.append(Paragraph("<b>Table: user_facts</b>", styles["h3"]))
    uf = [
        ("id", "VARCHAR(255)", "PRIMARY KEY", "Composite: {user_id}_{key}"),
        ("user_id", "VARCHAR(255)", "NOT NULL, INDEX", "User identifier"),
        ("key", "VARCHAR(255)", "NOT NULL", "Fact name e.g. 'preferred_currency'"),
        ("value", "TEXT", "NOT NULL", "Fact value e.g. 'USD'"),
        ("created_at", "DATETIME", "DEFAULT NOW()", "When fact was first saved"),
        ("updated_at", "DATETIME", "DEFAULT NOW()", "When fact was last updated"),
    ]
    elems += make_table(["Column", "Type", "Constraint", "Description"], uf, styles,
        col_widths=[3.5*cm, 3.5*cm, 4*cm, 7.5*cm])

    elems.append(Paragraph("<b>Table: conversation_history</b>", styles["h3"]))
    ch = [
        ("id", "VARCHAR(36)", "PRIMARY KEY", "UUID auto-generated"),
        ("user_id", "VARCHAR(255)", "NOT NULL, INDEX", "User identifier"),
        ("session_id", "VARCHAR(255)", "NOT NULL, INDEX", "Session identifier"),
        ("role", "VARCHAR(20)", "NOT NULL", "'human' or 'ai'"),
        ("content", "TEXT", "NOT NULL", "Message text"),
        ("timestamp", "DATETIME", "DEFAULT NOW()", "When message was saved"),
    ]
    elems += make_table(["Column", "Type", "Constraint", "Description"], ch, styles,
        col_widths=[3.5*cm, 3.5*cm, 4*cm, 7.5*cm])

    # ── Section 6: How to Run ────────────────────────────────────────────────
    elems += section_header("6. How to Run (Development)", styles)

    run_steps = [
        ("1. Install dependencies", "pip install -r requirements.txt"),
        ("2. Configure .env", "Copy .env.example to .env and fill in all API keys"),
        ("3. Start Redis (Docker)", "docker-compose -f docker/docker-compose.yml up redis -d"),
        ("4. Ingest documents", "python ingest.py data/company_policy.pdf"),
        ("5. Start FastAPI", "uvicorn api.main:app --reload --port 8000"),
        ("6. Start Streamlit", "streamlit run ui/app.py"),
        ("7. Run tests", "pytest tests/unit/ -v --cov=agents"),
    ]
    for step, cmd in run_steps:
        elems.append(Paragraph(f"<b>{step}:</b>", styles["h3"]))
        elems.append(Paragraph(cmd, styles["code"]))

    doc.build(elems)
    print(f"PDF 2 saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PDF 3 — Interview Explanation & Theory
# ══════════════════════════════════════════════════════════════════════════════

def build_pdf3(path, styles):
    doc = SimpleDocTemplate(path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN)
    elems = []

    elems += cover_page(
        "Interview Explanation & Theory",
        "All Concepts, Questions & Model Answers for Interview Preparation",
        "3", styles)

    def qa(q, a):
        elems.append(Paragraph(f"Q: {q}", styles["question"]))
        data = [[Paragraph(f"A: {a}", ParagraphStyle("ans", fontSize=10,
            fontName="Helvetica", leading=15, textColor=colors.HexColor("#1b5e20")))]]
        t = Table(data, colWidths=[W - 2*MARGIN])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#e8f5e9")),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 12),
            ("RIGHTPADDING", (0,0), (-1,-1), 12),
            ("BOX", (0,0), (-1,-1), 0.5, GREEN),
        ]))
        elems.append(t)
        elems.append(Spacer(1, 0.25*cm))

    # ── Topic 1: LangGraph ──────────────────────────────────────────────────
    elems += section_header("1. LangGraph & Multi-Agent Architecture", styles)

    qa("What is LangGraph and why did you use it?",
       "LangGraph is a framework for building stateful, multi-agent AI applications as directed graphs. "
       "Each node is a Python function, edges define flow, and state is a TypedDict that flows through the graph. "
       "I used it because: (1) it gives full control over agent orchestration, (2) built-in checkpointing for "
       "fault tolerance, (3) supports conditional edges for human-in-the-loop, and (4) each agent is its own "
       "compiled subgraph making it modular and testable.")

    qa("What is a StateGraph and how does it work?",
       "StateGraph is the core LangGraph class. You define: (1) a TypedDict as the state schema, (2) nodes as "
       "Python functions that take state and return updated state, (3) edges connecting nodes, and (4) conditional "
       "edges that route based on state values. The graph is compiled into a runnable object. When you call "
       ".invoke(initial_state), LangGraph executes nodes in order, passing state between them, checkpointing "
       "after each node.")

    qa("What is the Supervisor pattern in multi-agent systems?",
       "The Supervisor pattern has one coordinator agent (the supervisor) that decides which specialist agents "
       "to call based on the user's question. The supervisor: (1) plans which agents are needed, (2) routes "
       "the question to the right agent(s), (3) collects results, (4) merges them into a final answer. "
       "In my project, the supervisor uses GPT-4o-mini to route to RAG, SQL, Realtime, or Memory agents.")

    qa("Explain the Map-Reduce pattern you implemented.",
       "Map-Reduce in multi-agent AI: MAP phase = the supervisor distributes the question to multiple agents "
       "in parallel using ThreadPoolExecutor. Each agent independently processes the question. "
       "REDUCE phase = the merge_node collects all agent results and uses GPT-4o to synthesize them into "
       "one coherent answer. Example: 'What is EUR/USD and Q3 revenue?' routes to both realtime AND sql agents "
       "simultaneously, then merges results.")

    qa("What is a subgraph in LangGraph?",
       "A subgraph is a compiled StateGraph that is called as a single unit by a parent graph. Each specialist "
       "agent (RAG, SQL, Realtime, Memory) is implemented as its own subgraph with its own state TypedDict and "
       "nodes. The supervisor's route_node calls run_rag_agent(), run_sql_agent() etc., which internally invoke "
       "the compiled subgraph. This gives modular, independently testable agents.")

    qa("What is MemorySaver and why do you use it?",
       "MemorySaver is LangGraph's built-in checkpointer that saves the full graph state to RAM after every node. "
       "It enables: (1) conversation resumption — if a request fails mid-graph, it can resume from the last "
       "checkpoint; (2) human-in-the-loop — the graph can pause at a node waiting for human approval; "
       "(3) per-session isolation via thread_id in the config — each session gets its own state thread.")

    # ── Topic 2: RAG ────────────────────────────────────────────────────────
    elems += section_header("2. RAG (Retrieval-Augmented Generation)", styles)

    qa("Explain your RAG pipeline end-to-end.",
       "1. INGESTION: PDF loaded with PyPDFLoader → split into 500-char chunks with 50-char overlap using "
       "RecursiveCharacterTextSplitter → each chunk embedded with OpenAI text-embedding-3-small (1536 dims) → "
       "upserted to Pinecone with metadata {text, source}. "
       "2. RETRIEVAL: User question embedded → Pinecone cosine similarity search → top-5 chunks returned. "
       "3. GRADING: Chunks with score < 0.75 filtered out (if all filtered, keep top-2). "
       "4. GENERATION: GPT-4o answers using only the retrieved context. "
       "5. REFLECTION: GPT-4o-mini scores answer quality 0.0-1.0, retries if < 0.7 (max 2 retries).")

    qa("Why did you use cosine similarity for vector search?",
       "Cosine similarity measures the angle between two vectors regardless of magnitude, making it ideal for "
       "semantic similarity of text embeddings. Two chunks about the same topic will have a high cosine "
       "similarity (close to 1.0) even if they have different lengths. Pinecone uses cosine by default for "
       "text-embedding models. Alternative: dot product (faster but magnitude-sensitive), Euclidean (less "
       "suitable for high-dimensional embeddings).")

    qa("What is chunk size and why does it matter?",
       "Chunk size (500 chars in my project) determines how large each piece of text stored in the vector DB is. "
       "Too small: chunks lose context, embeddings are less meaningful. Too large: chunks contain multiple "
       "topics, retrieval is less precise. Overlap (50 chars) ensures sentences split across chunk boundaries "
       "are still captured. The right size depends on document type — 500 chars works well for policy documents.")

    qa("What is the Reflection pattern in RAG?",
       "Reflection is a self-evaluation loop where the LLM critiques its own output. After generating an answer, "
       "a critic node (using GPT-4o-mini for speed) scores the answer on criteria: groundedness, completeness, "
       "clarity. If score < 0.7 and retry_count < 2, the graph loops back to regenerate. This implements "
       "the ReAct (Reason-Act-Observe) pattern iteratively. In my project, reflection is skipped on the first "
       "attempt for latency optimization and only activates on retries.")

    qa("What is Pinecone's namespace and why did you use it?",
       "A namespace in Pinecone is a logical partition within an index. Vectors in different namespaces are "
       "completely isolated — a query in namespace 'hr' only searches 'hr' vectors. I use namespace='default' "
       "but the architecture supports per-user or per-department namespaces for data isolation. This allows "
       "one Pinecone index to serve multiple tenants without data leakage.")

    # ── Topic 3: NL2SQL ─────────────────────────────────────────────────────
    elems += section_header("3. NL2SQL (Natural Language to SQL)", styles)

    qa("Explain your NL2SQL pipeline.",
       "1. INTENT: GPT-4o-mini reads available CSV filenames and selects which ones are needed for the question. "
       "2. LOAD: Selected CSVs loaded into DuckDB via pandas (pandas.read_csv → conn.register) — avoids file "
       "lock issues when Excel has the file open. "
       "3. SCHEMA: Column names, types, and 3 sample rows extracted to give LLM exact context. "
       "4. SQL GEN: GPT-4o generates DuckDB SQL using schema + question. "
       "5. REFLECTION: GPT-4o-mini validates SQL for correct columns, GROUP BY, JOIN syntax. "
       "6. EXECUTE: DuckDB executes SQL → returns DataFrame. "
       "7. FORMAT: GPT-4o converts DataFrame to natural language answer.")

    qa("Why DuckDB instead of a traditional database?",
       "DuckDB is an in-process analytical SQL database — no server needed, runs in RAM. Advantages: "
       "(1) reads CSVs directly into tables in milliseconds, (2) supports full SQL including complex JOINs "
       "and aggregations, (3) columnar storage = fast for analytics queries like SUM, GROUP BY, (4) zero "
       "setup — just conn = duckdb.connect(':memory:'). Perfect for ad-hoc CSV analysis in an AI agent.")

    qa("How do you handle multi-CSV JOINs?",
       "The understand_intent_node prompts GPT-4o-mini with ALL available CSV filenames and asks it to identify "
       "which files are needed. If the question needs data from multiple files (e.g., 'Show sales with customer "
       "names'), it returns ['sales.csv', 'customers.csv']. Both are loaded as separate DuckDB tables. "
       "The schema shows both tables to GPT-4o, which generates a JOIN query. Example: "
       "SELECT s.product, c.name FROM sales s JOIN customers c ON s.customer_id = c.id")

    # ── Topic 4: Memory Architecture ────────────────────────────────────────
    elems += section_header("4. Memory Architecture", styles)

    qa("Explain your 3-tier memory architecture.",
       "TIER 1 — Short-term (RAM): LangGraph MemorySaver checkpoints full graph state per thread_id. "
       "Also ShortTermMemory class holds last 20 messages as LangChain HumanMessage/AIMessage objects. "
       "Lost on server restart. "
       "TIER 2 — Long-term (SQL Server user_facts): Persistent key-value facts extracted by GPT-4o-mini "
       "from every conversation (name, role, company, preferences). Never deleted. "
       "TIER 3 — Conversation History (SQL Server conversation_history): Every human/AI message ever sent, "
       "with user_id, session_id, timestamp. Used to inject past context into new sessions.")

    qa("How do you extract user facts from conversations?",
       "After every chat exchange, memory_save_node() calls save_memory() which: (1) saves both messages to "
       "conversation_history, (2) calls GPT-4o-mini with the conversation + already-known facts, asking it "
       "to extract only NEW facts not already known. The LLM returns a JSON dict e.g. "
       "{\"name\": \"Priyabrat\", \"company\": \"TCS\"}. Each fact is saved to user_facts table with "
       "user_id as the partition key. Already-known facts are NOT duplicated.")

    # ── Topic 5: Security ───────────────────────────────────────────────────
    elems += section_header("5. Security & Guardrails", styles)

    qa("How does JWT authentication work in your system?",
       "POST /auth/token receives username+password → create_token() creates a JWT payload with {sub: username, "
       "exp: now+30min, iat: now} → signed with HS256 algorithm using JWT_SECRET_KEY → token returned to client. "
       "On every /chat request: FastAPI's OAuth2PasswordBearer extracts Bearer token from Authorization header → "
       "verify_token() decodes and validates signature + expiry → raises 401 if invalid. "
       "The Depends(get_current_user) pattern means FastAPI automatically calls this before the route handler.")

    qa("How does prompt injection detection work?",
       "check_input_safety() runs 20+ regex patterns against the lowercased input before any LLM call. "
       "Patterns cover: instruction overrides (ignore previous instructions, disregard, forget), identity "
       "hijacking (act as, pretend you are, you are now), jailbreaks (DAN mode, bypass restrictions), "
       "SQL injection (UNION SELECT, 1=1, DROP TABLE), XSS (&lt;script&gt;, onerror=), and SSTI ({{...}}, ${...}). "
       "If any pattern matches, raises HTTP 400 Bad Request before the question reaches any LLM.")

    qa("What is Presidio and how did you use it?",
       "Microsoft Presidio is an open-source PII detection and anonymization library. It uses NLP models to "
       "identify PII entities in text. I use it in two places: (1) INPUT: anonymize_pii_input() detects "
       "PERSON, EMAIL, PHONE, CREDIT_CARD in user messages and replaces them with placeholders before sending "
       "to LLM. (2) OUTPUT: anonymize_pii_output() scrubs EMAIL, PHONE, CREDIT_CARD from LLM responses "
       "(NOT PERSON — employee names from internal data are valid). "
       "Key design: PERSON is intentionally excluded from output scrubbing to allow employee name answers.")

    # ── Topic 6: Design Patterns ─────────────────────────────────────────────
    elems += section_header("6. AI Design Patterns Used", styles)

    patterns = [
        ("ReAct Pattern", "Reason-Act-Observe loop. Agent reasons about what to do, acts (calls tool/LLM), observes result, reasons again. Used in: RAG agent (retrieve→grade→generate→reflect), Realtime agent (parse→call tool→observe result→format)."),
        ("Reflection Pattern", "LLM critiques its own output and retries if quality is low. Used in: RAG agent (GPT-4o-mini scores answer 0.0-1.0, retries if <0.7), SQL agent (GPT-4o-mini validates SQL before execution). Prevents hallucinated or incorrect answers."),
        ("Map-Reduce Pattern", "Supervisor maps question to N agents running in parallel (ThreadPoolExecutor), then reduces all results into one answer (merge_node with GPT-4o). Enables handling complex multi-part questions efficiently."),
        ("Subgraph Pattern", "Each specialist agent is a compiled LangGraph subgraph with its own TypedDict state, nodes, and edges. The supervisor calls them as black boxes. Enables: independent development, testing, and deployment of each agent."),
        ("Human-in-the-Loop", "Graph pauses at a conditional edge (needs_human_approval) before executing sensitive operations (delete, drop, transfer). Returns to END for human review instead of auto-executing. LangGraph MemorySaver preserves state while waiting."),
        ("Lazy Singleton", "Agent graphs are compiled once on first call and reused (_rag_graph, _sql_graph etc.). Compilation is expensive (builds internal execution plan). Without this, every request would recompile all 4 subgraphs + supervisor."),
        ("Guardrail Pattern", "Security checks layered before and after LLM calls: input sanitization → LLM call → output sanitization. Defense-in-depth ensures even if one layer fails, others catch threats."),
    ]
    for name, desc in patterns:
        elems.append(Paragraph(f"<b>{name}</b>", styles["h3"]))
        elems.append(Paragraph(desc, styles["body"]))

    # ── Topic 7: Latency Optimization ───────────────────────────────────────
    elems += section_header("7. Performance & Latency Optimization", styles)

    qa("How did you reduce latency in the system?",
       "Multiple optimizations applied: "
       "(1) Model tiering: GPT-4o-mini for routing/reflection/classification (5x faster, cheaper), "
       "GPT-4o only for final answer generation. "
       "(2) Skip reflection on first pass: RAG agent skips the critic node on the first generate attempt, "
       "only activating if retry_count > 0. Saves 1 full LLM call per request. "
       "(3) Lazy singleton graphs: all 4 agent subgraphs compiled once at first call, reused thereafter. "
       "(4) Parallel agent execution: when multiple agents needed, ThreadPoolExecutor runs them simultaneously. "
       "(5) Pandas CSV loading: avoids file lock overhead and DuckDB file open errors.")

    qa("What would you do to reduce latency further?",
       "(1) Streaming: use FastAPI StreamingResponse + LangChain streaming callbacks to send tokens as they "
       "arrive instead of waiting for full response. "
       "(2) Response caching: cache identical questions in Redis with 5-min TTL. "
       "(3) Async agents: make agent nodes async to avoid blocking I/O. "
       "(4) Pinecone warm-up: pre-warm Pinecone connection at startup instead of first request. "
       "(5) Smaller embedding model: text-embedding-3-small already chosen (cheaper/faster than large). "
       "(6) Pre-compute SQL schemas: cache DuckDB table schemas at startup.")

    # ── Topic 8: Common Interview Questions ─────────────────────────────────
    elems += section_header("8. Common Interview Questions", styles)

    qa("How is this different from a simple chatbot?",
       "A simple chatbot has one LLM call per request with no specialization. This system has: "
       "(1) Multiple specialist agents each optimized for their domain, "
       "(2) Intelligent routing — the supervisor decides which agent(s) to use, "
       "(3) Tool use — SQL agent actually queries data, realtime agent makes live API calls, "
       "(4) Persistent memory — remembers users across sessions, "
       "(5) Reflection loops — agents self-critique and retry poor answers, "
       "(6) Production security — JWT, rate limiting, PII scrubbing, injection detection.")

    qa("What happens if an agent fails? How do you handle errors?",
       "Each agent has multiple error handling layers: "
       "(1) SQL agent: if SQL fails DuckDB execution, error is returned in state and format_node outputs a "
       "friendly message. (2) Realtime agent: has dedicated handle_error_node for API failures with fallback "
       "message. (3) RAG agent: if Pinecone returns no chunks, keeps top-2 anyway. (4) Supervisor: "
       "if JSON plan parsing fails, keyword-based fallback routing kicks in. (5) Memory: save failures "
       "are silent — don't break the main chat flow.")

    qa("How would you scale this to 1000 concurrent users?",
       "(1) Run multiple FastAPI instances behind a load balancer (nginx/AWS ALB). "
       "(2) Move Redis to a managed Redis cluster (AWS ElastiCache) for distributed rate limiting. "
       "(3) Use a connection pool for SQL Server (SQLAlchemy pool_size=20). "
       "(4) Move LangGraph checkpointing from MemorySaver (RAM) to PostgresSaver (DB-backed). "
       "(5) Use async FastAPI with async LangGraph invocation. "
       "(6) Cache Pinecone results for repeated questions. "
       "(7) Deploy agents as separate microservices with message queues (Redis Streams/Kafka).")

    qa("How do you evaluate the quality of your RAG responses?",
       "RAGAS framework (configured but can be run on demand): measures "
       "(1) Faithfulness — is the answer grounded in retrieved chunks? (threshold: 0.85) "
       "(2) Answer Relevancy — does the answer address the question? (threshold: 0.80) "
       "(3) Context Recall — were all relevant chunks retrieved? (threshold: 0.75) "
       "(4) Context Precision — are retrieved chunks relevant? (threshold: 0.75) "
       "LangSmith traces every LLM call with inputs, outputs, latency, and token counts for debugging.")

    qa("Why SQL Server instead of PostgreSQL?",
       "The user's local environment has SQL Server Express (PRIYABRAT\\SQLEXPRESS) with Windows "
       "Authentication already configured. The SQLAlchemy connection string uses pyodbc with the "
       "ODBC Driver 17 for SQL Server. The code is DB-agnostic via SQLAlchemy ORM — switching back "
       "to PostgreSQL requires only changing the POSTGRES_URL env var. String lengths were fixed to "
       "VARCHAR(255) for primary keys because SQL Server does not allow VARCHAR(MAX) as a PK column.")

    qa("What is the purpose of LangSmith in this project?",
       "LangSmith is Anthropic's LLM observability platform (by LangChain). With LANGCHAIN_TRACING_V2=true, "
       "every LLM call is automatically traced: inputs, outputs, latency, token usage, errors. In the "
       "supervisor, you can see the full multi-agent execution tree — which agents ran, what each node "
       "received and returned, where latency was spent. Critical for debugging incorrect agent routing "
       "and hallucinated SQL queries in production.")

    doc.build(elems)
    print(f"PDF 3 saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    styles = make_styles()
    base = "D:\\InterView\\All Code Hub\\production-chatbot\\data"

    build_pdf1(f"{base}\\01_Application_Flow_Architecture.pdf", styles)
    build_pdf2(f"{base}\\02_Complete_Documentation.pdf", styles)
    build_pdf3(f"{base}\\03_Interview_Explanation_Theory.pdf", styles)

    print("\nAll 3 PDFs generated successfully!")
    print(f"Location: {base}\\")
