# Funnel Intelligence RAG Platform

A production-grade, psychographic-aware Retrieval-Augmented Generation (RAG) system for generating data-backed funnel strategy recommendations. Built for marketing agencies who want to productize their expertise at scale.

## Architecture

Follows **CLEAN architecture** with dependencies pointing inward:

```
frameworks/          → UI, CLI, DB connection, LangGraph
interface_adapters/  → Concrete implementations (SQLAlchemy, Ollama, sentence-transformers)
use_cases/           → Orchestration (ingest, search, classify, generate)
domain/              → Pure business logic (models, rules, services)
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Ollama](https://ollama.com) installed and running

### 1. Clone & Setup

```bash
git clone <repo>
cd marketing-ai-kb-rag
cp .env.example .env
```

### 2. Start PostgreSQL + pgvector

```bash
docker-compose up -d postgres
```

### 3. Pull Ollama Models

```bash
python scripts/setup_ollama.py
```

### 4. Install Dependencies

```bash
pip install -e ".[dev]"
# or with uv:
uv pip install -e ".[dev]"
```

### 5. Run Migrations

```bash
alembic upgrade head
```

### 6. Ingest Blog Data

```bash
python scripts/ingest_data.py --heyflow
python scripts/ingest_data.py --perspective
```

### 7. Launch Streamlit UI

```bash
streamlit run app.py
```

### 8. Evaluate Retrieval Quality

```bash
python scripts/evaluate.py --top-k 5
```

## Project Structure

```
├── src/
│   ├── domain/                    # Pure business logic
│   │   ├── models.py              # Chunk, PsychographicProfile, StrategyResponse
│   │   ├── repositories.py        # Abstract interfaces
│   │   ├── services.py            # RRF fusion, outcome ranking, chunking
│   │   └── exceptions.py          # Domain errors
│   ├── use_cases/                 # Orchestration
│   │   ├── ingest_documents.py
│   │   ├── search_chunks.py
│   │   ├── classify_intent.py
│   │   ├── decompose_queries.py
│   │   └── generate_strategy.py
│   ├── interface_adapters/        # Concrete implementations
│   │   ├── repositories/          # SQLAlchemy + pgvector
│   │   ├── llm/                   # Ollama client + prompts
│   │   ├── embeddings/            # sentence-transformers
│   │   └── parsers/               # Markdown parser
│   └── frameworks/                # UI, CLI, DB, LangGraph
│       ├── config.py
│       ├── database.py
│       ├── streamlit_app.py
│       └── agent_graph.py
├── scripts/
│   ├── ingest_data.py             # CLI for document ingestion
│   ├── setup_ollama.py            # Pull required models
│   └── evaluate.py                # Retrieval quality evaluation
├── tests/
│   ├── unit/                      # Per-component tests
│   ├── integration/               # End-to-end pipeline tests
│   └── fixtures/                  # Golden dataset + sample docs
├── docs/
│   ├── ideas/                     # Refined idea one-pager
│   ├── spec.md                    # Engineering spec
│   ├── plan.md                    # Implementation plan
│   ├── task1_infrastructure.md    # Task documentation
│   ├── task2_storage_layer.md
│   ├── task3_ingestion_pipeline.md
│   ├── task4_retrieval_layer.md
│   ├── task5_intelligence_layer.md
│   ├── task6_generation_validation.md
│   ├── task7_streamlit_ui.md
│   └── task8_evaluation.md
├── data/
│   ├── heyflow_blog_export/       # 165+ English blog posts
│   └── perspective_blog_export/   # 30 German blog posts
├── app.py                         # Streamlit entry point
├── docker-compose.yml             # PostgreSQL + pgvector
├── pyproject.toml                 # Dependencies & tool config
└── alembic/                       # Database migrations
```

## Key Features

- **Psychographic Intent Classification:** Extracts 5 audience dimensions (risk tolerance, purchase cycle, tech savviness, age bracket, price sensitivity) from briefs
- **Query Decomposition:** Breaks complex briefs into 2–4 focused retrieval sub-queries
- **Hybrid Search:** Semantic (pgvector HNSW) + keyword (PostgreSQL full-text) with Reciprocal Rank Fusion
- **Outcome-Aware Re-Ranking:** Case studies with metrics surface higher than generic explainers
- **Structured Generation:** JSON-mode LLM output with strategy name, funnel type, key steps, rationale, and citations
- **Validation Layer:** Citation verification + faithfulness checking with 1-retry fallback
- **LangGraph Orchestration:** Stateful agent graph: classify → decompose → retrieve → generate

## Testing

```bash
# Unit tests (fast, mocked LLM)
pytest tests/unit/

# Integration tests (requires PostgreSQL)
pytest tests/integration/

# Smoke tests with real Ollama (optional, slow)
pytest --run-smoke
```

## Documentation

Each task includes detailed documentation covering:
- What was built and why
- Key architectural choices and trade-offs
- Production adjustments required for scale

See `docs/task1_infrastructure.md` through `docs/task8_evaluation.md`.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Database | PostgreSQL 15 + pgvector |
| ORM | SQLAlchemy 2.0 |
| Migrations | Alembic |
| Embeddings | sentence-transformers (multilingual) |
| LLM | Ollama (mistral:7b, llama3.2:3b) |
| Agent Orchestration | LangGraph |
| Testing | pytest |
| Code Quality | ruff, mypy |

## License

MIT
