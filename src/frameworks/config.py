"""Application configuration and settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://funnel_rag:funnel_rag_local@localhost:5432/funnel_rag",
)

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "mistral:7b")
OLLAMA_CLASSIFICATION_MODEL = os.getenv(
    "OLLAMA_CLASSIFICATION_MODEL", "llama3.2:3b"
)

# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Application
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
HEYFLOW_EXPORT_DIR = DATA_DIR / "heyflow_blog_export"
PERSPECTIVE_EXPORT_DIR = DATA_DIR / "perspective_blog_export"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)
