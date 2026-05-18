"""SQLAlchemy database configuration — framework layer."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from frameworks.config import DATABASE_URL

# Engine with production-ready connection pooling defaults.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_timeout=30,
    connect_args={"connect_timeout": 10},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative ORM models.
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for dependency injection.

    Usage (Streamlit / CLI contexts):
        with get_db() as db:
            repo = SQLAlchemyChunkRepo(db)
            repo.save_all(chunks)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
