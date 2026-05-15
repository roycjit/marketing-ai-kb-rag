"""SQLAlchemy database configuration — framework layer."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from frameworks.config import DATABASE_URL

# Engine with sensible connection pooling defaults for local development.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative ORM models.
Base = declarative_base()


def get_db():
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
