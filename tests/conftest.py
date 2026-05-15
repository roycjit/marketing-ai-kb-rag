"""Shared pytest fixtures and configuration."""

import os
from typing import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ensure we don't accidentally hit a real database during tests
os.environ.setdefault("ENVIRONMENT", "test")

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg2://funnel_rag:funnel_rag_local@localhost:5432/funnel_rag_test",
)


@pytest.fixture(scope="session")
def engine():
    """Create a test database engine."""
    eng = create_engine(TEST_DATABASE_URL)
    # Ensure pgvector extension exists
    with eng.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    yield eng
    eng.dispose()


@pytest.fixture(scope="function")
def db_session(engine) -> Generator:
    """Provide a transactional database session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
