"""End-to-end integration test: full pipeline from brief to strategy.

This test exercises all layers together with a real PostgreSQL database
but mocked LLM calls for speed and determinism.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from domain.models import Chunk
from frameworks.agent_graph import FunnelAgentGraph
from frameworks.database import SessionLocal
from interface_adapters.embeddings.sentence_transformer_client import (
    SentenceTransformerEmbedder,
)
from interface_adapters.llm.ollama_client import OllamaClient
from interface_adapters.repositories.pgvector_search_repo import (
    PgVectorSearchRepository,
)
from interface_adapters.repositories.sqlalchemy_chunk_repo import (
    SQLAlchemyChunkRepository,
)
from use_cases.classify_intent import ClassifyIntentUseCase
from use_cases.decompose_queries import DecomposeQueriesUseCase
from use_cases.generate_strategy import GenerateStrategyUseCase
from use_cases.search_chunks import HybridSearchUseCase


@pytest.fixture
def seeded_db():
    """Seed the test database with sample chunks."""
    db = SessionLocal()
    repo = SQLAlchemyChunkRepository(db)

    chunks = [
        Chunk(
            chunk_id="e2e-001",
            content="Solar savings calculators achieve 150% conversion lift for 1KOMMA5°.",
            embedding=[1.0] + [0.0] * 383,
            source_doc="case-study-1komma5grad.md",
            doc_type="case_study",
            doc_subtype="case_study",
            last_updated=datetime(2024, 12, 16, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.95,
            summary="1KOMMA5° case study with conversion metrics",
        ),
        Chunk(
            chunk_id="e2e-002",
            content="Renewable energy funnels need mobile-first design and TCPA compliance.",
            embedding=[0.9] + [0.1] * 383,
            source_doc="funnel-builder-renewable-energy-sector.md",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2026, 3, 10, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.7,
            summary="Guide to renewable energy funnel best practices",
        ),
        Chunk(
            chunk_id="e2e-003",
            content="Retirees respond best to trust signals, testimonials, and financing options.",
            embedding=[0.8] + [0.2] * 383,
            source_doc="funnel-builder-renewable-energy-sector.md",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2026, 3, 10, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.65,
            summary="Audience-specific advice for older homeowners",
        ),
    ]
    repo.save_all(chunks)
    yield db
    # Cleanup
    db.rollback()
    db.close()


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns deterministic JSON responses."""
    client = OllamaClient.__new__(OllamaClient)
    call_count = [0]

    def fake_generate(*, model, prompt, system=None, temperature=0.7, max_tokens=256, json_mode=False):
        call_count[0] += 1
        # Classification
        if "psychographic" in system.lower() or "marketing strategy analyst" in system.lower():
            return json.dumps({
                "risk_tolerance": "low",
                "purchase_cycle": "long",
                "tech_savviness": "low",
                "age_bracket": "55+",
                "price_sensitivity": "high",
            })
        # Decomposition
        if "research assistant" in system.lower() or "sub-queries" in system.lower():
            return json.dumps([
                "solar calculator funnel for retirees",
                "trust signals for cautious buyers",
            ])
        # Generation
        if "senior marketing strategist" in system.lower() or "strategy recommendation" in system.lower():
            return json.dumps({
                "strategy_name": "Solar Retiree Trust Funnel",
                "target_audience": "Risk-averse homeowners 55+",
                "recommended_funnel_type": "Multi-step qualification with savings calculator",
                "key_steps": ["ROI calculator", "Video testimonials", "Financing pre-qualification"],
                "rationale": "Retirees need quantitative proof and social proof before committing to high-ticket solar.",
                "citations": ["case-study-1komma5grad.md", "funnel-builder-renewable-energy-sector.md"],
                "confidence": "high",
            })
        # Validation
        if "fact-checking" in system.lower() or "faithfulness" in system.lower():
            return json.dumps({"faithful": True, "unsupported_claims": [], "suggested_fix": ""})

        return json.dumps({})

    client.generate = fake_generate
    client.chat = lambda **kwargs: "mock response"
    return client


class TestEndToEnd:
    def test_full_pipeline(self, seeded_db, mock_llm):
        """Run the complete agent graph and verify output structure."""
        search_repo = PgVectorSearchRepository(seeded_db)
        embedder = SentenceTransformerEmbedder()

        classify_uc = ClassifyIntentUseCase(mock_llm)
        decompose_uc = DecomposeQueriesUseCase(mock_llm)
        search_uc = HybridSearchUseCase(search_repo, embedder)
        generate_uc = GenerateStrategyUseCase(mock_llm)

        graph = FunnelAgentGraph(classify_uc, decompose_uc, search_uc, generate_uc)
        result = graph.run("Sell PV systems to cautious retirees in Germany")

        # Verify profile
        assert result["profile"] is not None
        assert result["profile"].age_bracket == "55+"
        assert result["profile"].risk_tolerance == "low"

        # Verify sub-queries
        assert len(result["sub_queries"]) >= 1

        # Verify search results
        assert len(result["search_results"]) > 0
        sources = {r.chunk.source_doc for r in result["search_results"]}
        assert "case-study-1komma5grad.md" in sources

        # Verify strategy
        strategy = result["strategy"]
        assert strategy is not None
        assert strategy.strategy_name == "Solar Retiree Trust Funnel"
        assert len(strategy.key_steps) >= 2
        assert "case-study-1komma5grad.md" in strategy.citations
