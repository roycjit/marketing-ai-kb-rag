#!/usrusr/bin/env python3
"""Evaluation script for the RAG pipeline.

Computes retrieval quality metrics against a golden dataset:
- Precision@K
- Mean Reciprocal Rank (MRR)
- Source Hit Rate

Usage:
    uv run python scripts/evaluate.py --dataset tests/fixtures/golden_dataset.jsonl --top-k 5
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List

import structlog

from frameworks.database import SessionLocal
from interface_adapters.embeddings.sentence_transformer_client import (
    SentenceTransformerEmbedder,
)
from interface_adapters.repositories.pgvector_search_repo import (
    PgVectorSearchRepository,
)
from use_cases.search_chunks import HybridSearchUseCase

logger = structlog.get_logger(__name__)


def load_dataset(path: Path) -> List[dict]:
    """Load golden dataset from JSONL file."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_precision_at_k(results: List, expected: set, k: int) -> float:
    """Compute Precision@K."""
    top_k = results[:k]
    relevant = sum(1 for r in top_k if Path(r.chunk.source_doc).name in expected)
    return relevant / k


def compute_mrr(results: List, expected: set) -> float:
    """Compute Mean Reciprocal Rank."""
    for rank, result in enumerate(results, start=1):
        if Path(result.chunk.source_doc).name in expected:
            return 1.0 / rank
    return 0.0


def compute_source_hit_rate(results: List, expected: set) -> float:
    """Compute what fraction of expected sources appear in results."""
    found = {Path(r.chunk.source_doc).name for r in results}
    hits = expected & found
    return len(hits) / len(expected) if expected else 0.0


def evaluate(dataset_path: Path, top_k: int) -> dict:
    """Run evaluation and return metrics."""
    logger.info("evaluation_started", dataset=str(dataset_path), top_k=top_k)

    dataset = load_dataset(dataset_path)
    db = SessionLocal()

    try:
        search_repo = PgVectorSearchRepository(db)
        embedder = SentenceTransformerEmbedder()
        search_uc = HybridSearchUseCase(search_repo, embedder)

        precisions = []
        mrrs = []
        hit_rates = []

        for record in dataset:
            query = record["query"]
            expected = set(record.get("expected_sources", []))
            description = record.get("description", query[:50])

            logger.info("evaluating_query", query=query[:60], description=description)

            try:
                results = search_uc.execute(query, top_k=top_k * 2)
            except Exception as exc:
                logger.error("query_failed", query=query, error=str(exc))
                db.rollback()
                continue

            p_at_k = compute_precision_at_k(results, expected, top_k)
            mrr = compute_mrr(results, expected)
            hit = compute_source_hit_rate(results, expected)

            precisions.append(p_at_k)
            mrrs.append(mrr)
            hit_rates.append(hit)

            logger.info(
                "query_result",
                precision_at_k=round(p_at_k, 2),
                mrr=round(mrr, 2),
                hit_rate=round(hit, 2),
            )

    finally:
        db.close()

    if not precisions:
        logger.warning("no_evaluable_queries")
        return {}

    report = {
        "queries_evaluated": len(precisions),
        f"precision_at_{top_k}": {
            "mean": round(statistics.mean(precisions), 3),
            "median": round(statistics.median(precisions), 3),
            "min": round(min(precisions), 3),
            "max": round(max(precisions), 3),
        },
        "mrr": {
            "mean": round(statistics.mean(mrrs), 3),
            "median": round(statistics.median(mrrs), 3),
            "min": round(min(mrrs), 3),
            "max": round(max(mrrs), 3),
        },
        "source_hit_rate": {
            "mean": round(statistics.mean(hit_rates), 3),
            "median": round(statistics.median(hit_rates), 3),
            "min": round(min(hit_rates), 3),
            "max": round(max(hit_rates), 3),
        },
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/fixtures/golden_dataset.jsonl"),
        help="Path to golden dataset JSONL file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="K value for Precision@K",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evaluation_report.md"),
        help="Path to write markdown report",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    report = evaluate(args.dataset, args.top_k)

    if not report:
        print("No queries could be evaluated.", file=sys.stderr)
        return 1

    # Print to console
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    # Write markdown report
    md = f"""# Evaluation Report

Generated: {__import__('datetime').datetime.now().isoformat()}

## Configuration

- Dataset: `{args.dataset}`
- Queries evaluated: {report['queries_evaluated']}
- Top-K: {args.top_k}

## Results

### Precision@{args.top_k}

| Metric | Value |
|--------|-------|
| Mean   | {report[f'precision_at_{args.top_k}']['mean']} |
| Median | {report[f'precision_at_{args.top_k}']['median']} |
| Min    | {report[f'precision_at_{args.top_k}']['min']} |
| Max    | {report[f'precision_at_{args.top_k}']['max']} |

### Mean Reciprocal Rank (MRR)

| Metric | Value |
|--------|-------|
| Mean   | {report['mrr']['mean']} |
| Median | {report['mrr']['median']} |
| Min    | {report['mrr']['min']} |
| Max    | {report['mrr']['max']} |

### Source Hit Rate

| Metric | Value |
|--------|-------|
| Mean   | {report['source_hit_rate']['mean']} |
| Median | {report['source_hit_rate']['median']} |
| Min    | {report['source_hit_rate']['min']} |
| Max    | {report['source_hit_rate']['max']} |

## Interpretation

- **Precision@{args.top_k}**: Proportion of top-{args.top_k} results that are relevant.
  Target: >0.60 (MVP), >0.80 (production).
- **MRR**: How close the first relevant result is to the top.
  Target: >0.70.
- **Source Hit Rate**: Fraction of expected sources that appear anywhere in results.
  Target: >0.80.

## Next Steps

1. Review low-performing queries and add missing content to knowledge base.
2. Tune hybrid search weights (RRF k, outcome boost coefficients).
3. Add question-variant indexing to bridge vocabulary gaps.
4. Evaluate generation quality (faithfulness, citation accuracy) separately.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"\n✓ Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
