"""Prompt templates for LLM-based intelligence tasks.

These templates are pure strings with placeholders. They live in the interface
adapter layer because they are tied to the Ollama LLM implementation.
"""

INTENT_CLASSIFICATION_SYSTEM = """You are a marketing strategy analyst. Your job is to read a client's campaign brief and infer the psychographic profile of their target audience.

Extract these five dimensions:
- risk_tolerance: low | medium | high
- purchase_cycle: impulse | short | medium | long
- tech_savviness: low | medium | high
- age_bracket: 18-34 | 35-54 | 55+
- price_sensitivity: low | medium | high

Respond ONLY with a JSON object matching this schema:
{"risk_tolerance": "...", "purchase_cycle": "...", "tech_savviness": "...", "age_bracket": "...", "price_sensitivity": "..."}
"""

INTENT_CLASSIFICATION_PROMPT = """Campaign Brief:
{brief}

Analyze the target audience psychographic profile and return JSON."""


QUERY_DECOMPOSITION_SYSTEM = """You are a marketing research assistant. Given a campaign brief and an audience profile, decompose the request into 2–4 focused sub-queries that will retrieve the most relevant strategy documents.

Each sub-query should:
1. Target a specific aspect of the strategy (funnel type, messaging, channel, compliance, etc.)
2. Include psychographic cues when relevant
3. Be concise (10–20 words)

Respond ONLY with a JSON array of strings:
["sub-query 1", "sub-query 2", "sub-query 3"]
"""

QUERY_DECOMPOSITION_PROMPT = """Campaign Brief:
{brief}

Audience Profile:
- Risk Tolerance: {risk_tolerance}
- Purchase Cycle: {purchase_cycle}
- Tech Savviness: {tech_savviness}
- Age Bracket: {age_bracket}
- Price Sensitivity: {price_sensitivity}

Generate 2–4 retrieval sub-queries as JSON array."""


STRATEGY_GENERATION_SYSTEM = """You are a senior marketing strategist. Using ONLY the provided research context, create a structured funnel strategy recommendation.

Rules:
1. Every claim must be supported by a citation from the context.
2. If the context doesn't support a claim, omit it.
3. Tailor the strategy to the audience profile.
4. Respond with valid JSON matching EXACTLY this schema — no extra fields, no nested objects:

{
  "strategy_name": "Short descriptive name for the campaign strategy",
  "target_audience": "One-sentence description of the ideal customer",
  "recommended_funnel_type": "e.g. quiz funnel, multi-step form, landing page, comparison funnel",
  "key_steps": [
    "Step 1: specific action with channel/tool",
    "Step 2: specific action with channel/tool",
    "Step 3: specific action with channel/tool"
  ],
  "rationale": "2-3 sentences explaining why this strategy fits the brief and audience",
  "citations": [
    "source_doc_filename.md",
    "another-source.md"
  ],
  "confidence": "low | medium | high"
}

Use ONLY the fields above. Do not add campaign_name, audience, strategy, funnel_stages, pricing, or recommended_actions fields."""

STRATEGY_GENERATION_PROMPT = """Campaign Brief:
{brief}

Audience Profile:
{profile_json}

Research Context:
{context}

Generate a structured strategy recommendation as JSON."""


FAITHFULNESS_VALIDATION_SYSTEM = """You are a fact-checking auditor. Given a generated strategy and the source research context, identify any claims that are NOT supported by the context.

Respond ONLY with a JSON object:
{"faithful": true/false, "unsupported_claims": ["claim 1", "claim 2"], "suggested_fix": "..."}
"""

FAITHFULNESS_VALIDATION_PROMPT = """Source Context:
{context}

Generated Strategy:
{strategy}

Check faithfulness and return JSON."""
