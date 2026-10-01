"""Deterministic, normalized ranking of active historical evidence."""

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from math import isfinite

from local_dev_rag.config import RankingWeights
from local_dev_rag.domain import MemoryCandidate, VectorHit


def _tokens(text: str) -> set[str]:
    # Preserve identifier and error-string spelling, including dotted names and paths.
    return set(re.findall(r"[\w]+(?:[./:\-][\w]+)*", text))


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value)) if isfinite(value) else 0.0


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def rank_memories(
    query_text: str,
    hits: Sequence[VectorHit],
    now: datetime,
    *,
    weights: RankingWeights | None = None,
) -> list[MemoryCandidate]:
    weights = weights or RankingWeights()
    coefficients = {
        name: getattr(weights, name)
        for name in ("semantic", "importance", "recency", "overlap", "diversity")
    }
    total = sum(coefficients.values())
    query_tokens = _tokens(query_text)
    pending: list[tuple[VectorHit, dict[str, float], set[str]]] = []
    for hit in hits:
        if hit.memory.state != "active" or not isfinite(hit.distance):
            continue
        age = max(0.0, (_utc(now) - _utc(hit.memory.created_at)).total_seconds() / 86400)
        tokens = _tokens(hit.memory.text)
        # Orthogonal/unrelated memories cannot qualify through bonus factors alone.
        # Exact identifier/error overlap remains useful when embeddings miss a match.
        if not tokens & query_tokens and hit.distance > 1 - weights.min_semantic_similarity:
            continue
        pending.append(
            (
                hit,
                {
                    "semantic": _clamp(1 - hit.distance / 2),
                    "importance": _clamp(hit.memory.importance),
                    "recency": 2 ** (-age / weights.recency_half_life_days),
                    "overlap": len(tokens & query_tokens) / len(query_tokens)
                    if query_tokens
                    else 0,
                    "diversity": 1.0,
                },
                tokens,
            )
        )
    result: list[MemoryCandidate] = []
    accepted_tokens: list[set[str]] = []
    seen_ids: set[object] = set()
    seen_text: set[str] = set()
    while pending:
        for _, components, tokens in pending:
            components["diversity"] = 1 - max(
                (
                    len(tokens & old) / len(tokens | old) if tokens | old else 0
                    for old in accepted_tokens
                ),
                default=0,
            )

        def score(entry: tuple[VectorHit, dict[str, float], set[str]]) -> float:
            return sum(entry[1][name] * weight for name, weight in coefficients.items()) / total

        pending.sort(
            key=lambda entry: (
                -score(entry),
                str(entry[0].memory.id),
                entry[0].distance,
                entry[0].memory.text,
            )
        )
        selected = pending.pop(0)
        memory = selected[0].memory
        text_key = " ".join(memory.text.split()).casefold()
        if memory.id in seen_ids or text_key in seen_text:
            continue
        seen_ids.add(memory.id)
        seen_text.add(text_key)
        result.append(MemoryCandidate(memory, score(selected), selected[1]))
        accepted_tokens.append(selected[2])
    return result
