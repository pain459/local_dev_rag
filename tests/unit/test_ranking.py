from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from local_dev_rag.domain import MemoryItem, VectorHit

NOW = datetime(2026, 10, 1, tzinfo=UTC)
PROJECT = UUID(int=100)


def memory(number=1, **changes):
    values = dict(
        id=UUID(int=number),
        project_id=PROJECT,
        source_session_id=UUID(int=200),
        source_event_id=UUID(int=300),
        kind="decision",
        text=f"memory {number}",
        confidence=0.9,
        importance=0.5,
        state="active",
        curator_model="curator",
        created_at=NOW,
        updated_at=NOW,
    )
    return MemoryItem(**(values | changes))


@pytest.mark.parametrize("component", ["semantic", "importance", "recency", "overlap"])
def test_each_normalized_component_changes_real_order(component):
    # Removing a ranking component must change which evidence is selected first.
    from local_dev_rag.ranking import rank_memories

    first = memory(1, text="unrelated", importance=0.5)
    second = memory(2, text="other", importance=0.5)
    distances = [0.8, 0.8]
    if component == "semantic":
        distances[1] = 0.1
    elif component == "importance":
        second = memory(2, text="other", importance=1)
    elif component == "recency":
        first = memory(1, text="unrelated", created_at=NOW - timedelta(days=365))
    else:
        second = memory(2, text="ECONNREFUSED in load_widget")
    ranked = rank_memories(
        "ECONNREFUSED load_widget",
        [
            VectorHit(first, distances[0]),
            VectorHit(second, distances[1]),
        ],
        NOW,
    )
    assert ranked[0].memory.id == UUID(int=2)
    assert ranked[0].score_components[component] > ranked[1].score_components[component]
    assert all(0 <= value <= 1 for c in ranked for value in c.score_components.values())


def test_cosine_components_are_hand_normalized_and_weights_are_configurable():
    from local_dev_rag.config import RankingWeights
    from local_dev_rag.ranking import rank_memories

    ranked = rank_memories(
        "missing",
        [VectorHit(memory(importance=2), 1)],
        NOW,
        weights=RankingWeights(
            semantic=1, importance=0, recency=0, overlap=0, diversity=0, min_semantic_similarity=0
        ),
    )
    assert ranked[0].score == 0.5
    assert ranked[0].score_components == {
        "semantic": 0.5,
        "importance": 1,
        "recency": 1,
        "overlap": 0,
        "diversity": 1,
    }


def test_inactive_and_duplicate_evidence_never_reaches_context_candidates():
    from local_dev_rag.ranking import rank_memories

    active = memory(1, text="Keep WAL enabled")
    hits = [
        VectorHit(memory(2, text=" keep  wal enabled "), 0.2),
        VectorHit(active, 0),
        VectorHit(active, 0.3),
    ]
    hits += [
        VectorHit(memory(i + 3, state=state), 0)
        for i, state in enumerate(["deleted", "superseded", "rejected"])
    ]
    assert [c.memory.id for c in rank_memories("WAL", hits, NOW)] == [active.id]


def test_diversity_promotes_distinct_evidence_and_ties_ignore_input_order():
    from local_dev_rag.ranking import rank_memories

    hits = [
        VectorHit(memory(1, text="enable WAL database durability"), 0),
        VectorHit(memory(2, text="enable WAL database durability always"), 0.01),
        VectorHit(memory(3, text="retry network timeouts"), 0.02),
    ]
    assert [c.memory.id.int for c in rank_memories("task", hits, NOW)] == [1, 3, 2]
    ties = [VectorHit(memory(4, text="aaa"), 0), VectorHit(memory(5, text="bbb"), 0)]
    assert rank_memories("task", ties, NOW) == rank_memories("task", ties[::-1], NOW)


def test_future_dates_and_extreme_distances_stay_normalized():
    from local_dev_rag.ranking import rank_memories

    result = rank_memories("task", [VectorHit(memory(created_at=NOW + timedelta(days=1)), -2)], NOW)
    assert result[0].score_components["semantic"] == 1
    assert result[0].score_components["recency"] == 1


def test_default_relevance_rejects_orthogonal_recent_important_cosmetic_memory():
    from local_dev_rag.config import Settings
    from local_dev_rag.ranking import rank_memories

    config = Settings(_env_file=None)
    cosmetic = memory(text="Prefer a blue sidebar", importance=1)
    assert (
        rank_memories(
            "ECONNREFUSED load_widget",
            [VectorHit(cosmetic, 1)],
            NOW,
            weights=config.ranking_weights,
        )
        == []
    )


def test_relevance_floor_can_be_tuned_independently_of_ranking_bonuses():
    from local_dev_rag.config import RankingWeights
    from local_dev_rag.ranking import rank_memories

    cosmetic = VectorHit(memory(text="Prefer a blue sidebar", importance=1), 0.3)
    assert (
        rank_memories(
            "ECONNREFUSED load_widget",
            [cosmetic],
            NOW,
            weights=RankingWeights(min_semantic_similarity=0.8),
        )
        == []
    )
    assert (
        len(
            rank_memories(
                "ECONNREFUSED load_widget",
                [cosmetic],
                NOW,
                weights=RankingWeights(min_semantic_similarity=0.6),
            )
        )
        == 1
    )


def test_exact_error_identifier_overlap_is_eligible_without_semantic_similarity():
    from local_dev_rag.ranking import rank_memories

    evidence = memory(text="load_widget: retry after ECONNREFUSED")
    assert [
        c.memory for c in rank_memories("ECONNREFUSED load_widget", [VectorHit(evidence, 1)], NOW)
    ] == [evidence]
