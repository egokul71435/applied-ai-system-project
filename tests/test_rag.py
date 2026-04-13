"""
Tests for the RAG pipeline — covers unit tests, adversarial edge cases,
and a mocked integration test representing the human-review checkpoint.

Human-review checkpoints are marked with # [HUMAN REVIEW] comments:
these are the cases where unexpected model behaviour should be flagged
to a human before shipping.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.recommender import load_songs
from src.rag import build_song_document, Retriever, SongDocument
from src.evaluator import (
    compute_diversity,
    compute_retrieval_relevance,
    compute_explanation_coverage,
    evaluate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_songs():
    return load_songs("data/songs.csv")


@pytest.fixture
def retriever(sample_songs):
    docs = [build_song_document(s) for s in sample_songs]
    return Retriever(docs)


# ---------------------------------------------------------------------------
# build_song_document
# ---------------------------------------------------------------------------

def test_build_song_document_text_contains_title(sample_songs):
    doc = build_song_document(sample_songs[0])
    assert sample_songs[0]["title"] in doc.text


def test_build_song_document_text_contains_genre(sample_songs):
    doc = build_song_document(sample_songs[0])
    assert sample_songs[0]["genre"] in doc.text


# ---------------------------------------------------------------------------
# Retriever — happy-path unit tests
# ---------------------------------------------------------------------------

def test_retriever_returns_k_results(retriever):
    results = retriever.retrieve("happy pop song", k=3)
    assert len(results) == 3


def test_retriever_result_types(retriever):
    results = retriever.retrieve("energetic rock", k=2)
    for doc, score in results:
        assert isinstance(doc, SongDocument)
        assert isinstance(score, float)


def test_retriever_scores_in_range(retriever):
    results = retriever.retrieve("chill acoustic guitar", k=5)
    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_retriever_top_result_is_most_similar(retriever):
    results = retriever.retrieve("chill acoustic", k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Retriever — adversarial edge cases  [HUMAN REVIEW]
# ---------------------------------------------------------------------------

def test_retriever_empty_query_does_not_crash(retriever):
    """[HUMAN REVIEW] Empty query — all similarity scores will be ~0;
    confirm results are still returned rather than raising an exception."""
    results = retriever.retrieve("", k=3)
    assert len(results) == 3


def test_retriever_nonsense_query_does_not_crash(retriever):
    """[HUMAN REVIEW] Gibberish input — should return results without crashing."""
    results = retriever.retrieve("xyzzy frobulate quux", k=3)
    assert len(results) == 3


def test_retriever_k_larger_than_corpus(retriever, sample_songs):
    """[HUMAN REVIEW] k > corpus size — should return all songs, not error."""
    results = retriever.retrieve("music", k=999)
    assert len(results) == len(sample_songs)


def test_retriever_single_word_query(retriever):
    results = retriever.retrieve("jazz", k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# Evaluator — unit tests
# ---------------------------------------------------------------------------

def test_retrieval_relevance_empty_list():
    assert compute_retrieval_relevance([]) == 0.0


def test_retrieval_relevance_non_zero():
    songs = [({"genre": "pop", "mood": "happy"}, 0.8)]
    assert compute_retrieval_relevance(songs) == 0.8


def test_diversity_single_song():
    songs = [({"genre": "pop", "mood": "happy"}, 0.9)]
    assert compute_diversity(songs) == 1.0


def test_diversity_all_same_genre_and_mood():
    songs = [
        ({"genre": "pop", "mood": "happy"}, 0.9),
        ({"genre": "pop", "mood": "happy"}, 0.7),
    ]
    assert compute_diversity(songs) < 1.0


def test_diversity_all_unique_genre_and_mood():
    songs = [
        ({"genre": "pop", "mood": "happy"}, 0.9),
        ({"genre": "rock", "mood": "intense"}, 0.7),
        ({"genre": "jazz", "mood": "relaxed"}, 0.5),
    ]
    assert compute_diversity(songs) == 1.0


def test_explanation_coverage_full_match():
    score = compute_explanation_coverage(
        "acoustic chill",
        "This acoustic song has a relaxed chill vibe.",
    )
    assert score == 1.0


def test_explanation_coverage_no_match():
    score = compute_explanation_coverage("jazz blues", "happy energetic pop track")
    assert score == 0.0


def test_explanation_coverage_empty_query():
    # Query made only of stopwords → coverage is 1.0 by convention
    score = compute_explanation_coverage("for the", "any explanation")
    assert score == 1.0


def test_evaluate_returns_required_keys():
    mock_result = {
        "query": "happy energetic pop",
        "retrieved": [({"genre": "pop", "mood": "happy"}, 0.85)],
        "generated_response": "A happy pop song with great energy.",
    }
    metrics = evaluate(mock_result)
    assert set(metrics.keys()) == {"retrieval_relevance", "diversity", "explanation_coverage"}


def test_evaluate_scores_in_range():
    mock_result = {
        "query": "happy energetic pop",
        "retrieved": [({"genre": "pop", "mood": "happy"}, 0.85)],
        "generated_response": "A happy pop song with great energy.",
    }
    metrics = evaluate(mock_result)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


# ---------------------------------------------------------------------------
# RAGRecommender — mocked integration test  [HUMAN REVIEW]
# ---------------------------------------------------------------------------

def test_rag_recommender_full_pipeline(sample_songs):
    """[HUMAN REVIEW] End-to-end smoke test with a mocked Claude response.
    A human should inspect real API outputs for tone, accuracy, and hallucination
    before this system is used in production."""
    from src.rag import RAGRecommender

    mock_msg = MagicMock()
    mock_msg.content = [
        MagicMock(text="I recommend Sunrise City — its upbeat pop energy matches perfectly.")
    ]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = mock_msg

        rag = RAGRecommender(sample_songs)
        result = rag.recommend("happy pop music for a party", k=3)

    assert result["query"] == "happy pop music for a party"
    assert len(result["retrieved"]) == 3
    assert "Sunrise City" in result["generated_response"]

    metrics = evaluate(result)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
