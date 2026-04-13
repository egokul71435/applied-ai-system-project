"""
Evaluation module for RAG recommendation quality.

Metrics (all return 0.0–1.0):
  - retrieval_relevance   : average cosine similarity of retrieved songs
  - diversity             : genre + mood variety across retrieved songs
  - explanation_coverage  : fraction of meaningful query words in the explanation
"""

from typing import List, Dict, Any, Tuple


def compute_retrieval_relevance(retrieved: List[Tuple[Dict, float]]) -> float:
    """Average cosine similarity score of retrieved songs."""
    if not retrieved:
        return 0.0
    return round(sum(score for _, score in retrieved) / len(retrieved), 3)


def compute_diversity(retrieved: List[Tuple[Dict, float]]) -> float:
    """
    Genre + mood diversity in the retrieved set.
    1.0 means every song has a unique genre AND unique mood.
    0.0 means all songs share the same genre and mood.
    """
    if len(retrieved) <= 1:
        return 1.0
    n = len(retrieved)
    genres = {song["genre"] for song, _ in retrieved}
    moods = {song["mood"] for song, _ in retrieved}
    return round((len(genres) / n + len(moods) / n) / 2, 3)


def compute_explanation_coverage(query: str, explanation: str) -> float:
    """
    Fraction of meaningful query words that appear somewhere in the explanation.
    Stopwords are excluded so common words don't inflate the score.
    """
    stopwords = {
        "i", "want", "a", "an", "some", "for", "the", "and", "or",
        "with", "that", "is", "me", "my", "to", "of", "in", "give",
    }
    query_keywords = set(query.lower().split()) - stopwords
    if not query_keywords:
        return 1.0
    explanation_words = set(explanation.lower().split())
    matched = query_keywords & explanation_words
    return round(len(matched) / len(query_keywords), 3)


def evaluate(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Run all three metrics on a RAG result dict and return a scores dict.

    Expected result shape:
        {
          "query": str,
          "retrieved": List[Tuple[song_dict, float]],
          "generated_response": str,
        }
    """
    return {
        "retrieval_relevance": compute_retrieval_relevance(result["retrieved"]),
        "diversity": compute_diversity(result["retrieved"]),
        "explanation_coverage": compute_explanation_coverage(
            result["query"], result["generated_response"]
        ),
    }
