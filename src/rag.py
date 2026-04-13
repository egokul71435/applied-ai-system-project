"""
RAG (Retrieval-Augmented Generation) pipeline for VibeMatcher.

Components:
  - SongDocument   : text description of a song for TF-IDF indexing
  - Retriever      : TF-IDF + cosine similarity over the song catalog
  - RAGRecommender : orchestrates retrieval → augmented prompt → Claude generation

Reliability features:
  - Structured logging (INFO for normal flow, WARNING for low-confidence results)
  - Error handling around the Claude API call with a graceful fallback message
  - confidence score attached to every result (composite of retrieval + diversity)
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

logger = logging.getLogger(__name__)


@dataclass
class SongDocument:
    """A song dict paired with a text description for retrieval."""
    song: Dict
    text: str


def build_song_document(song: Dict) -> SongDocument:
    """Convert a song dict (from load_songs) into a searchable text chunk."""
    energy_desc = (
        "high energy" if song["energy"] > 0.7
        else ("low energy" if song["energy"] < 0.4 else "medium energy")
    )
    acoustic_desc = "acoustic" if song["acousticness"] > 0.5 else "electronic"
    dance_desc = "danceable" if song["danceability"] > 0.6 else "not very danceable"
    valence_desc = "positive uplifting" if song["valence"] > 0.5 else "melancholic"

    text = (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"{energy_desc} {acoustic_desc} song. "
        f"Tempo: {song['tempo_bpm']} BPM. {dance_desc}. "
        f"{valence_desc} feeling."
    )
    return SongDocument(song=song, text=text)


class Retriever:
    """
    Retrieves songs relevant to a natural language query using TF-IDF
    representations of song descriptions and cosine similarity ranking.
    """

    def __init__(self, documents: List[SongDocument]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        corpus = [doc.text for doc in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        logger.info("Retriever indexed %d song documents.", len(documents))

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[SongDocument, float]]:
        """
        Return the top-k most relevant SongDocuments and their similarity scores.
        If k > corpus size, returns all documents.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        k = min(k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:k]
        results = [(self.documents[i], float(similarities[i])) for i in top_indices]

        top_score = results[0][1] if results else 0.0
        if top_score < 0.05:
            logger.warning(
                "Low retrieval confidence (top similarity=%.3f) for query: %r — "
                "results may not be relevant.",
                top_score, query,
            )
        else:
            logger.info(
                "Retrieved %d songs for query %r (top similarity=%.3f).",
                len(results), query, top_score,
            )
        return results


def _compute_confidence(retrieved: List[Tuple[SongDocument, float]]) -> float:
    """
    Composite confidence score (0.0–1.0) based on:
      - average retrieval similarity (how well songs matched the query)
      - genre diversity of the top results (avoid filter-bubble responses)
    """
    if not retrieved:
        return 0.0
    avg_sim = sum(s for _, s in retrieved) / len(retrieved)
    genres = {doc.song["genre"] for doc, _ in retrieved}
    diversity = len(genres) / len(retrieved)
    return round((avg_sim + diversity) / 2, 3)


class RAGRecommender:
    """
    Full RAG pipeline:
      1. Retrieve top-k song documents for the user's query
      2. Build an augmented prompt with those songs as context
      3. Call Claude to generate a natural language recommendation
      4. Attach a confidence score to the result
    """

    def __init__(self, songs: List[Dict]):
        documents = [build_song_document(s) for s in songs]
        self.retriever = Retriever(documents)
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def recommend(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Given a natural language query, return retrieved songs, a Claude-generated
        recommendation, and a confidence score.

        Returns:
            {
              "query": str,
              "retrieved": List[Tuple[song_dict, similarity_score]],
              "generated_response": str,
              "confidence": float,   # 0.0–1.0 composite score
            }
        """
        logger.info("RAG recommend called with query: %r (k=%d)", query, k)
        retrieved = self.retriever.retrieve(query, k=k)
        confidence = _compute_confidence(retrieved)

        context_lines = []
        for doc, score in retrieved:
            s = doc.song
            context_lines.append(
                f'- "{s["title"]}" by {s["artist"]} '
                f'(genre: {s["genre"]}, mood: {s["mood"]}, '
                f'energy: {s["energy"]}, acousticness: {s["acousticness"]:.2f}, '
                f'similarity: {score:.2f})'
            )
        context = "\n".join(context_lines)

        prompt = (
            f'A user asked: "{query}"\n\n'
            f"The most relevant songs retrieved from the catalog:\n{context}\n\n"
            f"Recommend the top songs from this list. For each, write 1-2 sentences "
            f"explaining why it fits the request, referencing specific musical qualities."
        )

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=(
                    "You are a concise music recommendation assistant. "
                    "Only recommend songs from the provided list. "
                    "Be specific about musical qualities like mood, energy, and genre."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            generated = message.content[0].text
            logger.info(
                "Generation complete. Confidence=%.3f. Response length=%d chars.",
                confidence, len(generated),
            )
        except anthropic.APIError as exc:
            logger.error("Claude API error for query %r: %s", query, exc)
            generated = (
                "Sorry, the recommendation service is temporarily unavailable. "
                "Please try again later."
            )

        return {
            "query": query,
            "retrieved": [(doc.song, score) for doc, score in retrieved],
            "generated_response": generated,
            "confidence": confidence,
        }
