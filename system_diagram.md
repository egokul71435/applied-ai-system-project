# VibeMatcher RAG System Diagram

## Overview

The RAG (Retrieval-Augmented Generation) layer sits alongside the existing
weighted-scoring recommender. It accepts **natural language queries** instead
of structured `UserProfile` objects, retrieves the most relevant songs from
the catalog, then passes them as grounded context to Claude to generate rich,
conversational explanations.

---

## Component Diagram (Mermaid)

```mermaid
flowchart TD
    U([User]) -->|natural language query\ne.g. 'chill acoustic music to study'| Q[Query Input]

    subgraph KB [Knowledge Base]
        direction TB
        CSV[(songs.csv\n18 songs)]
        DOCS[Song Documents\ntext descriptions]
        CSV -->|build_song_document| DOCS
    end

    subgraph RETRIEVER [Retriever · src/rag.py]
        direction TB
        TFIDF[TF-IDF Vectorizer\nfit on song descriptions]
        COS[Cosine Similarity\nquery vs. all songs]
        TOPK[Top-K Songs\n+ similarity scores]
        TFIDF --> COS --> TOPK
    end

    subgraph GENERATOR [Generator · src/rag.py]
        direction TB
        PROMPT[Augmented Prompt\nquery + retrieved context]
        LLM[Claude claude-sonnet-4-6\nRAGRecommender.recommend]
        RESP[Natural Language\nRecommendation]
        PROMPT --> LLM --> RESP
    end

    subgraph EVALUATOR [Evaluator · src/evaluator.py]
        direction TB
        M1[retrieval_relevance\navg cosine score]
        M2[diversity\ngenre + mood spread]
        M3[explanation_coverage\nquery keyword overlap]
    end

    subgraph HUMAN [Human-in-the-Loop · tests/test_rag.py]
        direction TB
        ADV[Adversarial Cases\nempty query · nonsense · k > corpus]
        MOCK[Mocked Integration Test\nhuman reviews real API output]
        FLAG[Edge Case Flag\nfails evaluation threshold]
    end

    Q --> TFIDF
    DOCS --> TFIDF
    TOPK --> PROMPT
    Q --> PROMPT
    RESP --> M1
    RESP --> M2
    RESP --> M3
    TOPK --> M1
    TOPK --> M2

    M1 & M2 & M3 --> THRESH{All metrics\npass threshold?}
    THRESH -->|Yes| OUT([Output to User])
    THRESH -->|No| FLAG
    FLAG --> HUMAN
    HUMAN -->|tune retriever or prompt| RETRIEVER
    HUMAN -->|expand catalog| KB

    style KB fill:#e8f5e9,stroke:#388e3c
    style RETRIEVER fill:#e3f2fd,stroke:#1976d2
    style GENERATOR fill:#fff8e1,stroke:#f9a825
    style EVALUATOR fill:#fff3e0,stroke:#ef6c00
    style HUMAN fill:#fce4ec,stroke:#c62828
```

---

## Data Flow (text summary)

```
INPUT
  User types: "I want something energetic and happy for working out"
       │
       ▼
KNOWLEDGE BASE
  songs.csv ──► build_song_document() ──► 18 SongDocument text chunks
  (loaded once at startup; re-used across queries)
       │
       ▼
RETRIEVER  (src/rag.py · Retriever)
  TF-IDF vectorizer encodes all 18 song descriptions + the user query
  Cosine similarity ranks every song against the query
  Top-K (default 3) songs + their scores are returned
       │
       ▼
GENERATOR  (src/rag.py · RAGRecommender → Claude claude-sonnet-4-6)
  Augmented prompt = user query  +  retrieved song metadata (as context)
  Claude generates natural-language recommendations with musical reasoning
  Output: a paragraph explaining why each song fits the request
       │
       ▼
EVALUATOR  (src/evaluator.py · evaluate())
  retrieval_relevance  — are the retrieved songs actually similar to the query?
  diversity            — do results span different genres and moods?
  explanation_coverage — does the explanation address the user's keywords?
       │
       ├─► All pass ──► Output delivered to user
       │
       └─► Any fail ──► HUMAN REVIEW (see below)

OUTPUT
  Recommended song titles + Claude-written explanations
```

---

## Human / Testing Involvement

| Where | Who | What they check |
|---|---|---|
| `tests/test_rag.py` — adversarial cases | Developer / CI | Empty queries, nonsense input, k > corpus — system must not crash |
| `tests/test_rag.py` — mocked integration | Developer | Full pipeline structure; real Claude output inspected manually before shipping |
| Evaluator threshold check | Automated + human | If any metric (relevance, diversity, coverage) falls below an acceptable level, the result is flagged for human review |
| `model_card.md` reflection | Human author | Bias audit: does the retriever over-retrieve the same genre? Does Claude hallucinate song details? |

---

## Component Responsibilities

| Component | File | Input | Output |
|---|---|---|---|
| Song Documents | `src/rag.py` | `songs.csv` dicts | Text descriptions for TF-IDF |
| Retriever | `src/rag.py` | Natural language query | Top-K songs + similarity scores |
| Generator | `src/rag.py` | Query + retrieved context | Natural language recommendation |
| Evaluator | `src/evaluator.py` | RAG result dict | `{relevance, diversity, coverage}` scores |
| Tester | `tests/test_rag.py` | Retriever + Evaluator + mock LLM | Pass/fail assertions + human flag comments |
| Knowledge Base | `data/songs.csv` | — | Song metadata (genre, mood, energy, …) |
