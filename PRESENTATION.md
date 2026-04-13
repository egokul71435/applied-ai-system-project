# VibeMatcher — Final Presentation

## Video Walkthrough

> **Loom Recording:** [ADD YOUR LOOM LINK HERE]
>
> *(Record a 5–7 minute walkthrough using [Loom](https://www.loom.com).
> See the script below for what to cover in each section.)*

---

## GitHub Repository

> **Repo:** [ADD YOUR GITHUB LINK HERE]
>
> *(Push your project with `git push origin main` and paste the URL above.)*

---

## Portfolio Reflection

*What this project says about me as an AI engineer:*

VibeMatcher shows that I approach AI engineering as a discipline of tradeoffs,
not just implementation. I started with the simplest possible system — a
weighted scorer with four hand-tuned rules — and only added complexity when a
specific limitation justified it: the inability to handle natural language
queries led to the RAG layer; the lack of any reliability signal led to the
evaluator and confidence score; the absence of adversarial testing led to the
`min(k, len(documents))` guard that silent failures would have hidden.
Throughout the project I documented what the system gets wrong as carefully as
what it gets right, because I believe the honest model card and the test that
flags edge cases are just as important as the feature that ships. That habit —
build the smallest thing that works, measure its failure modes, then decide
whether complexity is actually warranted — is the engineering practice I want
to carry into every AI project I work on.

---

## Presentation Script (5–7 minutes)

Use these as talking points, not a word-for-word script. Each section has a
suggested time and a demo action where applicable.

---

### Slide 1 — Introduction (0:00–0:45)

**Say:**
"Hi, I'm [your name]. This is VibeMatcher, a music recommendation system I
built across four modules. It started as a simple weighted scoring engine and
grew into a full RAG pipeline that takes natural language queries and generates
explanations using Claude."

**Show:** The project root in your terminal or IDE. Point out the three main
source files: `src/recommender.py`, `src/rag.py`, `src/evaluator.py`.

---

### Slide 2 — The Original System (0:45–1:45)

**Say:**
"The foundation is a content-based recommender. You give it a genre, mood, and
energy level, and it scores all 18 songs in the catalog using a formula — two
points for a genre match, one for mood, and up to one point for energy
closeness. That's it. Simple, but it already reveals something interesting."

**Demo action:** Run `python -m src.main` and show the output.

```
Top recommendations:
Sunrise City - Score: 3.98
Because: genre match (+2.0) | mood match (+1.0) | energy closeness (0.98)

Gym Hero - Score: 2.87
Because: genre match (+2.0) | energy closeness (0.87)
```

**Say:**
"Notice that Gym Hero — an intense workout song — ranks second for a happy pop
user. That's the energy filter bubble: the energy score dominates and pulls in
a song with a completely different mood. I documented this as a bias in the
model card, but fixing the algorithm without fixing the data doesn't actually
solve it."

---

### Slide 3 — System Architecture (1:45–3:00)

**Say:**
"To address the limitation around natural language, I built a RAG layer on top
of the original system. Here's how data flows through it."

**Show:** The diagram from `system_diagram.md` (open the Mermaid preview in
VS Code, or use a screenshot).

**Walk through each component:**

1. **Knowledge Base** — `songs.csv` is converted into text descriptions at
   startup. Each song becomes a sentence like "Sunrise City by The Glows.
   Genre: pop. Mood: happy. High energy electronic song."
2. **Retriever** — When a user types a natural language query, TF-IDF
   vectorizes it alongside all song descriptions and ranks by cosine similarity.
   No API call needed.
3. **Generator** — The top-k retrieved songs are passed as grounded context to
   Claude, which writes a natural language recommendation referencing specific
   musical qualities.
4. **Evaluator** — Three metrics score the result: retrieval relevance,
   diversity, and explanation coverage. A composite confidence score is logged.

**Say:**
"I kept the original scorer and the RAG layer as two separate paths. The scorer
is fast and deterministic — useful for structured input and debugging. The RAG
layer handles the fuzzy, conversational queries that the scorer can't touch."

---

### Slide 4 — Live Demo (3:00–5:00)

**Say:**
"Let me show the RAG recommender running end to end."

**Demo action:** Run the following in your terminal (requires your API key):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python - <<'EOF'
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
from src.recommender import load_songs
from src.rag import RAGRecommender
from src.evaluator import evaluate

songs = load_songs("data/songs.csv")
rag = RAGRecommender(songs)

result = rag.recommend("chill acoustic music for studying", k=3)
print("\n--- Response ---")
print(result["generated_response"])
print("\n--- Confidence:", result["confidence"], "---")
print("--- Metrics:", evaluate(result), "---")
EOF
```

**Point out while it runs:**
- The `INFO` log lines from the retriever showing which songs were pulled and
  their similarity scores.
- The generated response naming specific songs and referencing their musical
  qualities.
- The confidence score and how it reflects both retrieval quality and diversity.

**Say:**
"The model only recommends songs that actually exist in the catalog — it can't
hallucinate titles because the context constrains it. If I ask for something
the catalog doesn't have, the similarity scores will be low and the confidence
score flags it."

**Optional second query:**
```python
result2 = rag.recommend("xyzzy frobulate quux", k=3)
print("Confidence on nonsense query:", result2["confidence"])
```

"Even on a completely meaningless query, the system returns gracefully instead
of crashing — that's what the adversarial tests guard."

---

### Slide 5 — Testing and Reliability (5:00–5:45)

**Say:**
"The system has 23 automated tests across two test files. Let me run them."

**Demo action:** `pytest tests/ -v`

**Say:**
"The test groups cover the retriever's happy path, four adversarial edge cases,
all three evaluator metrics, and a mocked end-to-end integration test where the
Claude API is replaced with a fake response so the test runs without network
access. One thing I learned: the mocked test confirmed the pipeline structure
is correct, but it can't tell me whether Claude's actual output is good. That
required manual inspection. Automated testing and human judgment both matter —
neither is a substitute for the other."

---

### Slide 6 — Ethics and What I Learned (5:45–7:00)

**Say:**
"A few things this project taught me that I'll carry forward."

**Point 1 — Data shapes behavior more than the algorithm:**
"Every bias I found — the energy filter bubble, the Western-genre tilt,
the limited mood vocabulary — came from the dataset, not the code. I could
tune the weights forever and not fix a catalog that simply doesn't contain
certain kinds of music."

**Point 2 — Measuring the right thing is harder than measuring something:**
"My explanation coverage metric scores Claude's responses lower when it
paraphrases a concept rather than using the exact query word. A response about
'calm, focused listening' scores zero for the query 'studying' even though it's
the better answer. The gap between what's measurable and what's meaningful is
a real design problem."

**Point 3 — The failure modes of LLM systems are different:**
"The original scorer fails visibly — you can read the score and see the
mismatch. Claude fails quietly — it produces fluent, confident prose that can
be subtly wrong. That's why the evaluator and the confidence score exist: not
to replace human judgment, but to give it something to act on."

**Close:**
"VibeMatcher is a small system, but it surfaced real questions about how to
build AI responsibly. Thanks."

---

## What to Cover in Your Loom Recording

Record your screen + voice in one continuous take. Suggested structure:

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:20 | Project root in terminal | Quick intro: name, project name |
| 0:20–1:00 | `python -m src.main` output | Show original recommender, name the energy bias |
| 1:00–2:00 | `system_diagram.md` Mermaid | Walk through the RAG architecture |
| 2:00–4:00 | RAG demo in terminal | Run the live query, point to log lines and confidence |
| 4:00–4:45 | `pytest tests/ -v` | Show all 23 passing, mention adversarial tests |
| 4:45–5:30 | README.md in IDE | Point to Ethics section and Testing Summary |
| 5:30–7:00 | Back to terminal or IDE | One honest reflection on what surprised you |

**Tips:**
- Do a dry run before recording — the API call in step 3 takes a few seconds;
  don't let dead air feel like a crash.
- If you don't have an API key available during recording, show the mocked test
  output instead and explain that the live demo requires the key.
- Keep the font size large enough to read in a small video preview.
