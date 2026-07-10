# app.py — Semantic Book Recommender v2, interactive demo (Gradio)
# ---------------------------------------------------------------------------
# Design notes
# ------------
# * The app consumes two PRECOMPUTED artifacts exported by the v2 notebook:
#       catalog.parquet   — one row per book (title, author, snippet, subjects)
#       embeddings.npy    — (N, 384) float32 MiniLM sentence embeddings, L2-normalised
#   It deliberately does NOT depend on sentence-transformers/torch: all it does at
#   query time is an inner-product search over unit vectors, which keeps the
#   Hugging Face Space tiny and fast to cold-start.
# * Retrieval: FAISS IndexFlatIP when available (the scale-ready idiom), plain
#   NumPy matrix product otherwise — identical results either way at this size.
# * Explanations: a deterministic template built from the informative subject
#   tags the two books share. Optionally, if an ANTHROPIC_API_KEY is configured
#   (e.g. as a Space secret) and the checkbox is ticked, a small LLM phrases a
#   one-line "why you might like it" from the two descriptions instead. The LLM
#   path is wrapped in try/except and always falls back to the template, so the
#   public demo never breaks on a missing/expired key.
# ---------------------------------------------------------------------------

import os

import numpy as np
import pandas as pd
import gradio as gr

# ----------------------------- artifacts ----------------------------------
CATALOG_PATH = "catalog.parquet"
EMB_PATH = "embeddings.npy"

catalog = pd.read_parquet(CATALOG_PATH)
emb = np.load(EMB_PATH).astype(np.float32)

# Safety: re-normalise rows so inner product == cosine even if the artifact
# was produced without normalize_embeddings=True.
norms = np.linalg.norm(emb, axis=1, keepdims=True)
norms[norms == 0] = 1.0
emb = emb / norms

catalog["subject_set"] = catalog["subjects_info"].fillna("").apply(
    lambda s: frozenset(t for t in s.split("|") if t)
)

# ----------------------------- retrieval ----------------------------------
try:  # FAISS if present; NumPy fallback otherwise (same exact results here)
    import faiss

    _index = faiss.IndexFlatIP(emb.shape[1])
    _index.add(emb)

    def top_k(query_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = _index.search(emb[[query_idx]], k + 1)  # +1: self comes first
        keep = ids[0] != query_idx
        return ids[0][keep][:k], scores[0][keep][:k]

except ImportError:

    def top_k(query_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = emb @ emb[query_idx]
        scores[query_idx] = -np.inf
        ids = np.argsort(-scores)[:k]
        return ids, scores[ids]


# ------------------------- dropdown choices --------------------------------
# Display labels must be unique for Gradio to map a selection back to a row;
# duplicate "Title — Author" pairs get an ISBN suffix.
_labels = (catalog["title"].str.strip() + "  —  " + catalog["author"].str.strip()).tolist()
_seen: dict[str, int] = {}
CHOICES: list[str] = []
for i, lab in enumerate(_labels):
    if lab in _seen:
        lab = f"{lab} (ISBN {catalog.loc[i, 'isbn']})"
    _seen[lab] = i
    CHOICES.append(lab)
LABEL_TO_IDX = {lab: i for lab, i in _seen.items()}


# ------------------------- explanations ------------------------------------
def template_reason(q: int, r: int) -> str:
    shared = sorted(catalog.loc[q, "subject_set"] & catalog.loc[r, "subject_set"])
    if shared:
        return "Shared themes: " + ", ".join(shared[:4]) + "."
    return "Close to your pick in overall description semantics."


def llm_reason(q: int, r: int) -> str | None:
    """One-sentence LLM phrasing; returns None on any failure (caller falls back)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = (
            "In one short, concrete sentence, explain why a reader who liked the "
            f"first book might enjoy the second. No preamble.\n\n"
            f"First: {catalog.loc[q, 'title']} — {catalog.loc[q, 'description'] or 'no description'}\n"
            f"Second: {catalog.loc[r, 'title']} — {catalog.loc[r, 'description'] or 'no description'}"
        )
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return text or None
    except Exception:
        return None


# ------------------------- main callback ------------------------------------
def recommend(choice: str, k: int, use_llm: bool) -> str:
    if not choice:
        return "Pick a book above (the dropdown is searchable — just start typing)."
    q = LABEL_TO_IDX[choice]
    ids, scores = top_k(q, int(k))

    lines = [f"### Because you picked *{catalog.loc[q, 'title']}* "
             f"by {catalog.loc[q, 'author']}\n"]
    for rank, (r, s) in enumerate(zip(ids, scores), start=1):
        reason = (llm_reason(q, r) if use_llm else None) or template_reason(q, r)
        snippet = catalog.loc[r, "description"]
        snippet_md = f"\n   > {snippet[:220]}…" if snippet else ""
        lines.append(
            f"**{rank}. {catalog.loc[r, 'title']}** — {catalog.loc[r, 'author']} "
            f"*(similarity {s:.2f})*  \n   {reason}{snippet_md}\n"
        )
    return "\n".join(lines)


# ------------------------------- UI ----------------------------------------
with gr.Blocks(title="Semantic Book Recommender v2") as demo:
    gr.Markdown(
        "# Semantic Book Recommender v2\n"
        "15,000 Book-Crossing titles enriched with Open Library descriptions, "
        "embedded with `all-MiniLM-L6-v2`, retrieved by exact cosine search. "
        "Companion demo to the analysis notebook — pick a book, get its five "
        "nearest neighbours in semantic space."
    )
    with gr.Row():
        book = gr.Dropdown(choices=CHOICES, label="Pick a book (type to search)",
                           value=None, filterable=True, scale=3)
        k = gr.Slider(3, 10, value=5, step=1, label="Recommendations", scale=1)
    use_llm = gr.Checkbox(
        value=False,
        label="Phrase explanations with an LLM (requires ANTHROPIC_API_KEY; "
              "falls back to shared-themes template)",
    )
    btn = gr.Button("Recommend", variant="primary")
    out = gr.Markdown()
    btn.click(recommend, inputs=[book, k, use_llm], outputs=out)

if __name__ == "__main__":
    demo.launch()
