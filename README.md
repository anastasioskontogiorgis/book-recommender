# Semantic Book Recommender v2

A content-based book recommender built on the Book-Crossing catalogue, enriched with descriptions and subject tags from the Open Library API, and retrieved through sentence-transformer dense embeddings.

This repository contains the second iteration (v2) of the project. Building upon an initial proof-of-concept, this version introduces a fully semantic retrieval pipeline, simplifies the underlying architecture, and implements a rigorous quantitative evaluation harness to measure performance.

**[Live Demo](https://anastasios-k-book-recommender.hf.space/)** · **[Analysis Notebook](book_recommender_v2.ipynb)** · **[Enrichment Script](openlibrary_fetch.py)**

---

## Benchmarks & Results

| Variant | Architecture / Representation | Input Data | `subject-share@5` | `subject-Jaccard@5` | `author-match@5` |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **A (v1)** | TF-IDF (Lexical Baseline) | Title Only | 0.128 | 0.018 | 0.011 |
| **B** | TF-IDF (Lexical Baseline) | Title + Description | 0.150 | 0.025 | 0.017 |
| **C (v2)** | **MiniLM-L6-v2 (384-d Dense)** | **Title + Description** | **0.292** | **0.045** | **0.032** |
| — | *Random Baseline Floor* | N/A | 0.020 | 0.002 | 0.000 |

*Evaluated over 500 seeded queries drawn from 12,531 books carrying $\ge 2$ informative subject tags, retrieving top-5 recommendations. The primary metric, `subject-share@5`, measures the proportion of recommendations sharing at least one informative subject with the query book.*

### Comparison Breakdown
The comparison is arranged so that each change can be attributed separately. By introducing Variant B as an intermediate benchmark, it is possible to isolate the exact source of retrieval gains: (i) Data Enrichment ($A \rightarrow B$), holding the model fixed (TF-IDF) while adding description metadata yields a **17% relative improvement**; (ii) Encoder Upgrade ($B \rightarrow C$), holding the enriched text fixed while upgrading from lexical TF-IDF to dense `all-MiniLM-L6-v2` embeddings yields a further **95% relative improvement**; (iii) Overall Lift, the model architecture upgrade contributes roughly 5× more retrieval quality than data enrichment alone, and Variant C sits 14.6× above the random floor.

For Variant C, query items with descriptions achieve a `subject-share@5` of **0.357**, compared to **0.239** for title-only items. Recommendations perform best when descriptive metadata is present, while remaining well above the random floor (0.020) when it is absent.

---

## Metric Design & Limitations

Subject tags from Open Library serve as weak ground-truth labels. They are crowdsourced, incomplete, and non-standardised; two semantically related books often carry disjoint tag sets and register as a false "miss". Absolute metric values therefore understate real-world recommendation quality. The evaluation harness is designed as a **controlled relative comparison across variants on identical queries and labels**, not an absolute measure of reader satisfaction. Holding the query set, candidate pool, and target labels constant across all variants is what makes the relative deltas a reliable measurement of architectural change.

To prevent trivial matches from inflating scores, uninformative tags appearing on $>5\%$ of subject-bearing books (e.g., *"fiction"*) are filtered prior to scoring.

---

## Qualitative Comparison

While quantitative metrics capture overall performance, individual retrieval outputs illustrate the semantic shift:

* **Example 1: *The Hobbit (Young Adult Edition)***
  * **Lexical (Variant A):** Returns *Bullfrog and Gertrude Go Camping*, *The Young Martial Arts Enthusiast*, and *Growing Young* (matching the literal token *"Young"*).
  * **Semantic (Variant C):** Returns *The Lord of the Rings*, *Lord of the Rings Trilogy*, and *Tolkien: A Biography*.

* **Example 2: *Pride and Prejudice (Everyman's Classics)***
  * **Lexical (Variant A):** Returns unrelated books sharing the publisher imprint keyword *"Everyman"*, including *Moll Flanders*, *The Koran*, and *Frankenstein*.
  * **Semantic (Variant C):** Returns alternative editions of *Pride and Prejudice* alongside other Jane Austen works.

Publisher imprints, series names, and edition descriptors account for a substantial share of title tokens. Lexical matching cannot separate these metadata elements from words describing actual book content. Incorporating full descriptions dilutes this title noise, while semantic embeddings largely filter out the metadata altogether.

## System Architecture

* **Corpus Pipeline:** A deterministic 15,000-book sample of the deduplicated Book-Crossing dataset, enriched via the Open Library API (`openlibrary_fetch.py`). Resolved 99.9% of ISBNs, retrieving descriptions for 37% of catalogue items and subject tags for 91.5%. Items lacking descriptions fall back to title-only encoding.
* **Vector Representation:** Dense 384-dimensional sentence embeddings generated via `all-MiniLM-L6-v2` over concatenated title and description text. Embeddings are $L_2$-normalised so that inner product operations equal cosine similarity.
* **Retrieval Engine:** Exact $k$-nearest neighbour search across the full catalogue. Implemented via FAISS (`IndexFlatIP`) with an in-memory NumPy dot-product fallback. Both paths yield identical recommendations.
* **Clustering & Projection:** While v1 used K-Means clusters to bound candidate search pools, v2 removes clustering from the retrieval path entirely. Dimensionality reduction (UMAP) and K-Means are used for **exploratory analysis** (visualising genre separation in 2D space in Section 5 of the notebook).
* **Lightweight Deployment:** The notebook exports precomputed vectors (`embeddings.npy`) and metadata (`catalog.parquet`). The Gradio demo (`app.py`) loads these static artefacts directly, reducing retrieval to a single $15{,}000 \times 384$ inner product and requiring neither PyTorch nor a runtime model download.

---

## Iteration Retrospective: What Changed from v1

The v1 iteration built recommendations using TF-IDF and Word2Vec features over titles, reduced via PCA, clustered with K-Means, and restricted candidate retrieval strictly to the query book's cluster. Re-evaluating this design revealed three flaws:

1. **Title-Only Sparsity:** Book titles alone carry little content signal, causing lexical models to over-index on edition and publisher (metadata) tokens. The Open Library enrichment addresses this directly, and variant B isolates how much it contributes.
2. **Artificial Cluster Boundaries:** Hard clustering restricted nearest-neighbour search to arbitrary sub-pools, discarding high-quality candidates sitting on opposite sides of cluster borders without offering meaningful computational savings.
3. **Index Alignment Bug:** Deduplicating titles in v1 dropped 35,080 rows without resetting the DataFrame index. Pandas index *labels* were subsequently passed directly into *positional* lookups on the NumPy feature matrix, pairing each book's title with a different book's feature vector. A guard clause silently discarded those labels exceeding the matrix bounds, suppressing the `IndexError` that would otherwise have surfaced the fault. Verified against the original data, only 239 of 236,280 books retained correct alignment.

Because individual recommendations still appeared plausible during manual inspection, the bug survived review. This failure directly motivated v2's evaluation harness and random baseline floor: a retrieval score sitting at the random floor is unmistakable in a way that an individually odd recommendation is not.

---

## Quickstart & Execution

```bash
pip install -r requirements.txt

# optional: rebuild the enriched corpus from the Book-Crossing catalogue
# (several hours, rate-limited and resumable)
python openlibrary_fetch.py

jupyter lab book_recommender_v2.ipynb
```

The notebook runs end to end from `enriched_books.jsonl`, which is committed, so the enrichment step is not required to reproduce the results. Variant C downloads `all-MiniLM-L6-v2` on first use, approximately 90 MB. Every stochastic step draws from a single seed for exact reproducibility.

The demo application requires only the exported artefacts:

```bash
python app.py
```

## Future Work

Future developments can expand the system across two main tracks. On the data side, incorporating user interaction datasets (such as the Book-Crossing `Ratings.csv`) will enable true behavioural evaluation and hybrid collaborative filtering, while integrating secondary metadata APIs (such as Google Books) can expand description coverage beyond 37%. On the retrieval side, production quality can be enhanced by introducing a two-stage pipeline featuring a cross-encoder re-ranker over top-50 candidates, popularity de-biasing, and Maximal Marginal Relevance (MMR) for output diversity.

## Data and Licence

Book catalogue from the Book-Crossing dataset. Descriptions and subject tags from
[Open Library](https://openlibrary.org). Embeddings from
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Released under the MIT Licence.
