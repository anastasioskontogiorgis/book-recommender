#!/usr/bin/env python3
"""
Open Library enrichment for the Book-Crossing recommender (v2 data step).

Fetches a description and subject tags for a sample of books, keyed by ISBN,
via the Open Library API (edition record -> parent work record). Writes one
JSON object per line (JSONL), so the run is fully RESUMABLE: re-running with
the same arguments skips ISBNs already present in the output file, and
increasing --sample later tops up the same deterministic sample.

Usage (run locally, ideally overnight):
    python openlibrary_fetch.py --books Books.csv --out enriched_books.jsonl \
        --sample 15000 --sleep 1.0 --email your.email@example.com

Requirements: pandas, requests, nltk  (all already used by the notebook).
Please keep --sleep at 1.0 or higher: Open Library is a nonprofit and asks
bulk users to be gentle (their monthly data dumps are the alternative for
truly bulk needs).
"""
from __future__ import anotations
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://openlibrary.org"


def build_corpus(books_csv: str) -> pd.DataFrame:
    """Replicate the notebook's cleaning + dedup so the sample matches the
    modelling corpus exactly (same rows the recommender is trained on)."""
    import nltk
    for pkg in ("stopwords", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop = set(stopwords.words("english"))

    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join(w for w in word_tokenize(text) if w not in stop)

    df = pd.read_csv(books_csv, dtype=str, encoding="utf-8", quoting=1)
    df.columns = ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication",
                  "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L"]
    df = df.dropna(subset=["Book-Title"])
    df["Cleaned-Title"] = df["Book-Title"].apply(clean)
    df = df.drop_duplicates(subset=["Cleaned-Title"]).reset_index(drop=True)
    return df[["ISBN", "Book-Title", "Book-Author", "Cleaned-Title"]]


def get_json(session: requests.Session, url: str, sleep: float, max_retries: int = 4):
    """GET a JSON resource politely: sleep after every request, back off on
    rate-limits/server errors, treat 404 as 'not found'."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                time.sleep(sleep)
                return r.json()
            if r.status_code == 404:
                time.sleep(sleep)
                return None
            if r.status_code in (429, 500, 502, 503):
                time.sleep(sleep * (2 ** (attempt + 1)))
                continue
            time.sleep(sleep)
            return None
        except (requests.RequestException, ValueError):
            time.sleep(sleep * (2 ** (attempt + 1)))
    return None


def extract_description(obj) -> str | None:
    if not obj:
        return None
    d = obj.get("description")
    if isinstance(d, dict):
        d = d.get("value")
    if isinstance(d, str) and d.strip():
        return d.strip()
    return None


def extract_subjects(obj) -> list[str]:
    if not obj:
        return []
    return [s for s in obj.get("subjects", []) if isinstance(s, str)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--books", default="Books.csv", help="path to Books.csv")
    p.add_argument("--out", default="enriched_books.jsonl", help="output JSONL path")
    p.add_argument("--sample", type=int, default=15000,
                   help="number of books to enrich (deterministic sample)")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds to wait after each request (keep >= 1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--email", required=True,
                   help="your contact email for the User-Agent header (API etiquette)")
    args = p.parse_args()

    print("Building corpus (cleaning + dedup, same as notebook)...")
    corpus = build_corpus(args.books)
    print(f"Corpus: {len(corpus)} unique books.")

    # Deterministic, extendable sample: shuffle once with the seed, take the
    # first N. Increasing --sample later keeps all previously sampled ISBNs.
    shuffled = corpus.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    target = shuffled.head(min(args.sample, len(shuffled)))

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["isbn"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(done)} ISBNs already fetched.")

    todo = target[~target["ISBN"].isin(done)]
    est_h = len(todo) * args.sleep * 1.9 / 3600
    print(f"To fetch: {len(todo)} books  (rough estimate: {est_h:.1f} h at "
          f"--sleep {args.sleep}; the run is resumable, Ctrl-C is safe).")

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"BookRecommenderPortfolio/1.0 (contact: {args.email}); "
                      f"one-off dataset enrichment",
        "Accept": "application/json",
    })

    n_ok = n_desc = n_subj = 0
    t0 = time.time()
    with out_path.open("a") as f:
        rows = todo.itertuples(index=False, name=None)  # (ISBN, Title, Author, Cleaned)
        for i, (isbn, title, author, _cleaned) in enumerate(rows, start=1):
            isbn = str(isbn).strip()
            edition = get_json(session, f"{BASE}/isbn/{isbn}.json", args.sleep)

            description = extract_description(edition)
            subjects = extract_subjects(edition)
            work_key = None
            if edition and edition.get("works"):
                work_key = edition["works"][0].get("key")
                work = get_json(session, f"{BASE}{work_key}.json", args.sleep)
                description = description or extract_description(work)
                # merge subjects, preserving order, dropping duplicates
                subjects = list(dict.fromkeys(subjects + extract_subjects(work)))

            rec = {
                "isbn": isbn,
                "title": title,
                "author": author,
                "found": edition is not None,
                "work_key": work_key,
                "description": description,
                "subjects": subjects[:25],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n_ok += edition is not None
            n_desc += description is not None
            n_subj += bool(subjects)
            if i % 50 == 0:
                f.flush()
            if i % 100 == 0:
                el = time.time() - t0
                print(f"[{el/60:6.1f} min] {i}/{len(todo)}  "
                      f"found {n_ok/i:5.1%}  desc {n_desc/i:5.1%}  "
                      f"subjects {n_subj/i:5.1%}")

    print(f"\nDone. New records: {len(todo)}  "
          f"(found {n_ok}, with description {n_desc}, with subjects {n_subj}). "
          f"Total in {out_path}: {len(done) + len(todo)}.")


if __name__ == "__main__":
    main()
