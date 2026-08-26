"""
Hybrid retrieval over the legal corpus.

Keyword layer: simple substring / token overlap match against title + keywords.
Vector layer:  TF-IDF + cosine similarity over title + keywords + source_text.

This is intentionally dependency-light (scikit-learn + numpy only) so it runs
anywhere, including Replit's free tier, with no external embedding API call.
Swap `TfidfVectorizer` for real sentence embeddings + pgvector later without
changing the public interface (`search`, `get_by_id`).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "corpus.json"


class LegalCorpus:
    def __init__(self, path: Path = DATA_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        self._texts = [
            f"{d['title']} {d['keywords']} {d['source_text']}" for d in self.docs
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform(self._texts)

    def search(self, query: str, top_k: int = 3):
        """Hybrid search: TF-IDF cosine similarity, boosted by exact keyword hits."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        q_tokens = [t for t in query.lower().split() if len(t) > 3]
        boosted = sims.copy()
        for i, doc in enumerate(self.docs):
            haystack_words = f"{doc['title']} {doc['keywords']}".lower().split()
            # loose stem match: compare first 5 chars so "arrested"/"arrest",
            # "deposit"/"deposits" etc. still count as the same concept
            hits = sum(
                1
                for tok in q_tokens
                for hw in haystack_words
                if tok[:5] == hw[:5]
            )
            if hits:
                boosted[i] += 0.12 * hits  # keyword-match boost on top of vector similarity

        ranked_idx = np.argsort(boosted)[::-1][:top_k]
        results = []
        for i in ranked_idx:
            if boosted[i] <= 0:
                continue
            results.append({**self.docs[i], "score": float(boosted[i])})
        return results

    def list_all(self, filter_str: str = ""):
        if not filter_str:
            return [{"id": d["id"], "type": d["type"], "title": d["title"]} for d in self.docs]
        f = filter_str.lower()
        return [
            {"id": d["id"], "type": d["type"], "title": d["title"]}
            for d in self.docs
            if f in d["title"].lower() or f in d["keywords"].lower()
        ]

    def get_by_id(self, doc_id: str):
        for d in self.docs:
            if d["id"] == doc_id:
                return d
        return None


corpus = LegalCorpus()
