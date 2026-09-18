from rank_bm25 import BM25Okapi


class BM25PreFilter:
    def __init__(self, documents: list[tuple[str, str]]):
        """documents: list of (doc_id, text)."""
        self.ids = [d[0] for d in documents]
        self.texts = [d[1] for d in documents]
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def prefilter(self, query: str, top_n: int = 200) -> list[str]:
        """Returns the top_n doc_ids by BM25 score, to be re-ranked by vector search."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_n]]
