from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    def __init__(self, chunks, vector_store):
        self.chunks = chunks
        self.vector_store = vector_store

        # Tokenize for BM25
        tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, embedder, top_k=3, alpha=0.5):
        """
        alpha = weight for vector score
        (1-alpha) = weight for BM25 score
        """

        # -------- Vector Search --------
        query_embedding = embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=len(self.chunks))

        vector_scores = {r["text"]: r["score"] for r in vector_results}

        # -------- BM25 Search --------
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores
        bm25_scores = np.array(bm25_scores)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # -------- Combine Scores --------
        combined_results = []

        for i, chunk in enumerate(self.chunks):
            v_score = vector_scores.get(chunk, 0)
            b_score = bm25_scores[i]

            combined_score = alpha * v_score + (1 - alpha) * b_score

            combined_results.append({
                "text": chunk,
                "vector_score": float(v_score),
                "bm25_score": float(b_score),
                "combined_score": float(combined_score),
                "metadata": self.vector_store.metadata[i]
            })

        # Sort by combined score
        combined_results = sorted(
            combined_results,
            key=lambda x: x["combined_score"],
            reverse=True
        )

        return combined_results[:top_k]
