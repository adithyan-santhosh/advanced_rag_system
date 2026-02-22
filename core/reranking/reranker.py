from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, retrieved_chunks):
        """
        Rerank retrieved chunks using cross-encoder.
        Returns sorted chunks by relevance.
        """
        pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]
        scores = self.model.predict(pairs)

        reranked = []
        for i, chunk in enumerate(retrieved_chunks):
            reranked.append({
                "text": chunk["text"],
                "score": float(scores[i])
            })

        # Sort descending (higher score = more relevant)
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

        return reranked
