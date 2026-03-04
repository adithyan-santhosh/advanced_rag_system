from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, retrieved_chunks):

        pairs = [
            (query, chunk["text"])
            for chunk in retrieved_chunks
        ]

        scores = self.model.predict(pairs)

        reranked = sorted(
            zip(retrieved_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for chunk, score in reranked:
            results.append({
                "text": chunk["text"],
                "score": float(score),
                "metadata": chunk.get("metadata", {})
            })

        return results