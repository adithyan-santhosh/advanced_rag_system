import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        Using L2 distance for now.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add_embeddings(self, embeddings, texts):
        """
        Add embeddings and corresponding texts to index.
        """
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, top_k=3):
        """
        Search nearest neighbors.
        """
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results
