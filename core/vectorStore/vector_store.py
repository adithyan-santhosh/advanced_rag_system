import faiss
import numpy as np
import pickle
import os


class FAISSVectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.texts = []

    def add_embeddings(self, embeddings, texts):

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.texts.extend(texts)


    def search(self, query_embedding, top_k=3):

        query_embedding = np.array([query_embedding]).astype("float32")

        # Normalize query
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx < len(self.texts):

                results.append({
                    "text": self.texts[idx],
                    "score": float(score)
                })

        return results


    # -------- Persistence --------

    def save(self, storage_path):

        os.makedirs(storage_path, exist_ok=True)

        faiss.write_index(
            self.index,
            f"{storage_path}/faiss.index"
        )

        with open(f"{storage_path}/chunks.pkl", "wb") as f:
            pickle.dump(self.texts, f)


    def load(self, storage_path):

        self.index = faiss.read_index(
            f"{storage_path}/faiss.index"
        )

        with open(f"{storage_path}/chunks.pkl", "rb") as f:
            self.texts = pickle.load(f)