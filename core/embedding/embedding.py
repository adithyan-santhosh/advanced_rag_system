from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Convert list of texts into embeddings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_query(self, query):
        """
        Convert single query into embedding.
        """
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]
