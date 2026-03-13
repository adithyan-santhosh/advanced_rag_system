from sentence_transformers import SentenceTransformer
import numpy as np
import time


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        start = time.time()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        end = time.time()
        print(f"Document embedding time: {end - start:.3f} sec")
        return embeddings

    def embed_query(self, query):
        start = time.time()
        embedding = self.model.encode([query], convert_to_numpy=True)
        end = time.time()
        print(f"Query embedding time: {end - start:.3f} sec")
        return embedding[0]
    