from embedding.embedding import EmbeddingModel
from vectorStore.vector_store import FAISSVectorStore
from reranking.reranker import CrossEncoderReranker
from chunking.chunking import semantic_chunking
from generation.generation import OllamaLLM


class RAGPipeline:
    def __init__(self, document_text, chunk_size=200):

        self.embedder = EmbeddingModel("all-MiniLM-L6-v2")

        self.chunks = semantic_chunking(document_text, max_chunk_size=chunk_size)

        embeddings = self.embedder.embed_documents(self.chunks)

        self.vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
        self.vector_store.add_embeddings(embeddings.astype("float32"), self.chunks)

        self.reranker = CrossEncoderReranker()
        self.llm = OllamaLLM()

    def retrieve(self, query, top_k=3):

        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        reranked = self.reranker.rerank(query, vector_results)

        return reranked

    def generate_answer(self, query, top_k=3, confidence_threshold=0.0):

        retrieved = self.retrieve(query, top_k=top_k)

        top_score = retrieved[0]["score"]

        if top_score < confidence_threshold:
            return {
                "answer": "I don't have enough confidence to answer this based on the provided document.",
                "confidence_score": top_score,
                "context_used": []
            }

        context = "\n\n".join([r["text"] for r in retrieved])

        prompt = f"""
You are an AI assistant. Answer ONLY using the provided context.
If the answer is not present in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer.strip(),
            "confidence_score": top_score,
            "context_used": retrieved
        }
