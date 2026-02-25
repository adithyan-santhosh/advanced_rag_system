from embedding.embedding import EmbeddingModel
from vectorStore.vector_store import FAISSVectorStore
from reranking.reranker import CrossEncoderReranker
from chunking.chunking import semantic_chunking
from generation.generation import OllamaLLM
from analyzer.document_analyzer import DocumentAnalyzer
from retrieval.hybrid_retriever import HybridRetriever
import os
import json


class RAGPipeline:

    def __init__(self, document_text, chunk_size=200, storage_path="storage"):

        self.storage_path = storage_path

        analyzer = DocumentAnalyzer(document_text)
        self.use_hybrid = analyzer.is_code_heavy()

        print(
            f"Adaptive Retrieval Mode: "
            f"{'HYBRID' if self.use_hybrid else 'VECTOR'}"
        )

        self.embedder = EmbeddingModel("all-MiniLM-L6-v2")

        self.reranker = CrossEncoderReranker()

        self.llm = OllamaLLM()

        dimension = 384

        self.vector_store = FAISSVectorStore(dimension)

        index_exists = os.path.exists(
            f"{storage_path}/faiss.index"
        )

        if index_exists:

            print("Loading FAISS index from disk...")

            self.vector_store.load(storage_path)

            self.chunks = self.vector_store.texts

        else:

            print("Creating new FAISS index...")

            self.chunks = semantic_chunking(
                document_text,
                max_chunk_size=chunk_size
            )

            embeddings = self.embedder.embed_documents(
                self.chunks
            )

            self.vector_store.add_embeddings(
                embeddings.astype("float32"),
                self.chunks
            )

            self.vector_store.save(storage_path)

            config = {
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": chunk_size,
                "retrieval_mode":
                "hybrid" if self.use_hybrid else "vector"
            }

            with open(
                f"{storage_path}/config.json",
                "w"
            ) as f:

                json.dump(config, f, indent=2)


        if self.use_hybrid:

            self.hybrid = HybridRetriever(
                self.chunks,
                self.vector_store
            )

        else:

            self.hybrid = None


    def retrieve(self, query, top_k=3):

        if self.use_hybrid:
            retrieved = self.hybrid.search(
                query,
                self.embedder,
                top_k=top_k,
                alpha=0.6
            )
        else:
            query_embedding = self.embedder.embed_query(query)
            retrieved = self.vector_store.search(query_embedding, top_k=top_k)

        reranked = self.reranker.rerank(query, retrieved)

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
