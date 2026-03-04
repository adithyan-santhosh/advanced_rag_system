from core.embedding.embedding import EmbeddingModel
from core.vector_store.vector_store import FAISSVectorStore
from core.reranking.reranker import CrossEncoderReranker
from core.chunking.chunking import semantic_chunking
from core.generation.generation import OllamaLLM
from core.analyzer.document_analyzer import DocumentAnalyzer
from core.retrieval.hybrid_retriever import HybridRetriever
from core.loader.document_loader import DocumentLoader
from core.logger.logger import logger
import time
import os
import json


class RAGPipeline:

    def __init__(self, data_folder="data", chunk_size=200, storage_path="storage"):

        self.data_folder = data_folder
        self.storage_path = storage_path
        self.chunk_size = chunk_size

        print("\nInitializing RAG Pipeline...\n")

        # ---------- Load Documents ----------

        loader = DocumentLoader(data_folder)

        documents = loader.load_documents()

        if len(documents) == 0:
            raise Exception("No documents found in data folder")

        print(f"Loaded {len(documents)} documents")

        # ---------- Adaptive Retrieval Detection ----------

        combined_text = "\n".join([doc["text"] for doc in documents])

        analyzer = DocumentAnalyzer(combined_text)

        self.use_hybrid = analyzer.is_code_heavy()

        print(
            f"Adaptive Retrieval Mode: "
            f"{'HYBRID' if self.use_hybrid else 'VECTOR'}"
        )

        # ---------- Initialize Models ----------

        self.embedder = EmbeddingModel("all-MiniLM-L6-v2")

        self.reranker = CrossEncoderReranker()

        self.llm = OllamaLLM()

        # ---------- Vector Store ----------

        dimension = 384

        self.vector_store = FAISSVectorStore(dimension)

        index_exists = os.path.exists(
            f"{storage_path}/faiss.index"
        )

        # ---------- Load Existing Index ----------

        if index_exists:

            print("\nLoading FAISS index from disk...\n")

            self.vector_store.load(storage_path)

            self.chunks = self.vector_store.texts

        else:

            print("\nBuilding index for multiple documents...\n")

            all_chunks = []
            all_metadata = []

            for doc in documents:

                chunks = semantic_chunking(
                    doc["text"],
                    max_chunk_size=chunk_size
                )

                metadata = [
                    {"source": doc["source"]}
                    for _ in chunks
                ]

                all_chunks.extend(chunks)
                all_metadata.extend(metadata)

            self.chunks = all_chunks

            print(f"Total chunks created: {len(self.chunks)}")

            embeddings = self.embedder.embed_documents(
                self.chunks
            )

            self.vector_store.add_embeddings(
                embeddings.astype("float32"),
                self.chunks,
                all_metadata
            )

            self.vector_store.save(storage_path)

            # Save config

            config = {
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": chunk_size,
                "retrieval_mode":
                "hybrid" if self.use_hybrid else "vector",
                "documents": [
                    doc["source"] for doc in documents
                ]

            }

            os.makedirs(storage_path, exist_ok=True)

            with open(
                f"{storage_path}/config.json",
                "w"
            ) as f:

                json.dump(config, f, indent=2)

        # ---------- Hybrid Retriever ----------

        if self.use_hybrid:

            print("Initializing Hybrid Retriever\n")

            self.hybrid = HybridRetriever(
                self.chunks,
                self.vector_store
            )

        else:

            self.hybrid = None

        print("RAG Pipeline Ready\n")

    def retrieve(self, query, retrieval_k=8, final_k=3):

        # -------- Stage 1: Retrieval --------
        if self.use_hybrid:

            retrieved = self.hybrid.search(
                query,
                self.embedder,
                top_k=retrieval_k,
                alpha=0.6
            )

        else:

            query_embedding = self.embedder.embed_query(query)

            retrieved = self.vector_store.search(
                query_embedding,
                top_k=retrieval_k
            )

        # -------- Stage 2: Reranking --------
        reranked = self.reranker.rerank(
            query,
            retrieved
        )

        # -------- Stage 3: Return Top Final_k --------
        return reranked[:final_k]

    def generate_answer(self, query, top_k=3, confidence_threshold=0.0):

        start_time = time.time()

        retrieved = self.retrieve(
            query,
            retrieval_k=8,
            final_k=top_k
        )

        top_score = retrieved[0]["score"]
        latency = round(time.time() - start_time, 3)

        logger.info(
        f"QUERY | "
        f"Mode={ 'HYBRID' if self.use_hybrid else 'VECTOR'} | "
        f"Confidence={top_score:.3f} | "
        f"Latency={latency}s | "
        f"Query={query}"
    )
        # Confidence gating

        if top_score < confidence_threshold:

            return {
                "answer":
                "I don't have enough information to answer this question.",
                "confidence_score": top_score,
                "sources": []
            }

        # Build Context

        context = "\n\n".join(
            [r["text"] for r in retrieved]
        )

        prompt = f"""
You are an AI assistant.
Answer ONLY using the provided context.
Extract exact values when present.
Field names in the document may use abbreviations.
Examples:
REF_CODE = reference code
Voltage Ref = voltage reference
If the answer is not present say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

        answer = self.llm.generate(prompt)

        # Extract sources from metadata
        source_set = set()

        for r in retrieved:
            meta = r.get("metadata", {})
            if meta and "source" in meta:
                source_set.add(meta["source"])

        # Convert to clean list
        sources = {
            "documents": sorted(source_set)
        }

        return {
            "answer": answer.strip(),
            "confidence_score": top_score,
            "sources": sources
        }
    
    def rebuild_index(self):

        print("\nRebuilding FAISS index...\n")

        loader = DocumentLoader(self.data_folder)

        documents = loader.load_documents()

        if len(documents) == 0:
            raise Exception("No documents found")

        # ---------- Re-run Adaptive Retrieval Detection ----------

        combined_text = "\n".join(
            [doc["text"] for doc in documents]
        )

        analyzer = DocumentAnalyzer(combined_text)

        self.use_hybrid = analyzer.is_code_heavy()

        print(
            f"Adaptive Retrieval Mode: "
            f"{'HYBRID' if self.use_hybrid else 'VECTOR'}"
        )

        # ---------- Chunk Documents ----------

        all_chunks = []
        all_metadata = []

        for doc in documents:

            chunks = semantic_chunking(
                doc["text"],
                max_chunk_size=self.chunk_size
            )

            metadata = [
                {"source": doc["source"]}
                for _ in chunks
            ]

            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

        self.chunks = all_chunks

        print(f"Total chunks: {len(self.chunks)}")

        # ---------- Rebuild FAISS ----------

        embeddings = self.embedder.embed_documents(
            self.chunks
        )

        dimension = embeddings.shape[1]

        self.vector_store = FAISSVectorStore(dimension)

        self.vector_store.add_embeddings(
            embeddings.astype("float32"),
            self.chunks,
            all_metadata
        )

        self.vector_store.save(self.storage_path)

        # ---------- Rebuild Hybrid Retriever ----------

        if self.use_hybrid:

            print("Initializing Hybrid Retriever")

            self.hybrid = HybridRetriever(
                self.chunks,
                self.vector_store
            )

        else:

            self.hybrid = None

        print("\nIndex rebuilt successfully\n")