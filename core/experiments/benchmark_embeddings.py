import os
import csv
import time

from chunking.chunking import clean_text, overlapping_chunking, semantic_chunking
from embedding.embedding import EmbeddingModel
from vectorStore.vector_store import FAISSVectorStore
from reranking.reranker import CrossEncoderReranker
from retrieval.hybrid_retriever import HybridRetriever


DOCUMENT_TEXT = """
1. Battery and Charging Specifications
The Solaris-X fleet utilizes the Lithium-Sulfur "Gen-3" battery pack. Standard operating temperature for the battery is between 15°C and 35°C. If the core temperature exceeds 45°C, the vehicle must enter "Thermal Safe Mode" and limit speed to 20 km/h. Charging must only occur at Level 3 DC Fast Charging stations. A full charge from 10% to 80% takes exactly 22 minutes under optimal conditions.

2. Maintenance Intervals
Preventive maintenance is categorized into two tiers:

Tier Alpha: Performed every 5,000 km. Includes sensor calibration and tire rotation.

Tier Beta: Performed every 20,000 km. Includes coolant replacement and brake pad inspection.

3. Emergency Override Procedures
In the event of a LIDAR failure, the Remote Operator (RO) must be notified via the "Signal-Blue" encrypted channel. The RO has a maximum latency requirement of 150ms to maintain control. If the connection exceeds 500ms, the vehicle is programmed to execute an "Immediate Curb Pull-over."

4. Service Regional Boundaries
Currently, the Solaris-X fleet is authorized to operate in the Northwest District (Zones A, B, and C) and the Central Business District (Zones D and E). Operation in the South Waterfront (Zone F) is strictly prohibited due to ongoing construction and interference with GPS signals.
"""


TEST_QUERIES = [
    {
        "query": "What happens if battery temperature exceeds 45°C?", 
        "type": "relevant", 
        "expected_keyword": "Thermal Safe Mode"
    },
    {
        "query": "What is included in Tier Beta maintenance?", 
        "type": "relevant", 
        "expected_keyword": "coolant replacement"
    },
    {
        "query": "What happens if remote operator latency exceeds 500ms?", 
        "type": "relevant", 
        "expected_keyword": "Immediate Curb Pull-over"
    },
    {
        "query": "Where is Solaris-X prohibited from operating?", 
        "type": "relevant", 
        "expected_keyword": "South Waterfront"
    },
    {
        "query": "What type of engine does Solaris-X use?", 
        "type": "irrelevant", 
        "expected_keyword": None
    },
    {
        "query": "What fuel does Solaris-X consume?", 
        "type": "irrelevant", 
        "expected_keyword": None
    },
    {
        "query": "Who is the CEO of Solaris-X?", 
        "type": "irrelevant", 
        "expected_keyword": None
    },
    {
        "query": "What is the vehicle's horsepower?", 
        "type": "irrelevant", 
        "expected_keyword": None
    },
    {
        "query": "Does Solaris-X use diesel engine?", 
        "type": "irrelevant", 
        "expected_keyword": None
    }
]


EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2"
    # "BAAI/bge-small-en"
]


def run_benchmark():
    text = clean_text(DOCUMENT_TEXT)
    chunks = semantic_chunking(text, max_chunk_size=200)
     
    print(f"No. of chunks: {len(chunks)}")

    os.makedirs("experiments/results", exist_ok=True)
    csv_file = "experiments/results/embedding_benchmark_results.csv"

    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "query",
            "query_type",
            "top_score",
            "top1_correct",
            "top3_correct",
            "embedding_time_sec"
        ])

        for model_name in EMBEDDING_MODELS:

            print("\n" + "="*60)
            print(f"Benchmarking Model: {model_name}")
            print("="*60)

            embedder = EmbeddingModel(model_name)

            start_time = time.time()
            embeddings = embedder.embed_documents(chunks)
            embedding_time = time.time() - start_time

            vector_store = FAISSVectorStore(dimension=embeddings.shape[1])
            reranker = CrossEncoderReranker()
            hybrid = HybridRetriever(chunks, vector_store)

            vector_store.add_embeddings(embeddings.astype("float32"), chunks)

            relevant_scores = []
            irrelevant_scores = []
            top1_correct_count = 0
            relevant_count = 0
            top3_correct_count = 0

            for item in TEST_QUERIES:

                query = item["query"]
                query_type = item["type"]
                expected_keyword = item["expected_keyword"]

                if "bge" in model_name.lower():
                    query = "query: " + query

                hybrid_results = hybrid.search(
                    query,   # IMPORTANT: no "query:" prefix
                    embedder,
                    top_k=3,
                    alpha=0.5
                )

                results = hybrid_results

                # Apply reranking
                reranked_results = reranker.rerank(query, results)

                top_score = reranked_results[0]["score"]
                top_chunks = [r["text"] for r in reranked_results]

                print(f"\nQuery: {query}")
                print(f"Top-1 Score: {top_score:.4f}")

                print("\nTop-3 Hybrid Scores:")
                for r in results:
                    print(f"Vector: {r['vector_score']:.3f} | BM25: {r['bm25_score']:.3f} | Combined: {r['combined_score']:.3f}")


                top1_correct = False
                top3_correct = False

                if query_type == "relevant":
                    relevant_count += 1
                    relevant_scores.append(top_score)

                    # Top-1 check
                    if expected_keyword and expected_keyword.lower() in top_chunks[0].lower():
                        top1_correct = True
                        top1_correct_count += 1

                    # Top-3 check
                    if expected_keyword and any(
                        expected_keyword.lower() in chunk.lower() for chunk in top_chunks
                    ):
                        top3_correct = True
                        top3_correct_count += 1
                else:
                    irrelevant_scores.append(top_score)

                writer.writerow([
                    model_name,
                    query,
                    query_type,
                    round(top_score, 4),
                    top1_correct,
                    top3_correct,
                    round(embedding_time, 4)
                ])

            # Metrics
            if relevant_scores and irrelevant_scores:
                avg_relevant = sum(relevant_scores) / len(relevant_scores)
                avg_irrelevant = sum(irrelevant_scores) / len(irrelevant_scores)

                min_relevant = min(relevant_scores)
                max_irrelevant = max(irrelevant_scores)

                separation_margin = avg_relevant - avg_irrelevant
                hard_gap = min_relevant - max_irrelevant

                top1_accuracy = top1_correct_count / relevant_count
                top3_accuracy = top3_correct_count / relevant_count

                print("\n--- Model Summary ---")
                print(f"Avg Relevant Score: {avg_relevant:.4f}")
                print(f"Avg Irrelevant Score: {avg_irrelevant:.4f}")
                print(f"Separation Margin: {separation_margin:.4f}")
                print(f"Min Relevant Score: {min_relevant:.4f}")
                print(f"Max Irrelevant Score: {max_irrelevant:.4f}")
                print(f"Hard Separation Gap: {hard_gap:.4f}")
                print(f"Top-1 Accuracy (Relevant Queries): {top1_accuracy:.2f}")
                print(f"Top-3 Accuracy: {top3_accuracy:.2f}")

    print("\nBenchmark completed. Results saved to CSV.")



if __name__ == "__main__":
    run_benchmark()
