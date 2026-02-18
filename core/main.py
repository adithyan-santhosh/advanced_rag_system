from chunking.chunking import clean_text, overlapping_chunking
from embedding.embedding import EmbeddingModel
from vectorStore.vector_store import FAISSVectorStore
from generation.generation import OllamaLLM

if __name__ == "__main__":

    sample_text = """
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

    # Clean + chunk
    text = clean_text(sample_text)
    chunks = overlapping_chunking(text, chunk_size=150, overlap=30)

    print("Chunks created:", len(chunks))

    # Initialize embedding model
    embedder = EmbeddingModel()

    # Generate embeddings
    embeddings = embedder.embed_documents(chunks)

    print("Embedding shape:", embeddings.shape)

    # Initialize FAISS
    vector_store = FAISSVectorStore(dimension=embeddings.shape[1])

    # Add embeddings
    vector_store.add_embeddings(embeddings.astype("float32"), chunks)

    # Query
    query = "What is the battery's standard operating temperature?"
    query_embedding = embedder.embed_query(query)

    SIMILARITY_THRESHOLD = 0.2
    retrieved_results = vector_store.search(query_embedding, top_k=3)

    valid_chunks = []
    context_chunks = []

    print("\nRetrieved Chunks with Distance:\n")

    for item in retrieved_results:
        print("Score:", item["score"])
        print(item["text"])
        print("-" * 40)

        if item["score"] >= SIMILARITY_THRESHOLD:
            valid_chunks.append(item["text"])

    if not valid_chunks:
        print("\nNo relevant context found. Aborting generation.")
        exit()

    context = "\n\n".join(valid_chunks)

    prompt = f"""
    You are an AI assistant. Answer ONLY using the provided context.
    If the answer is not in the context, say you don't know.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    llm = OllamaLLM()
    answer = llm.generate(prompt)

    print("\nGenerated Answer:\n")
    print(answer)
