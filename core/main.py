from chunking.chunking import clean_text, overlapping_chunking
from embedding.embedding import EmbeddingModel
from vectorStore.vector_store import FAISSVectorStore
from generation.generation import OllamaLLM

if __name__ == "__main__":

    sample_text = """
    Retrieval Augmented Generation (RAG) enhances large language models by providing external documents during generation.

    It involves embedding documents into vector space and retrieving relevant context based on similarity search.

    The retrieved context is injected into the prompt before generating the final answer.
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
    query = "How does Dog retrieve information?"
    query_embedding = embedder.embed_query(query)

    retrieved_chunks = vector_store.search(query_embedding, top_k=2)

    # Build grounded prompt
    context = "\n\n".join(retrieved_chunks)

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
