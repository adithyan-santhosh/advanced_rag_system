from chunking.chunking import clean_text, overlapping_chunking, fixed_size_chunking, paragraph_chunking

if __name__ == "__main__":
    sample_text = """
    Retrieval Augmented Generation (RAG) is a technique that enhances LLMs by providing external knowledge sources during generation.

    It involves embedding documents into vector space and retrieving relevant context based on similarity search.

    The retrieved context is injected into the prompt before generating the final answer.
    """

    text = clean_text(sample_text)
    print("\nOverlapping chunking:")
    overlapping_chunks = overlapping_chunking(text, chunk_size=100, overlap=20)

    for i, chunk in enumerate(overlapping_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)

    print("\nFixed chunking:")
    fixed_chunks = fixed_size_chunking(text, chunk_size=100)

    for i, chunk in enumerate(fixed_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)

    print("\nParagraph chunking:")
    paragraph_chunks = paragraph_chunking(text, max_chunk_size=150)

    for i, chunk in enumerate(paragraph_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)