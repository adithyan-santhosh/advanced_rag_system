from core.pipeline.rag_pipeline import RAGPipeline

if __name__ == "__main__":

    rag = RAGPipeline()

    print("RAG CLI Mode. Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        result = rag.generate_answer(query)

        print("\nAnswer:")
        print(result["answer"])
        print("\nConfidence Score:", result["confidence_score"])
        print("=" * 60)
