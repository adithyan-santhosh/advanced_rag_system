from pipeline.rag_pipeline import RAGPipeline

DOCUMENT_TEXT = """
[SECTION 01: INVENTORY DATA] Beginning sync for hardware batch INV-2024-AX9347. The system identified the following hardware components registered under UID: 88472-99102-ZZ-001.

Status Code: ERR_SIG_404_NOT_FOUND

Asset Key: K-92837465-V2

Voltage Ref: 12.05V / 0.05A

Timestamp: 20260223115757

[SECTION 02: TRANSACTIONAL STRINGS] Validating transaction sequence TXN-8847-PL-Q9. Please ensure the following strings match the encrypted ledger output:
AUTH_TOKEN_ACTIVE_VERIFIED

SYS_PROCESS_MGMT_SIG_88

REF_CODE: 000-111-222-333-ABCD-EFGH

[SECTION 03: NUMERIC DENSITY CHECK] 99.45% of nodes are active. Parameters: 10.0.0.1, 192.168.1.254, 8080, 443. Latitude/Longitude: 9.9312° N, 76.2673° E. Floating point offset detected at 0.000000451.
"""

if __name__ == "__main__":

    rag = RAGPipeline(DOCUMENT_TEXT)

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
