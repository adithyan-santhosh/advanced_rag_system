from pipeline.rag_pipeline import RAGPipeline

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
