# Adaptive Hybrid RAG System
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![FAISS](https://img.shields.io/badge/FAISS-blue?style=for-the-badge&logo=facebook&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-RAG-orange?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-LLM-white?style=for-the-badge)

A production-oriented **Retrieval-Augmented Generation (RAG)** system built using open-source models.
The system supports **adaptive retrieval (Vector + Hybrid search), cross-encoder reranking, multi-document ingestion, FAISS persistence, and API deployment**.

This project demonstrates how to build a **robust, explainable, and scalable document QA system** using modern GenAI engineering practices.

---

# Project Motivation

Large Language Models often hallucinate when answering questions without context.
Retrieval-Augmented Generation (RAG) mitigates this by retrieving relevant information from a knowledge base before generating answers.

This project was designed to explore:

* High-quality **document retrieval**
* **Hybrid semantic + lexical search**
* **Cross-encoder reranking**
* **production-ready AI system architecture**

---

# System Architecture

User Query
↓
Query Embedding
↓
Adaptive Retrieval Decision
↓
Vector Retrieval (FAISS) OR Hybrid Retrieval (FAISS + BM25)
↓
Top-K Candidate Retrieval
↓
Cross-Encoder Reranking
↓
Top-3 Context Selection
↓
LLM Answer Generation
↓
Answer + Confidence Score + Sources

---

# Key Features

### Adaptive Retrieval

The system automatically chooses between retrieval strategies depending on the document structure.

**Vector Retrieval**

* Used for natural language documents
* Semantic similarity using embeddings

**Hybrid Retrieval**

* Used for code-heavy or ID-heavy documents
* Combines:

  * Vector similarity
  * BM25 lexical scoring

---

### Cross-Encoder Reranking

Initial retrieval returns Top-10 candidates.
A cross-encoder model reranks them to select the **Top-3 most relevant chunks**.

This improves answer accuracy significantly.

---

### Multi-Document Support

Users can upload multiple documents via the API.
All documents are indexed and queried together.

---

### FAISS Vector Database with Persistence

Embeddings are stored locally using FAISS.

Persistent storage includes:

* FAISS index
* chunked document text
* metadata

This allows fast startup without re-embedding documents.

---

### Source Attribution

The system returns document sources for transparency.

Example output:

```json
{
  "answer": "Zone F is restricted due to construction and GPS interference.",
  "confidence_score": 7.22,
  "sources": [
    "solarix.txt"
  ]
}
```

---

### Logging and Monitoring

System activity is logged for observability.

Logged events include:

* user queries
* confidence scores
* retrieval mode
* latency
* document uploads
* retrieval time
* generation time
* overall time
  
Example log entry:

```
QUERY | Mode=VECTOR | Confidence=7.21 | Latency=1.12s
UPLOAD | File=solarix.pdf | RebuildTime=2.3s
```

---

# Tech Stack

Language

* Python

LLM Inference

* Ollama (local models)

Embeddings

* SentenceTransformers

Vector Database

* FAISS

Hybrid Retrieval

* BM25 (rank-bm25)

Reranking

* CrossEncoder

API Framework

* FastAPI

Other Libraries

* NumPy
* Scikit-learn

---
# Architecture Diagram

<img width="1536" height="1024" alt="Adaptive_hybrid_rag_system_architecture_diagram" src="https://github.com/user-attachments/assets/264f913d-c374-43fc-bf6b-830077b300f3" />

---

# API Endpoints

Base path:

```
/rag-system
```

### Ask Question

```
POST /rag-system/ask
```

Example request:

```json
{
  "question": "Why is Zone F restricted?"
}
```

Example response:

```json
{
  "answer": "Operation in Zone F is strictly prohibited due to ongoing construction and interference with GPS signals.",
  "confidence_score": 6.819939613342285,
  "sources": {
    "documents": [
      "solaris_x2.txt"
    ]
  }
}
```

### Ask Question (Streaming Response)
Use this endpoint for real-time response generation. It is ideal for frontend "typewriter" effects where the answer appears word-by-word.

```
POST /rag-system/ask-stream
```

Example request:

```json
{
  "question": "Why is Zone F restricted?"
}
```

Example response:

```
The response is sent as a chunked stream (text/event-stream or text/plain). Each chunk represents a piece of the generated answer as the LLM processes it.
Example Stream Sequence:

Plaintext:
Operation
in
Zone
F
is
strictly
prohibited
due
to
ongoing
construction
and
interference
with
GPS
signals.

```
### Upload Document

```
POST /rag-system/upload
```

Supports:

* TXT
* PDF

PDF files are automatically converted to text before indexing.

---

### Health Check

```
GET /rag-system/health
```

Used to confirm API status.

---

# Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Start the API server:

```
uvicorn app.api.api:app --reload
```

API documentation will be available at:

```
http://localhost:8000/docs
```

---

# Benchmarking Experiments

Embedding models were benchmarked using:

* relevant queries
* irrelevant queries
* similarity separation
* Top-K retrieval accuracy

Metrics evaluated:

* Top-1 accuracy
* Top-3 accuracy
* similarity margin
* embedding latency

This helped select the most suitable embedding model for the system.

---

# Future Improvements

Planned enhancements include:

* inline answer citations
* query expansion
* dockerized deployment
* evaluation dashboard
* retrieval performance metrics

---

# Learning Outcomes

This project demonstrates practical knowledge of:

* Retrieval-Augmented Generation
* hybrid search systems
* vector databases
* LLM integration
* AI system deployment
* production-level API design

---

# Author

Adithyan Santhosh
AI / ML Engineer

Interested in building intelligent systems that transform unstructured data into actionable knowledge.
