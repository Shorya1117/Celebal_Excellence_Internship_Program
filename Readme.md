# Document Question Answering System using Retrieval-Augmented Generation (RAG)

> **Instructor Note**
>
> This README contains:
>
> - Project overview
> - Execution steps
> - System architecture
> - Validation procedure
> - Validation logs format
> - System metrics
> - Assignment requirement mapping
>
> The application generates validation logs dynamically at runtime when user queries are executed. The `my_documents` folder is intentionally kept empty in the repository so that any custom documents can be added during evaluation.

---

# Project Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline for answering questions from custom documents.

The system accepts user documents, converts them into embeddings, stores them inside a vector database, retrieves the most relevant document chunks for every query, and generates grounded answers using a language model.

---

# Supported Document Types

The system supports:

- PDF (.pdf)
- Text Files (.txt)
- Word Documents (.docx)

All documents should be placed inside:

```
my_documents/
```

The repository intentionally contains an empty `my_documents` folder.

During evaluation, any custom documents can be copied into this folder before creating the vector database.

---

# Project Structure

```
project/

│── app.py
│── setup_db.py
│── requirements.txt
│── README.md

│── my_documents/
│     (Place documents here)

│── local_chroma_db/
│     (Automatically created)

```

---

# Installation

## Clone Repository


---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Step 1 — Add Documents

Copy any supported files into

```
my_documents/
```

Example

```
my_documents/

English_12th.pdf


research.docx
```

---

# Step 2 — Build Vector Database

Run

```bash
python setup_db.py
```

The script performs

- Document Loading
- PDF Text Extraction
- Page Filtering
- Text Chunking
- Embedding Generation
- Vector Database Creation

Expected Output

```
Initializing Database Setup...

Scanning documents...

Loading...

Total Documents Loaded

Total Chunks

Generating Embeddings

Saving Chroma Database

Done.
```

---

# Step 3 — Run Application

```bash
streamlit run app.py
```

Application starts on

```
http://localhost:8501
```

---

# RAG Pipeline Architecture

```
                PDF / TXT / DOCX
                       │
                       ▼
              Document Loader
                       │
                       ▼
      Recursive Character Text Splitter
                       │
                       ▼
      HuggingFace Embedding Model
      all-MiniLM-L6-v2 (384 Dimensions)
                       │
                       ▼
                Chroma Vector DB
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Vector Retrieval             BM25 Search
          │                         │
          └────────────┬────────────┘
                       ▼
            Ensemble Retriever
                       │
                       ▼
        Cross Encoder Re-ranking
                       │
                       ▼
              Prompt Construction
                       │
                       ▼
          Language Model (FLAN-T5)
                       │
                       ▼
            Grounded Response
```

---

# Validation Procedure

The application can be validated using any custom documents.

Example Questions

```
Who is Sam Weiner?

Explain the central theme of The Third Level.

Why was the Maharaja called Tiger King?

What is Gondwana?

Why is Antarctica important?
```

Hallucination Test

```
What is Charley's phone number?

How many children did Sam have?

What is Louisa's profession?
```

Expected Behaviour

If the requested information is unavailable inside the uploaded documents, the system responds that the answer could not be found instead of generating unsupported information.

---

# Validation Logs

During execution, the application automatically prints validation logs in the terminal.

Logs include

- User Query
- Retrieved Chunks
- Retrieved Context
- Final Generated Answer

Example

```
QUERY:
Who is Sam Weiner?

Chunk 1
-----------------------------
Retrieved Context
-----------------------------

Chunk 2
-----------------------------
Retrieved Context
-----------------------------

Answer
-----------------------------
Generated grounded answer
-----------------------------
```

These logs demonstrate the retrieval performance and provide evidence that the generated answer is based on retrieved document chunks.

---

# System Metrics

| Component | Configuration |
|------------|---------------|
| Document Types | PDF, TXT, DOCX |
| Chunking Strategy | Recursive Character Splitter |
| Chunk Size | 600 |
| Chunk Overlap | 60 |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Embedding Dimension | 384 |
| Vector Database | ChromaDB |
| Retriever | Hybrid (Vector + BM25) |
| Search Strategy | MMR |
| Re-ranking | Cross Encoder |
| Cross Encoder Model | ms-marco-MiniLM-L-6-v2 |
| Language Model | Google FLAN-T5 Base |
| User Interface | Streamlit |

---

# Assignment Requirement Mapping

| Assignment Requirement | Status |
|------------------------|--------|
| Document Ingestion | ✅ |
| Text Chunking | ✅ |
| Embedding Generation | ✅ |
| Vector Database Creation | ✅ |
| Query Embedding | ✅ |
| Similarity Search | ✅ |
| Context Retrieval | ✅ |
| Prompt Construction | ✅ |
| Grounded Question Answering | ✅ |
| Hybrid Retrieval | ✅ |
| Re-ranking | ✅ |

---

# Instructor Evaluation Checklist

### Requirement 1

Operational end-to-end Question Answering Pipeline

✔ Supported

The application successfully performs

- Document ingestion
- Embedding generation
- Vector storage
- Hybrid retrieval
- Re-ranking
- Grounded answer generation

---

### Requirement 2

Validation Logs

✔ Supported

The application prints

- User Query
- Retrieved Context Chunks
- Generated Answer

during runtime for verification.

---

### Requirement 3

System Metrics Report

✔ Included in this README

This document includes

- Chunking configuration
- Embedding dimensions
- Vector database
- Retrieval strategy
- Re-ranking model
- Language model
- Overall system architecture

---

# Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Transformers
- Sentence Transformers
- BM25 Retriever
- Cross Encoder Re-ranking

---

# Author

Shorya Prajapat

