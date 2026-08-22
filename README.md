# GenAI Capstone Project

## Retrieval-Augmented Generation Application

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to upload documents in multiple formats, process and index their content, retrieve relevant information using semantic search, and generate grounded answers using Google Gemini.

The project uses LangChain for document processing and model integration, LangGraph for workflow orchestration, ChromaDB for vector storage and similarity search, and Gemini embeddings for semantic retrieval.

---

## 1. Project Overview

The GenAI Capstone Project demonstrates a complete Retrieval-Augmented Generation pipeline.

Users can upload documents in the following formats:

- PDF
- TXT
- CSV
- XLSX

The application processes the uploaded documents, divides them into smaller chunks, generates vector embeddings, stores the embeddings in ChromaDB, retrieves relevant content for a user question, and generates an answer using a language model.

The generated response is grounded in the retrieved document context and includes source information.

---

## 2. Project Objectives

The main objectives of this project are:

1. Build a functional RAG application.
2. Support multiple document formats.
3. Implement document loading and processing.
4. Implement text chunking.
5. Generate embeddings for document content.
6. Store embeddings in a vector database.
7. Implement semantic document retrieval.
8. Generate answers using retrieved context.
9. Use LangGraph to orchestrate the RAG workflow.
10. Provide source attribution.
11. Implement reliability and safety controls.
12. Provide a user-friendly Streamlit interface.

---

## 3. Key Features

### Multi-format document upload

The application supports:

- PDF documents
- Text files
- CSV files
- Excel XLSX files

### Document processing

Uploaded documents are:

1. Loaded
2. Converted into application documents
3. Split into chunks
4. Embedded
5. Stored in ChromaDB

### Semantic retrieval

The user's question is converted into an embedding and compared with stored document embeddings to retrieve relevant content.

### Grounded answer generation

The retrieved content is provided to the language model as context so that the answer is based on the uploaded documents.

### Source attribution

The application tracks the source documents associated with retrieved content and returns source information with the answer.

### LangGraph orchestration

LangGraph is used to organize the RAG processing flow into multiple steps.

### Reliability and safety

The application includes validation and controlled handling for:

- Empty questions
- Whitespace-only questions
- Unsupported file types
- Empty uploads
- Invalid retrieval parameters
- Missing retrieved context
- API errors
- General application errors
- Source metadata

---

## 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Application programming language |
| Streamlit | User interface |
| LangChain | LLM and document-processing framework |
| LangGraph | RAG workflow orchestration |
| ChromaDB | Vector database |
| Google Gemini | Language model |
| Gemini Embeddings | Document and query embeddings |
| PyPDF | PDF processing |
| Pandas | CSV processing |
| OpenPyXL | XLSX processing |
| python-dotenv | Environment variable management |

---

## 5. AI Models

### Language Model

The current application uses:

```text
gemini-3.1-flash-lite