# GenAI Capstone Project Guide

## Purpose
This project is a Python and Streamlit Retrieval-Augmented Generation (RAG) application. Users upload PDF, TXT, CSV, or XLSX documents, index their content, and ask questions answered by Google Gemini using retrieved document context.

## Problem Statement and Objectives
A general-purpose LLM may answer from knowledge outside a user's documents. This application adds document retrieval so answers are grounded in an uploaded collection. The implemented objectives are multi-format ingestion, chunking, Gemini embeddings, persistent ChromaDB storage, semantic retrieval, grounded generation, source metadata, LangGraph orchestration, and input/error handling.

## Implemented Capabilities
- Upload multiple supported files in the Streamlit UI.
- Load PDF pages, non-empty TXT files, CSV rows, and XLSX worksheet rows as LangChain `Document` objects.
- Split content with `RecursiveCharacterTextSplitter` using size 1000 and overlap 200.
- Embed and persist chunks in the `genai_capstone_documents` Chroma collection at `data/chroma`.
- Retrieve four chunks by default, with graph top-k normalized to 1 through 8.
- Generate a concise answer with `ChatGoogleGenerativeAI` at temperature 0.0.
- Return source metadata and retrieval scores through the graph state.
- Validate questions and files and sanitize user-facing errors.

## Technology Stack
| Technology | Role |
|---|---|
| Python | Application language |
| Streamlit 1.60.0 | UI |
| LangChain 1.3.14 | Documents, prompts, integrations |
| LangGraph 1.2.9 | Ordered RAG workflow |
| ChromaDB 1.5.9 / `langchain-chroma` | Persistent vector storage and search |
| Google Gemini integration 4.3.2 | Chat model and embeddings |
| PyPDF | PDF text extraction |
| pandas / OpenPyXL | CSV and XLSX loading |
| python-dotenv | Environment loading |

## High-Level Workflow
```text
Upload -> Load Documents -> Chunk -> Gemini Embeddings -> ChromaDB
Question -> Validate -> Retrieve -> Build Context -> Gemini Answer -> Safe Sources
```

## Architecture and Data Flow
`app/streamlit_app.py` is the presentation entry point. Ingestion uses `document_loader.py` and `indexer.py`; processing uses `chunker.py`; embeddings use `embedding_service.py`; persistence uses `chroma_store.py`; retrieval uses `retriever.py`; generation uses `rag_chain.py`; orchestration uses `graph/rag_graph.py`; validation and sanitization use `reliability/safety.py`; settings come from `utils/config.py`.

During ingestion, each loaded `Document` carries `source` and format-specific metadata. Chunking preserves metadata and adds `chunk_id`. Chroma stores the chunks and computes embeddings through the configured Gemini embedding model. During question answering, the query is searched against the collection, retrieved content is labeled by source/location, and that context is placed in the grounded prompt.

## Application Components
- **Presentation:** upload, process, clear, question, answer, and source controls.
- **Ingestion:** `load_document`, format-specific private loaders, `index_documents`, and `index_uploaded_file`.
- **Processing:** `chunk_documents`, `CHUNK_SIZE`, and `CHUNK_OVERLAP`.
- **Embedding:** `get_embedding_model`, `embed_query`, and `embed_documents`.
- **Storage:** `get_vector_store`, add/search/count/clear functions.
- **Retrieval:** `retrieve_documents`, scored retrieval, and `build_context`.
- **Generation:** `get_llm`, `_extract_response_text`, `generate_answer`, and `answer_question`.
- **Graph:** `RAGState`, five nodes, `build_rag_graph`, and `run_rag_graph`.
- **Reliability:** validation, top-k normalization, context checks, error sanitization, and safe sources.

## Source Attribution
Loaded metadata records filename and, where applicable, PDF page, XLSX sheet, and row. `build_context` labels retrieved chunks with these locations. `source_node` uses `build_safe_source` to expose only source, file type, page, sheet, and row. The current Streamlit display renders source, file type, and PDF page; sheet and row are retained in backend results but are not displayed by the UI.

## Reliability and Safety
Implemented protections include empty/whitespace question validation, supported-extension checks, empty-upload messages, top-k bounds, usable-context checks, a no-context fallback, generic provider/general error messages, and restricted source metadata. The prompt instructs Gemini to use only retrieved context. These controls reduce accidental disclosure and unsupported answers but do not prove factual correctness.

## Testing Strategy
The `tests/` directory contains script-style tests using `main()` and assertions, plus a helper pytest-style fixture in `test_real_documents.py`. Tests cover loaders, chunking, embeddings, Chroma operations, retrieval, indexing, RAG, LangGraph, safety, Streamlit import, real documents, and provider connectivity. Some provider and vector-store tests make live API calls or mutate persistent Chroma data. Tests were not run as part of this documentation work, so no pass claim is made here. See [GENAI_TESTING_GUIDE.md](GENAI_TESTING_GUIDE.md).

## Directory Structure
```text
app/
  streamlit_app.py
  embeddings/ embedding_service.py
  graph/ rag_graph.py
  ingestion/ document_loader.py, indexer.py
  processing/ chunker.py
  rag/ rag_chain.py
  reliability/ safety.py
  retrieval/ retriever.py
  utils/ config.py
  vectorstore/ chroma_store.py
data/                 fixtures, sample documents, ignored Chroma runtime data
tests/                component, integration, and provider checks
requirements.txt      pinned dependencies
commands.txt          setup and verification commands
```

## Lessons Learned
Implemented RAG depends on document boundaries, metadata, embedding configuration, persistent storage, retrieval limits, and model behavior together. Retrieval scores are returned but no relevance threshold is implemented. A populated collection can return nearest neighbors for an unrelated question, so the no-context fallback is not a semantic relevance detector.

## Future Enhancements
These are not implemented: deduplication and document versioning, per-user collections, cleanup of temporary upload files, displaying sheet/row sources, a relevance threshold or reranker, broader UI end-to-end tests, and a provider abstraction supported by the pinned requirements. Deployment should also address shared-store concurrency and secrets management.

## Related Guides
See [GENAI_ARCHITECTURE.md](GENAI_ARCHITECTURE.md), [GENAI_RAG_DEEP_DIVE.md](GENAI_RAG_DEEP_DIVE.md), and [GENAI_TROUBLESHOOTING.md](GENAI_TROUBLESHOOTING.md).
