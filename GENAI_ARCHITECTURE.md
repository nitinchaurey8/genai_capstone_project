# GenAI Architecture

## Overview
The application is layered around a persistent document index and a question-time RAG workflow.

```text
                        +----------------+
                        | Streamlit UI   |
                        | streamlit_app  |
                        +-------+--------+
                                |
              +-----------------+------------------+
              |                                    |
       upload/index path                    question path
              |                                    |
   document_loader -> indexer              LangGraph graph
              |                                    |
          chunker                         validate_node
              |                                    |
      embedding_service                   retrieve_node
              |                                    |
        chroma_store <------------------- context_node
              ^                                    |
              |                              generate_node
              +------------------------------ sources_node
```

## Presentation Layer
`app/streamlit_app.py` configures the page, accepts multiple uploads, processes valid files, shows indexing counts, accepts a question, invokes `run_rag_graph`, and renders answers or safe errors. It also calls `clear_vector_store` for a full collection reset. Streamlit session state tracks whether documents were indexed, indexed filenames, and the indexing summary.

## Document Ingestion Layer
`app/ingestion/document_loader.py` validates existence and extensions, then dispatches to `_load_pdf`, `_load_txt`, `_load_csv`, or `_load_xlsx`. `app/ingestion/indexer.py` coordinates loaded documents, chunking, embedding-model initialization, storage, and summary counts.

## Document Processing Layer
`app/processing/chunker.py` uses `RecursiveCharacterTextSplitter` with separators ordered as paragraph, line, sentence, space, and character. It preserves metadata and assigns sequential `chunk_id` values.

## Embedding Layer
`app/embeddings/embedding_service.py` constructs `GoogleGenerativeAIEmbeddings` from configuration. `embed_query` rejects blank input; `embed_documents` strips text and returns no vectors for empty input. Chroma receives the embedding function for both indexing and search.

## Vector Storage Layer
`app/vectorstore/chroma_store.py` opens a persistent Chroma store at relative path `data/chroma`, collection `genai_capstone_documents`. It exposes add, similarity search, scored search, count, and clear operations. The path is relative to the process working directory.

## Retrieval Layer
`app/retrieval/retriever.py` delegates scored search to Chroma, returns documents or document-score pairs, and formats context using source, page, sheet, and row metadata. Its default is four results.

## RAG Generation Layer
`app/rag/rag_chain.py` creates `ChatGoogleGenerativeAI` with configured model and temperature 0.0. `generate_answer` rejects a blank question, returns a fixed fallback for absent/blank context, formats the grounded prompt, invokes Gemini, and normalizes string or list content. `answer_question` is a direct complete-flow helper returning answer, sources, and retrieved documents.

## LangGraph Orchestration Layer
`RAGState` is a `TypedDict` carrying question, top_k, documents, context, answer, sources, retrieval scores, error, and status. `build_rag_graph` connects `START -> validate -> retrieve -> context -> generate -> sources -> END`. Nodes return updated state rather than raising expected workflow errors to the UI.

## Reliability and Safety Layer
`app/reliability/safety.py` defines supported extensions, top-k limits 1..8, user-safe messages, `validate_question`, file validators, `has_retrieved_context`, `sanitize_error_message`, and `build_safe_source`. Graph nodes use these helpers to classify validation, retrieval, no-context, and generation outcomes.

## Configuration Layer
`app/utils/config.py` calls `load_dotenv`, reads Google key/model variables, reads OpenAI values retained as backup configuration, and raises on missing Google API key at import. The application code constructs Google clients; it does not switch providers dynamically.

## End-to-End Document Flow
1. Streamlit writes upload bytes to a temporary file and restores the original filename in metadata.
2. `load_document` creates format-specific LangChain `Document` objects.
3. `index_documents` calls `chunk_documents`.
4. The configured Gemini embedding model is verified.
5. Chunks are added to Chroma, which embeds and persists them.
6. A summary reports loaded documents, chunks, vectors, and distinct sources.

## End-to-End Question Flow
1. The UI validates the text and indexed-document state.
2. `run_rag_graph` initializes `RAGState`.
3. `validate_node` strips the question and normalizes top-k.
4. `retrieve_node` obtains documents and scores.
5. `context_node` checks usable content and calls `build_context`.
6. `generate_node` calls `generate_answer`.
7. `source_node` restricts metadata for display.
8. Streamlit renders answer and available sources.

## Module Relationships
```text
config -> embedding_service -> chroma_store -> retriever
loader -> indexer -> chunker -> chroma_store
retriever -> rag_chain
safety -> streamlit_app and rag_graph
rag_chain + retriever + safety -> rag_graph -> streamlit_app
```

## Boundaries and Limitations
The Chroma collection is shared and persistent, clearing removes every stored entry, and repeated indexing adds vectors without visible deduplication. The UI validates files one at a time rather than calling `validate_uploaded_files`. These are current behaviors, not hidden architecture.
