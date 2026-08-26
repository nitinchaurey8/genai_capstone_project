# Embeddings, Vector Databases, and ChromaDB

## Embeddings
An embedding maps text to a numeric vector so semantically related text can be compared mathematically. The project uses `GoogleGenerativeAIEmbeddings` from `app/embeddings/embedding_service.py` for both document chunks and questions.

## Vector Database
A vector database stores vectors alongside text and metadata and supports nearest-neighbor search. ChromaDB is used through `langchain_chroma.Chroma`.

## Project Configuration
- Directory: `data/chroma`
- Collection: `genai_capstone_documents`
- Default search count: 4
- Graph-normalized top-k range: 1 through 8
- Embedding model: configured by `GOOGLE_EMBEDDING_MODEL`

## Indexing Lifecycle
```text
loaded Document -> chunk -> Chroma.add_documents
                                  |
                           embedding function
                                  |
                      persistent vector + metadata
```

`add_documents` opens the store and adds chunks. Chroma invokes the configured embedding function. `get_document_count` reads the collection count. `clear_vector_store` deletes all IDs in the shared collection.

## Similarity Search
At question time, Chroma embeds the query using the same configured embedding function and returns nearest stored chunks. `similarity_search_with_scores` preserves each document-score pair; the application returns scores in `RAGState` but does not apply a relevance threshold.

## Metadata and Sources
Metadata is stored with each document/chunk: filename and file type, plus PDF page or spreadsheet sheet/row. `build_context` uses location metadata in labels. `build_safe_source` limits the output dictionary to source, file type, page, sheet, and row.

## Runtime Data and Git
The repository's `.gitignore` excludes `data/chroma/` and SQLite runtime files. This keeps generated local index state out of source control, avoids committing derived data, and prevents accidental exposure of indexed content. It also means a fresh checkout needs documents indexed again.

## Common Concepts
- **Vector:** numeric embedding representation.
- **Dimension:** number of values in a vector; the application does not hard-code a dimension in its source.
- **Collection:** named Chroma grouping of stored entries.
- **Metadata:** searchable/traceable attributes associated with text.
- **Top-k:** number of nearest results requested.
- **Distance/score:** numeric ordering signal returned by the store integration; no threshold is applied here.

## Interview/Viva Questions
**Why embed both documents and queries?** They must be represented in the same vector space for comparison.

**Why preserve metadata?** It supports source attribution and location-aware context.

**Does Chroma store only vectors?** The application adds documents, so text and metadata accompany the vector entries.

**What does clear do?** It deletes every ID in the shared configured collection.

**Is the store isolated per user?** No. The current implementation uses one persistent collection.

**Is low relevance rejected?** No explicit threshold or reranker exists; nearest neighbors are returned when available.
