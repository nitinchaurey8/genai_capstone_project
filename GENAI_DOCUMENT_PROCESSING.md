# Document Processing Pipeline

## Supported Files
`SUPPORTED_EXTENSIONS` accepts `.pdf`, `.txt`, `.csv`, and `.xlsx`, case-insensitively. The Streamlit uploader exposes the same four formats.

## Loading
`load_document(file_path)` first checks existence and extension. It dispatches to format-specific loaders returning `list[Document]`.

| Format | Document granularity | Metadata |
|---|---|---|
| PDF | One per non-empty page | source, file_type, page |
| TXT | One document if non-empty | source, file_type |
| CSV | One per non-empty row | source, file_type, row |
| XLSX | One per non-empty row on every sheet | source, file_type, sheet, row |

TXT uses UTF-8 with replacement for invalid bytes. CSV/XLSX rows are converted to `column: value` lines. Empty text, empty extracted pages, or empty row output produces no documents.

## Chunking
`chunk_documents` returns early for empty input. It uses `RecursiveCharacterTextSplitter` with `CHUNK_SIZE = 1000` and `CHUNK_OVERLAP = 200`, then preserves metadata and adds a sequential `chunk_id`.

## Indexing
`index_documents` performs:
1. Reject empty input.
2. Chunk documents and reject empty chunks.
3. Initialize/verify the Gemini embedding model.
4. Add chunks to ChromaDB.
5. Reject a missing ID result.
6. Return `documents_loaded`, `chunks_created`, `vectors_created`, and distinct `sources`.

`index_uploaded_file` checks a filesystem path, loads it, and delegates to `index_documents`.

## Streamlit Upload Behavior
The UI validates each uploaded filename, writes bytes to a `delete=False` temporary file, loads it, then replaces temporary-path source metadata with the original filename. Invalid files are skipped while valid files continue. Temporary files are not visibly deleted in the current implementation.

## Retrieval After Indexing
Stored chunks are searched by `similarity_search_with_scores`; `retriever.py` extracts documents and builds source-labeled context for Gemini.

## Error Handling
Missing files and unsupported extensions raise from the loader. Empty documents and failed indexing stages raise `ValueError`. The Streamlit layer catches processing exceptions and shows a generic message.

## Test Coverage
The test suite covers TXT, CSV, XLSX, and PDF loaders; unsupported extensions; empty inputs; chunk size/overlap/metadata/chunk IDs; indexing counts/storage/retrieval; embeddings; and Chroma operations. See [GENAI_TESTING_GUIDE.md](GENAI_TESTING_GUIDE.md). Tests were not run for this documentation task.

## Known Boundaries
PDF extraction is text-based through PyPDF, so image-only PDFs may yield no documents. Reprocessing the same source adds another vector set. There is no deduplication or per-session collection.
