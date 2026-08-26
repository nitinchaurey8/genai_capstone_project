# RAG Deep Dive in This Project

## Generative AI
Generative AI models produce new text from an input prompt. This project uses Google Gemini through `ChatGoogleGenerativeAI` to generate an answer, but constrains the prompt to retrieved application context.

## What RAG Is
Retrieval-Augmented Generation combines a retrieval step with language-model generation. The system first finds relevant stored chunks, then supplies them as context to the model. Here, `retrieve_documents_with_scores` and `build_context` provide the retrieval side, while `generate_answer` performs generation.

## Why RAG Is Used
RAG lets the application answer questions about user-provided files without requiring the model's training data to contain those files. The prompt explicitly says to use only retrieved context and not invent information.

## RAG Versus Direct Generation
Direct generation sends a question to an LLM. This RAG implementation sends a question plus labeled chunks from Chroma. RAG can improve grounding and provide source metadata, but it remains dependent on extraction quality, chunking, embeddings, retrieval quality, and model compliance.

## Project Pipeline
```text
File -> Document -> Chunk -> Embedding -> Chroma
Question -> Query embedding/search -> Retrieved chunks -> Context -> Gemini -> Answer + sources
```

## Document Ingestion and Loading
`load_document` accepts `.pdf`, `.txt`, `.csv`, and `.xlsx`, case-insensitively. PDF pages become documents; a non-empty TXT becomes one document; CSV rows and XLSX rows become individual documents. Metadata includes filename and format location. Missing files raise `FileNotFoundError`; unsupported extensions raise `ValueError`; empty text or empty extracted content produces no documents.

## Chunking
`chunk_documents` uses `RecursiveCharacterTextSplitter`, size 1000 and overlap 200. It tries paragraph, line, sentence, space, then character boundaries. Metadata is copied to chunks and `chunk_id` is added sequentially. Overlap helps preserve meaning across neighboring chunks, while smaller chunks make retrieval more focused.

## Embeddings
An embedding is a numeric representation of text. `get_embedding_model` constructs `GoogleGenerativeAIEmbeddings` using `GOOGLE_EMBEDDING_MODEL`. Document chunks are embedded when added to Chroma; a question is embedded during similarity search. Blank queries are rejected and blank document input produces no vectors.

## Vector Storage and Similarity Search
Chroma persists vectors, text, and metadata in the `genai_capstone_documents` collection under `data/chroma`. Search requires a non-empty query and positive `k`. Chroma returns nearest documents, and the scored variant also returns numeric similarity/distance results as supplied by the integration.

## Retrieval and Context Construction
`retrieve_documents` returns documents only; `retrieve_documents_with_scores` retains pairs. `build_context` labels each chunk as `[Source n: filename (page/sheet/row)]` followed by content. The resulting string is passed as `{context}` in `RAG_SYSTEM_PROMPT`.

## Prompt and Answer Generation
The system prompt instructs Gemini to use only retrieved context, avoid outside knowledge and assumptions, answer concisely, and use a fixed fallback when context is insufficient. `generate_answer` handles missing context and empty model output. `_extract_response_text` supports string content and list content blocks.

## Source Attribution
`answer_question` returns source metadata from retrieved documents. The LangGraph path uses `build_safe_source`, restricting output fields to source, file type, page, sheet, and row. The Streamlit UI currently displays source, file type, and page.

## Benefits and Limitations
**Benefits:** multi-format input, persistent indexing, semantic retrieval, grounded prompting, metadata, and explicit fallbacks.

**Limitations:** no relevance threshold, reranker, citation verification, deduplication, or per-user store. Nearest-neighbor results can still be returned for unrelated questions when the collection is populated. Extracted PDF text may be empty for scanned/image-only PDFs.

## Common Failure Modes
- Missing Google key or model configuration.
- Unsupported, missing, empty, or malformed files.
- Empty question or invalid retrieval count.
- Chroma has no usable context.
- Embedding/model quota, authentication, or network failures.
- Repeated indexing creates additional entries.
- Running outside the repository root changes relative paths.

## Interview Concepts
Remember the distinction between indexing-time document embeddings and query-time query embeddings; chunk overlap is not retrieval; metadata supports attribution but is not automatically shown everywhere; and a fallback for empty context is not the same as a relevance threshold.
