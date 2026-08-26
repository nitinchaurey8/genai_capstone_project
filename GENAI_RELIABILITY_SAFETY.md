# Reliability and Safety

## Why It Matters
RAG quality depends on valid input, usable retrieval, controlled model calls, and safe presentation of errors and metadata. This project centralizes these concerns in `app/reliability/safety.py` and uses them in the UI and LangGraph.

## Implemented Controls
| Control | Implementation |
|---|---|
| Empty/whitespace question | `validate_question` returns `EMPTY_QUESTION_MESSAGE` |
| File validation | `validate_file_extension` accepts PDF/TXT/CSV/XLSX |
| Empty upload | `validate_uploaded_files` returns `NO_DOCUMENTS_MESSAGE` |
| Top-k validation | `validate_top_k` accepts integer 1 through 8 |
| Top-k normalization | `normalize_top_k` clamps values to 1..8; non-integers use 4 |
| Context validation | `has_retrieved_context` requires a document with non-blank content |
| Fallback | `NO_CONTEXT_MESSAGE` / `FALLBACK_ANSWER` |
| Error sanitization | `sanitize_error_message` maps provider-like failures to generic messages |
| Source sanitization | `build_safe_source` exposes selected metadata only |

## Question and File Behavior
The UI validates a question before graph invocation and warns when documents have not been indexed. It validates uploaded extensions individually. The collection-level validator exists and is tested but is not the UI call path.

## Retrieved Context and Fallback
An empty document list or documents with only blank content leads to `no_context` in LangGraph and skips generation. `generate_answer` also returns the fallback for empty documents or blank built context. This is not a semantic relevance threshold: populated Chroma may still return nearest neighbors for unrelated questions.

## API and General Errors
`sanitize_error_message` looks for API/model/provider indicators such as API, quota, rate limit, authentication, permission, Google, Gemini, or model. It returns the generic LLM message for those cases and a generic processing message otherwise. Internal exception text is not placed in user-facing graph state.

## Source and Configuration Protection
`build_safe_source` limits metadata to source, file type, page, sheet, and row. Configuration is loaded from environment variables; secrets are not part of this documentation. `.env` is ignored by Git and `.env.example` contains placeholders rather than credentials.

## Reliability-Aware Graph
Statuses include `started`, `validated`, `validation_failed`, `retrieved`, `retrieval_failed`, `context_ready`, `no_context`, `generated`, and `generation_failed`. Nodes preserve failed states and avoid inappropriate downstream work.

## Tested Areas
`tests/test_reliability.py`, `tests/test_rag_graph_reliability.py`, and related retriever/RAG tests cover question, file, top-k, context, sanitization, graph status, and fallback behavior. The tests were not run during documentation generation.

## Known Limitations
Validation does not verify file contents before parsing; sanitization is indicator-based; no prompt-injection detector or citation verifier is present; no relevance threshold is applied; and the shared Chroma collection is not isolated between users.
