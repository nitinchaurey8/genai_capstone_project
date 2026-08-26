# Testing and Validation Guide

## Philosophy
Tests are mostly executable script modules with `main()` and `assert` statements. They validate components and integrations at their existing boundaries. Live-provider tests require configuration and network access. This guide describes coverage, not a claim that the suite passed; tests were not run during documentation creation.

## Test Inventory
| Test file | What it covers |
|---|---|
| `test_document_loader.py` | Supported/unsupported loading behavior |
| `test_pdf_loader.py` | PDF extraction |
| `test_real_documents.py` | Fixture/real document loading, including PDF |
| `test_chunker.py` | Size, overlap, metadata, IDs, empty input |
| `test_embedding_service.py` | Embedding helpers and empty inputs |
| `test_google_embedding.py` | Live Google embedding behavior |
| `test_google_connection.py` | Live Google connection |
| `test_openai_connection.py` | Live OpenAI connection import/use |
| `test_chroma_store.py` | Chroma add/search/count/clear and validation |
| `test_chroma.py` | Additional Chroma behavior |
| `test_retriever.py` | Retrieval, scores, context, invalid input |
| `test_indexer.py` | Indexing counts, storage, retrieval, empty input |
| `test_rag_chain.py` | Generation, complete flow, empty question, fallback |
| `test_rag_graph.py` | Graph construction and execution |
| `test_rag_graph_reliability.py` | Graph validation/errors/fallback/sources |
| `test_reliability.py` | Safety validators and sanitization |
| `test_streamlit_import.py` | Importability of Streamlit module |

## Coverage Themes
- **Unit/component:** loader, chunker, embedding helpers, safety helpers, retriever formatting.
- **Integration:** indexer with embeddings/Chroma, retrieval with stored data, RAG chain.
- **LangGraph:** graph build, state flow, answers, sources, scores, empty questions.
- **UI smoke check:** Streamlit import only; visible upload-to-answer workflow is not clearly exercised end to end.
- **Provider checks:** Google and OpenAI connection scripts are live checks rather than offline unit tests.

## Commands
The repository documents environment setup in `commands.txt`. Individual scripts can be run from the repository root, for example:

```powershell
python -m tests.test_rag_chain
python -m tests.test_reliability
```

Because the files are script-style, inspect each module's `main()` for its exact invocation behavior. A standard pytest discovery run may not collect every assertion because most tests do not use conventional `test_*` functions.

## Expected Outcomes
A successful script reaches its completion output without an assertion failure. Live tests additionally require valid provider configuration, network access, compatible dependencies, and a usable Chroma directory. No universal expected output or pass result is asserted here.

## Test Data
Fixtures include `data/loader_test/test.txt`, `test.csv`, `test.xlsx`, and sample files under `data/test_documents/`, plus `data/test_document.pdf` according to the repository inventory. Chroma tests can mutate ignored persistent runtime data.

## Troubleshooting Failures
- Confirm execution from the repository root because paths are relative.
- Activate the project virtual environment and install `requirements.txt`.
- Check placeholder-based environment configuration without printing secrets.
- For provider failures, distinguish missing key, quota/authentication, network, and model-name issues.
- Clear or isolate the local Chroma collection when stale test data affects counts.
- Note that `test_openai_connection.py` imports `langchain_openai`, which is not pinned in `requirements.txt`.
- Do not interpret a live-provider failure as proof that offline application logic is wrong.
