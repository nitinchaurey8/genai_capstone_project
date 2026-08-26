# Troubleshooting

## Import or Environment Errors
- **Symptom:** Import fails for Streamlit, LangChain, or Chroma. **Likely cause:** wrong interpreter or dependencies missing. **Diagnose:** activate `venv`, run `pip install -r requirements.txt`, and use the same `python` to run the app. **Safe solution:** use the project environment. **Relevant:** `requirements.txt`, `commands.txt`, `tests/test_streamlit_import.py`.
- **Symptom:** `GOOGLE_API_KEY is not configured`. **Cause:** missing environment value. **Diagnose:** verify the local secret configuration without printing it. **Solution:** configure the key through the environment/deployment secret store. **Relevant:** `app/utils/config.py`.

## Dependency and Provider Problems
- **Symptom:** Gemini call fails, times out, or reports quota/authentication. **Cause:** key, network, quota, permissions, or model configuration. **Diagnose:** inspect the sanitized UI error and run the relevant live connection test only with valid local configuration. **Solution:** correct provider configuration or access; do not hard-code keys. **Relevant:** `app/rag/rag_chain.py`, `app/embeddings/embedding_service.py`, `tests/test_google_connection.py`.
- **Symptom:** OpenAI connection test cannot import. **Cause:** `langchain-openai` is not in `requirements.txt`. **Diagnose:** inspect the test import and pinned requirements. **Solution:** treat it as an environment/dependency mismatch; do not claim OpenAI is an active application path. **Relevant:** `tests/test_openai_connection.py`.

## ChromaDB Problems
- **Symptom:** counts or search results are unexpected. **Cause:** persistent data from earlier runs or repeated indexing. **Diagnose:** inspect collection state through tests or the app. **Solution:** use Clear Documents only when clearing the shared collection is intended, then re-index. **Relevant:** `app/vectorstore/chroma_store.py`.
- **Symptom:** data is stored in an unexpected location. **Cause:** app started outside repository root. **Diagnose:** check current directory. **Solution:** run from project root because `data/chroma` is relative. **Relevant:** `app/vectorstore/chroma_store.py`, `commands.txt`.

## Document Loading
- **Symptom:** file rejected. **Cause:** extension is not PDF/TXT/CSV/XLSX. **Diagnose:** check the filename suffix. **Solution:** use a supported format. **Relevant:** `app/ingestion/document_loader.py`, `app/reliability/safety.py`.
- **Symptom:** no documents loaded. **Cause:** empty text, empty PDF extraction, empty table, or unsupported content. **Diagnose:** test the loader directly and inspect file content. **Solution:** provide non-empty text/table data; scanned PDFs may need OCR, which is not implemented. **Relevant:** loader tests.
- **Symptom:** CSV/XLSX rows look different than expected. **Cause:** pandas parsing/header or worksheet structure. **Diagnose:** inspect resulting `Document` content and metadata. **Solution:** use a readable table with expected headers/sheets. **Relevant:** `document_loader.py`.

## Retrieval and Generation
- **Symptom:** answer says there is not enough information. **Cause:** empty collection, blank retrieved content, or unavailable model context. **Diagnose:** confirm indexing summary and Chroma count; test retrieval. **Solution:** index valid documents and ask a document-related question. **Relevant:** `retriever.py`, `rag_chain.py`, `rag_graph.py`.
- **Symptom:** unrelated question still returns an answer. **Cause:** no semantic relevance threshold is implemented. **Diagnose:** inspect returned scores and sources. **Solution:** treat thresholding/reranking as a future enhancement, not an existing setting. **Relevant:** `retriever.py`, `GENAI_RAG_DEEP_DIVE.md`.

## LangGraph and Streamlit
- **Symptom:** graph reports a safe error/status. **Cause:** validation, retrieval, or generation stage failed. **Diagnose:** inspect `status`, `error`, and `retrieval_scores` in a controlled test. **Solution:** fix the underlying input/configuration and retry. **Relevant:** `app/graph/rag_graph.py`.
- **Symptom:** Streamlit starts but UI action fails. **Cause:** upload parsing, indexing, or provider exception. **Diagnose:** use the generic UI message and narrow backend tests; internal details are intentionally hidden. **Solution:** validate file and environment, then retry. **Relevant:** `app/streamlit_app.py`.

## Git and Windows
- **Symptom:** secrets appear in a status/diff. **Cause:** ignored-file configuration or accidental tracking. **Diagnose:** inspect Git status and ignore rules without displaying secret contents. **Solution:** remove secrets from tracked history using an approved repository process and rotate exposed credentials; keep `.env` local. **Relevant:** `.gitignore`.
- **Symptom:** PowerShell activation fails. **Cause:** execution policy or wrong path. **Diagnose:** confirm the venv path and current shell. **Solution:** use the repository's environment activation instructions or an approved process-scoped policy; do not bypass security broadly. **Relevant:** `commands.txt`.

## Test Failures
Run scripts from the repository root with the intended interpreter. Separate offline assertion failures from live provider/network failures. Chroma tests may alter ignored runtime data; stale state can affect counts. See [GENAI_TESTING_GUIDE.md](GENAI_TESTING_GUIDE.md).
