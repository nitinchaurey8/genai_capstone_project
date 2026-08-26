# Configuration and Deployment

## Prerequisites
Use a supported Python environment and a working virtual environment. The repository's `commands.txt` documents setup commands, but its example directory is `C:\project\genai_capstone_project`; run equivalent commands from the actual repository root.

## Environment Setup
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Pinned dependencies include Streamlit, LangChain, LangGraph, Chroma integrations, Gemini integration, text splitters, PyPDF, pandas, OpenPyXL, and dotenv.

## Environment Variables
`app/utils/config.py` loads `.env` with `python-dotenv`. The active application path requires `GOOGLE_API_KEY`; it uses `GOOGLE_LLM_MODEL` and `GOOGLE_EMBEDDING_MODEL`, with source-defined defaults when model variables are absent. Do not include actual secret values in documentation or Git.

`.env.example` contains placeholders and does not mirror every variable read by `config.py`. OpenAI variables are loaded as retained backup configuration, but the application constructs Google clients and does not dynamically switch providers. The separate OpenAI connection test imports a package not listed in `requirements.txt`.

## Running the Application
Run from the repository root:

```powershell
streamlit run app\streamlit_app.py
```

The UI supports upload, processing, clearing the shared collection, and question answering. The relative `data/chroma` path assumes the current working directory is the project root.

## Running Tests
Most tests are executable modules:

```powershell
python -m tests.test_rag_chain
python -m tests.test_reliability
```

Review [GENAI_TESTING_GUIDE.md](GENAI_TESTING_GUIDE.md) before running the full set. Provider tests can make live calls and Chroma tests can mutate local runtime data.

## Git and Secrets
`.gitignore` excludes `.env`, virtual environments, Python caches, Streamlit secrets, temporary files, SQLite files, and `data/chroma/`. Keep API keys in local environment configuration or the deployment platform's secret store. Never paste keys into Markdown, source, test output, or commits.

## Chroma Runtime Data
The persistent collection is generated under `data/chroma`. It is intentionally excluded from Git, so deployment and fresh environments need an indexing step. `clear_vector_store` clears the entire collection, not only one upload.

## Deployment Considerations
The current implementation is suitable for a local or controlled demonstration. A multi-user deployment needs a deliberate storage isolation strategy, lifecycle management, duplicate handling, temporary-file cleanup, and provider secret configuration. These are deployment concerns, not currently implemented features.

## Common Setup Problems
- Missing `GOOGLE_API_KEY`: configuration import raises `ValueError`.
- Wrong model variable: verify the configured Google model names without exposing keys.
- Running from the wrong directory: relative Chroma/data paths resolve elsewhere.
- Missing dependency: activate the intended environment and install pinned requirements.
- OpenAI test import error: `langchain-openai` is not in the pinned requirements.
- Stale local collection: use the UI clear action only when clearing the shared collection is intended.
