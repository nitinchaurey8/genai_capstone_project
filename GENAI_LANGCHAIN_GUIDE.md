# LangChain Guide

## What LangChain Does Here
LangChain supplies the `Document` abstraction, text splitter, prompt template, and Google/Chroma integrations used by the application. It is the component framework around the RAG pipeline; LangGraph separately controls the ordered state workflow.

## Components Actually Used
| Project use | API/component |
|---|---|
| Loaded content | `langchain_core.documents.Document` |
| Splitting | `RecursiveCharacterTextSplitter` |
| Prompting | `ChatPromptTemplate.from_messages` |
| Chat model | `ChatGoogleGenerativeAI` |
| Embeddings | `GoogleGenerativeAIEmbeddings` |
| Vector store | `langchain_chroma.Chroma` |

## Documents and Loading
`document_loader.py` constructs `Document(page_content=..., metadata=...)`. It uses PyPDF and pandas for extraction, then hands standard LangChain documents to the rest of the pipeline. LangChain is not the file parser for every format here; the project uses format-specific libraries before constructing `Document` objects.

## Text Splitting
`chunker.py` creates `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`, `length_function=len`, and explicit separators. `split_documents` preserves document metadata on generated chunks.

## Embeddings and Chroma
`embedding_service.py` creates the Google embedding integration. `chroma_store.py` gives `Chroma` the embedding function, collection name, and persistent directory. Calling `add_documents` lets the vector-store integration embed and store chunks; similarity search embeds the query through the same configured function.

## Retrieval
`retriever.py` wraps `similarity_search_with_score`, extracts documents for ordinary retrieval, and formats them into application context. It adds application-specific validation and metadata formatting around the LangChain/Chroma call.

## Prompt and LLM Integration
`RAG_PROMPT` is a `ChatPromptTemplate` with a system message containing `{context}` and a human message containing `{question}`. `generate_answer` calls `format_messages`, invokes the configured `ChatGoogleGenerativeAI`, then normalizes the response content.

## Important Functions
- `load_document`: application loader returning LangChain documents.
- `chunk_documents`: splitter boundary.
- `get_embedding_model`: Gemini embedding factory.
- `get_vector_store`: persistent `Chroma` factory.
- `build_context`: application context formatter.
- `get_llm`: Gemini chat factory.
- `generate_answer`: grounded generation boundary.

## LangChain Data Flow
```text
Document -> RecursiveCharacterTextSplitter -> Document chunks
chunks + embedding function -> Chroma
question -> Chroma similarity_search_with_score -> Documents
Documents -> ChatPromptTemplate -> ChatGoogleGenerativeAI
```

## Why It Helps
The integrations standardize document objects and model/vector-store interfaces, reducing glue code between ingestion, retrieval, and generation. It also makes the model and store boundaries visible and testable.

## Limitations in This Project
The application uses direct helper calls rather than a single declarative LangChain retrieval chain. Google is the active implementation. OpenAI values are loaded in configuration but are not used by the application path, and the pinned requirements do not include `langchain-openai` for the separate OpenAI connection test.

## Interview/Viva Questions
**Why use `Document`?** It keeps text and metadata together across loading, chunking, retrieval, and attribution.

**Why use a splitter?** Large files must be divided into retrievable units; overlap preserves boundary context.

**How is grounding implemented?** `RAG_SYSTEM_PROMPT` supplies retrieved context and instructs the model to use only it.

**Does LangChain perform all parsing?** No. PyPDF and pandas parse files; LangChain represents and processes the resulting text.

**Where is orchestration?** LangGraph in `app/graph/rag_graph.py`, not LangChain itself.
