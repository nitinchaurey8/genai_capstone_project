# Interview and Viva Guide

## Beginner Questions

**What is Generative AI?** A model-based approach that generates new content; this project uses Gemini to generate document-grounded answers.

**What is RAG?** Retrieval-Augmented Generation retrieves relevant chunks and supplies them to an LLM before generation.

**Why use RAG?** It connects answers to uploaded documents and supports source metadata instead of relying only on model pretraining.

**What formats are supported?** PDF, TXT, CSV, and XLSX.

**What is Streamlit's role?** `app/streamlit_app.py` provides upload, indexing, question, answer, source, and clear controls.

## Intermediate Questions

**What is an embedding?** A numeric representation used to compare semantic similarity. Gemini embeddings represent both chunks and queries.

**Why chunk documents?** Smaller overlapping units improve the chance that relevant text is retrieved; this project uses 1000-character chunks and 200-character overlap.

**What is ChromaDB doing?** It persists chunks, vectors, and metadata and performs similarity search.

**What is source attribution?** Carrying filename/location metadata into context and safe result dictionaries.

**What does LangChain provide?** `Document`, the recursive splitter, prompt template, Gemini integrations, and Chroma integration.

**What does LangGraph provide?** A fixed state graph with validate, retrieve, context, generate, and source nodes.

**What is top-k?** The number of retrieved chunks; graph input is normalized to 1 through 8, default 4.

## Advanced Questions

**Does the application use an agent loop?** No. The graph is linear and has no planning, tool-use, or autonomous retry nodes.

**How is grounding enforced?** The system prompt says to use only retrieved context, avoid outside knowledge, and use a fixed fallback when insufficient. This is an instruction, not a formal guarantee.

**Is there a relevance threshold?** No. Scores are returned but not thresholded, so nearest neighbors can be returned for unrelated questions.

**How are failures handled?** Safety helpers validate inputs, graph statuses classify failures, and provider/general exception details are replaced with generic messages.

**Is Chroma isolated per user?** No. One persistent shared collection is used and clear deletes all entries.

**How are CSV/XLSX sources represented?** Rows become documents; XLSX metadata includes sheet and row, while CSV metadata includes row.

**What is a notable dependency issue?** The OpenAI connection test imports `langchain_openai`, but that package is absent from `requirements.txt`; OpenAI is not the active application provider.

## Project-Specific Design Questions

**Why restore the original filename after temporary upload storage?** The UI writes a temporary path for loading, then replaces the source metadata so answers and sources identify the uploaded filename.

**What does the indexing summary contain?** Loaded document count, chunk count, vector count, and distinct source names.

**What does `answer_question` return?** The question, answer, source dictionaries, and retrieved documents.

**What happens for an empty question?** Validation returns a safe message and the graph uses `validation_failed`.

**What happens when context is empty?** The fallback answer is returned and generation is skipped in the graph.

**What is tested?** Loader, chunker, embedding, Chroma, retriever, indexer, RAG, graph, reliability, Streamlit import, real documents, and live provider checks. The suite was not run for this documentation task.

## Limitations and Future Improvements
Implemented behavior lacks deduplication, per-user collections, a relevance threshold, citation verification, temporary-file cleanup, and a complete UI workflow test. These are possible improvements, not current capabilities. Concepts such as Agentic RAG and Agentic AI are learning/reference terminology here rather than implemented graph behavior.
