# GenAI Glossary

| Term | Meaning in this project |
|---|---|
| Generative AI | Model-generated content; Gemini generates answers. **Implemented.** |
| RAG | Retrieval followed by grounded generation. **Implemented.** |
| Agentic RAG | RAG with agent-like planning/tool choices. **Reference concept; not implemented.** |
| LangChain | Framework integrations and abstractions used for documents, prompts, models, embeddings, and Chroma. **Implemented.** |
| LangGraph | State graph orchestration library. **Implemented.** |
| AI Agent | A system that can choose actions/tools toward a goal. **Learning/reference; no agent node here.** |
| Agentic AI | AI systems with autonomous decision/action behavior. **Reference concept.** |
| Embedding | Numeric representation of text used for semantic comparison. **Implemented.** |
| Vector | Numeric values making up an embedding. **Implemented through provider integration.** |
| Vector Database | Store supporting vector similarity search and metadata. **Implemented with ChromaDB.** |
| ChromaDB | Persistent vector store used at `data/chroma`. **Implemented.** |
| Semantic Search | Retrieval by embedding similarity rather than exact keywords. **Implemented.** |
| Retriever | Code that obtains relevant chunks from Chroma. **Implemented in `retriever.py`.** |
| Chunk | Smaller text unit created from a document. **Implemented.** |
| Chunking | Splitting with size 1000 and overlap 200. **Implemented.** |
| Context | Labeled retrieved text supplied to Gemini. **Implemented.** |
| LLM | Large language model; Gemini chat model generates answers. **Implemented.** |
| Prompt | Structured instruction and question sent to the model. **Implemented with `ChatPromptTemplate`.** |
| Grounded Generation | Generation constrained to retrieved context. **Implemented as a prompt rule and fallback.** |
| Source Attribution | Returning source/location metadata for retrieved content. **Implemented.** |
| Document Ingestion | Loading supported files into LangChain documents. **Implemented.** |
| Indexing | Chunking, embedding, and storing documents. **Implemented.** |
| Similarity Search | Chroma nearest-neighbor search. **Implemented.** |
| Top-k | Requested result count; default 4, graph range 1..8. **Implemented.** |
| Metadata | Filename, type, page, sheet, row, and chunk ID attributes. **Implemented.** |
| State | `RAGState` values passed through graph nodes. **Implemented.** |
| Graph Node | One workflow function: validate, retrieve, context, generate, or sources. **Implemented.** |
| Streamlit | UI framework used by `streamlit_app.py`. **Implemented.** |
| Gemini | Google model/embedding provider used by the active path. **Implemented.** |
| Reliability | Validation, fallbacks, status handling, and safe errors. **Implemented.** |
| Safety | Avoiding secret/error/metadata overexposure and rejecting invalid inputs. **Implemented in defined scope.** |

## Important Distinctions
The project demonstrates RAG and graph orchestration but does not implement an autonomous agent, tool selection, multi-agent collaboration, relevance thresholds, or citation verification. Those terms may appear in learning material and sample documents, but they should not be described as active runtime features without additional code.
