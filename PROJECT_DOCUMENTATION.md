# GenAI Capstone Project

# Project Documentation

## 1. Project Overview

The GenAI Capstone Project is a Retrieval-Augmented Generation (RAG) application developed using Python and Streamlit.

The application allows users to upload documents in four supported formats:

\- PDF

\- TXT

\- CSV

\- XLSX

The uploaded documents are loaded, processed, divided into smaller chunks, converted into vector embeddings, stored in ChromaDB, and retrieved using semantic similarity when a user asks a question.

The retrieved document content is then provided as context to a Google Gemini language model to generate a grounded answer.

The project uses:

\- LangChain for document processing and LLM integration

\- LangGraph for RAG workflow orchestration

\- ChromaDB for vector storage and similarity search

\- Google Gemini for language generation

\- Gemini Embeddings for semantic vector representations

\- Streamlit for the application interface

---

## 2. Project Objectives

The main objectives of the project are:

1. Build a functional Generative AI application.

2. Implement a Retrieval-Augmented Generation pipeline.

3. Support multiple document formats.

## 4. Implement document ingestion and processing.

## 5. Implement document chunking.

## 6. Generate embeddings for documents and queries.

## 7. Store embeddings in a vector database.

## 8. Implement semantic document retrieval.

## 9. Generate answers using retrieved document context.

## 10. Use LangGraph to orchestrate the RAG workflow.

## 11. Provide source information with generated answers.

## 12. Implement reliability and safety controls.

## 13. Provide a user-friendly Streamlit interface.

## 14. Demonstrate concepts related to Generative AI, RAG, LangChain, LangGraph, AI Agents, Agentic AI, and Agentic RAG.

---

## 3. System Architecture

The application is organized into multiple functional layers.

\`\`\`text

                    USER

                      |

                      v

              +---------------+

              |   Streamlit   |

              |      UI       |

              +---------------+

                      |

             Document Upload

                      |

                      v

              +---------------+

              |   Document    |

              |    Loader     |

              +---------------+

                      |

                      v

              +---------------+

              |    Chunker    |

              +---------------+

                      |

                      v

              +---------------+

              |    Gemini     |

              |   Embeddings  |

              +---------------+

                      |

                      v

              +---------------+

              |   ChromaDB    |

              | Vector Store   |

              +---------------+



              USER QUESTION

                      |

                      v

              +---------------+

              |   Validation  |

              +---------------+

                      |

                      v

              +---------------+

              |   Retriever   |

              +---------------+

                      |

                      v

              +---------------+

              |    Context    |

              |    Builder    |

              +---------------+

                      |

                      v

              +---------------+

              |   LangGraph   |

              | RAG Workflow  |

              +---------------+

                      |

                      v

              +---------------+

              |  Gemini LLM   |

              +---------------+

                      |

                      v

              +---------------+

              | Answer +      |

              | Sources       |

              +---------------+





### 3.1 Presentation Layer

File:

app/streamlit_app.py

Responsibilities:

Provide the Streamlit user interface.

Allow users to upload documents.

Accept questions from users.

Trigger document processing.

Execute the RAG workflow.

Display generated answers.

Display source information.

### 3.2 Document Ingestion Layer

Files:

app/ingestion/document_loader.py

app/ingestion/indexer.py

Responsibilities:

Validate uploaded documents.

Identify supported file types.

Load PDF documents.

Load TXT documents.

Load CSV documents.

Load XLSX documents.

Preserve source metadata.

Prepare documents for processing and indexing.

Supported formats:

PDF

TXT

CSV

XLSX

### 3.3 Document Processing Layer

File:

app/processing/chunker.py

The application uses:

RecursiveCharacterTextSplitter

to divide documents into smaller chunks.

The tested configuration is:

Chunk Size: 1000

Chunk Overlap: 200

Chunking allows the retrieval system to work with smaller sections of documents and improves the ability to retrieve relevant content.

### 3.4 Embedding Layer

File:

app/embeddings/embedding_service.py

The application uses Google Gemini embeddings.

Current embedding model:

gemini-embedding-2

The embedding service is responsible for:

Creating the embedding model.

Generating document embeddings.

Generating query embeddings.

Handling empty document lists.

Rejecting empty queries.

The embedding service was tested successfully.

The tested embedding vector length is:

3072

### 3.5 Vector Database Layer

File:

app/vectorstore/chroma_store.py

The application uses:

ChromaDB

as the vector database.

ChromaDB is responsible for:

Storing document embeddings.

Storing document metadata.

Performing similarity searches.

Returning relevant documents.

Providing retrieval scores.

Generated ChromaDB runtime data is intentionally excluded from GitHub and the final source-code submission.

### 3.6 Retrieval Layer

File:

app/retrieval/retriever.py

The retriever receives a user question and searches ChromaDB for relevant document chunks.

The retrieval process is:

User Question

      |

      v

Query Embedding

      |

      v

Vector Similarity Search

      |

      v

Relevant Documents

      |

      v

Retrieved Context

The retriever supports:

Configurable top-k retrieval.

Retrieval scores.

Source metadata.

Context construction.

Empty-question validation.

Invalid top-k validation.

The default top-k configuration tested in the project is:

4

## 4. RAG Chain

File:

app/rag/rag_chain.py

The RAG chain combines:

User question

Retrieved document context

Google Gemini language model

The process is:

User Question

      |

      v

Retriever

      |

      v

Relevant Documents

      |

      v

Context Builder

      |

      v

Gemini LLM

      |

      v

Grounded Answer

The application is designed to generate answers using the retrieved document context.

The RAG chain also handles empty questions and empty retrieved context.

## 5. Language Model

The current language model is:

gemini-3.1-flash-lite

The application uses Google Gemini through:

langchain-google-genai

The Gemini connection was tested successfully.

The final application uses Google Gemini as the active language-model provider.

OpenAI configuration remains in the configuration module as backup-provider configuration from the development process, but the current RAG implementation uses Google Gemini.

## 6. LangGraph Workflow

File:

app/graph/rag_graph.py

LangGraph is used to organize the RAG process into a structured workflow.

The workflow can be represented as:

Question

   |

   v

Question Validation

   |

   v

Document Retrieval

   |

   v

Context Creation

   |

   v

Answer Generation

   |

   v

Source Generation

   |

   v

Final Response

The LangGraph implementation was tested successfully.

The workflow provides a structured foundation that can be extended in future versions toward more advanced agentic workflows.

### 7. Agent Roles and Workflow Roles

The current project should be described as a workflow-oriented RAG system, rather than a fully autonomous multi-agent system.

The following functional roles exist within the application.

### 7.1 Document Ingestion Role

Responsible for:

Accepting documents.

Validating supported formats.

Loading document content.

Creating application documents.

### 7.2 Document Processing Role

Responsible for:

Splitting documents into chunks.

Maintaining document metadata.

Preparing content for embedding.

### 7.3 Embedding Role

Responsible for:

Creating document embeddings.

Creating query embeddings.

Preparing vectors for semantic search.

### 7.4 Retrieval Role

Responsible for:

Searching ChromaDB.

Finding relevant document chunks.

Tracking retrieval scores.

Returning source metadata.

### 7.5 Context Construction Role

Responsible for:

Combining retrieved chunks.

Creating the context supplied to the language model.

Preserving source information.

### 7.6 Answer Generation Role

Responsible for:

Sending the question and retrieved context to Gemini.

Generating the final grounded answer.

### 7.7 Source Attribution Role

Responsible for:

Tracking document sources.

Returning source names and metadata.

Supporting transparent answer attribution.

### 7.8 Reliability and Safety Role

File:

app/reliability/safety.py

Responsible for:

Question validation.

File validation.

Retrieval parameter validation.

Error sanitization.

Source metadata sanitization.

Sensitive configuration protection.

No-context handling.

## 8. Reliability and Safety

The application implements validation and safety controls for:

Empty questions.

Whitespace-only questions.

Supported file validation.

Unsupported file validation.

Empty uploads.

Multiple supported files.

Mixed file validation.

Valid top-k values.

Invalid top-k values.

Maximum top-k limits.

Empty retrieved context.

API error sanitization.

General error sanitization.

Source metadata sanitization.

Sensitive configuration protection.

The reliability-aware LangGraph workflow was also tested successfully.

If useful retrieved context is not available, the application provides a controlled fallback instead of treating unsupported information as retrieved evidence.

## 9. Supported File Formats

9.1 PDF

PDF documents are processed using PyPDF.

9.2 TXT

Plain-text documents are loaded and converted into application documents.

9.3 CSV

CSV files are processed using Pandas and converted into retrievable document content.

9.4 XLSX

Excel files are processed using Pandas/OpenPyXL and converted into retrievable document content.

The project includes sample documents for all four supported formats.

## 10. Sample Test Documents

The project includes sample reference documents covering topics including:

Generative AI

AI Agents

Agentic AI

Agentic RAG

LangChain

LangGraph

The sample files are located under:

data/test_documents/

These files were used to test multi-format ingestion, indexing, retrieval, and question answering.

## 11. Complete RAG Flow

The complete application flow is:

                 DOCUMENT INGESTION

          PDF / TXT / CSV / XLSX

                    |

                    v

             Document Loader

                    |

                    v

              Document Chunks

                    |

                    v

             Gemini Embeddings

                    |

                    v

                 ChromaDB

                    |

                    |

                    |

                    v

                 QUESTION FLOW

               User Question

                    |

                    v

            Question Validation

                    |

                    v

              Query Embedding

                    |

                    v

                 Retriever

                    |

                    v

          Relevant Document Chunks

                    |

                    v

             Context Builder

                    |

                    v

                LangGraph

                    |

                    v

                Gemini LLM

                    |

                    v

              Grounded Answer

                    |

                    v

              Sources / Metadata

## 12. System Setup

12.1 Prerequisites

The project requires:

Python 3.12

Internet connection

Google Gemini API key

Git for source-code retrieval, if required

The Python dependencies are pinned in:

requirements.txt

12.2 Installation

From the project root:

python -m pip install -r requirements.txt

## 13. Environment Configuration

The application uses environment variables for configuration.

Create a local:

.env

file using:

.env.example

as the template.

Example:

GOOGLE_API_KEY=your_google_api_key_here

GOOGLE_LLM_MODEL=gemini-3.1-flash-lite

GOOGLE_EMBEDDING_MODEL=gemini-embedding-2

The real .env file is intentionally excluded from GitHub and the submission ZIP.

API keys must never be hard-coded into application source code.

## 14. Running the Application

From the project root:

streamlit run app\streamlit_app.py

The Streamlit application will normally be available at:

http\://localhost:8501

## 15. Application Usage

The basic user workflow is:

Step 1 — Start the application

Run:

streamlit run app\streamlit_app.py

Step 2 — Upload documents

Upload one or more supported documents:

PDF

TXT

CSV

XLSX

Step 3 — Process the documents

The application loads, chunks, embeds, and indexes the documents.

Step 4 — Ask a question

Enter a question related to the uploaded documents.

Step 5 — Retrieval

The application searches ChromaDB for relevant document chunks.

Step 6 — Answer generation

The retrieved context is passed to the Gemini language model.

Step 7 — Review the result

The application displays the generated answer and relevant source information.

## 16. Testing

The project contains component-level and workflow-level tests.

Examples:

python -m tests.test_google_connection

python -m tests.test_google_embedding

python -m tests.test_document_loader

python -m tests.test_real_documents

python -m tests.test_pdf_loader

python -m tests.test_chunker

python -m tests.test_embedding_service

python -m tests.test_chroma

python -m tests.test_chroma_store

python -m tests.test_retriever

python -m tests.test_rag_chain

python -m tests.test_rag_graph

python -m tests.test_reliability

python -m tests.test_rag_graph_reliability

python -m tests.test_indexer

python -m tests.test_streamlit_import

Tests are executed from the project root using Python module syntax.

Example:

python -m tests.test_rag_chain

## 17. Testing Results

The following components were successfully tested during development:

Component   Result

Google Gemini connection    PASSED

Gemini embeddings   PASSED

TXT loader  PASSED

CSV loader  PASSED

XLSX loader PASSED

PDF loader  PASSED

Unsupported file validation PASSED

Document chunking   PASSED

ChromaDB    PASSED

ChromaDB vector store   PASSED

Retriever   PASSED

RAG chain   PASSED

LangGraph RAG   PASSED

Reliability controls    PASSED

Reliability-aware LangGraph PASSED

Document indexer    PASSED

Streamlit import    PASSED

End-to-end RAG interaction  PASSED

The application was also tested using sample documents covering Generative AI, AI Agents, Agentic AI, Agentic RAG, LangChain, and LangGraph.

## 18. Deployment Steps

18.1 Local Deployment

Install Python 3.12.

Obtain the project source code.

Create and activate a Python virtual environment.

Install dependencies from requirements.txt.

Configure the Google Gemini API key.

Run the Streamlit application.

Example:

python -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

streamlit run app\streamlit_app.py

18.2 Cloud / Production Deployment

A production deployment should:

Use a Python 3.12-compatible environment.

Install dependencies from requirements.txt.

Configure API keys through the hosting platform's secret/environment-variable manager.

Never expose API keys in source code.

Provide persistent storage if indexed data needs to survive application restarts.

Configure the Streamlit application entry point.

Verify document upload.

Verify document indexing.

Verify retrieval.

Verify answer generation.

Monitor API usage and application errors.

The current capstone implementation is designed for Streamlit execution. Enterprise authentication, managed vector storage, production monitoring, and cloud-specific scaling infrastructure are outside the current scope.

## 19. GitHub Repository

The project source code is maintained at:

https\://github.com/nitinchaurey8/genai_capstone_project

The repository contains:

Application source code

Test code

Sample documents

Requirements

README documentation

Project documentation

Sensitive files and generated runtime data are excluded.

## 20. Limitations

The current implementation has the following limitations:

The application requires access to the Google Gemini API.

API usage is subject to provider quotas, availability, and rate limits.

Retrieval quality depends on document quality, chunking configuration, and embedding quality.

Poorly structured documents may produce less accurate retrieval results.

Very large document collections may require additional indexing and performance optimization.

Runtime ChromaDB data is not included in the source repository.

A new deployment needs to index documents before retrieving information from them.

The application does not currently implement enterprise authentication.

The application does not currently provide role-based access control.

Production-scale monitoring is not included.

Automated RAG evaluation metrics are not currently implemented.

Cloud-specific persistent storage and scaling configuration are outside the current capstone scope.

The current implementation is a workflow-oriented RAG system rather than a fully autonomous multi-agent architecture.

## 21. Challenges Faced During Development

21.1 OpenAI API Quota

Initial OpenAI API testing produced an API quota error with HTTP status 429.

The project was subsequently configured to use Google Gemini as the active language-model and embedding provider.

This allowed development and testing to continue.

21.2 Python Module Path

Running tests directly as scripts caused:

ModuleNotFoundError: No module named 'app'

The issue was resolved by running tests from the project root using Python module syntax.

Example:

python -m tests.test_rag_chain

21.3 XLSX File Locking on Windows

The XLSX document loader test initially encountered:

PermissionError: [WinError 32]

The issue occurred because the Excel file was still being held when the temporary test directory was cleaned up.

The Excel file handling was corrected so that file resources are released before cleanup.

The final document loader test passed successfully.

21.4 Gemini Response Format

During RAG testing, the Gemini response content was returned as a list rather than a plain string.

This initially caused:

AttributeError: 'list' object has no attribute 'strip'

The RAG chain was updated to safely handle the Gemini response format.

The complete RAG chain test subsequently passed.

21.5 Chunking Test

The initial chunking test document was not large enough to produce multiple chunks.

The test document was adjusted so that the configured chunk size and overlap could be properly verified.

The final test produced:

Original document length: 1199

Configured chunk size: 1000

Configured chunk overlap: 200

Number of chunks: 2

The document chunker test passed successfully.

21.6 ChromaDB Runtime Data

Generated ChromaDB binary files were initially included in the Git repository.

They were removed from Git tracking and the runtime directory was added to .gitignore.

This keeps generated runtime data separate from source code.

21.7 Dependency Management

The project initially contained OpenAI-related dependency configuration because OpenAI was tested during development.

After Google Gemini became the active provider, the final requirements.txt was cleaned up to reflect the dependencies required by the current application.

The final dependencies are pinned to tested versions.

## 22. Security Considerations

The application uses environment variables for API credentials.

The following are intentionally excluded from source control and final submission:

.env

venv/

.git/

data/chroma/

__pycache__/

.pytest_cache/

The repository contains:

.env.example

with placeholder configuration.

Real API keys must never be included in:

GitHub

README files

Source code

Documentation

Project ZIP files

## 23. Source Code Structure

genai_capstone_project/

│

├── app/

│   ├── embeddings/

│   │   ├── __init__.py

│   │   └── embedding_service.py

│   │

│   ├── graph/

│   │   ├── __init__.py

│   │   └── rag_graph.py

│   │

│   ├── ingestion/

│   │   ├── __init__.py

│   │   ├── document_loader.py

│   │   └── indexer.py

│   │

│   ├── processing/

│   │   ├── __init__.py

│   │   └── chunker.py

│   │

│   ├── rag/

│   │   ├── __init__.py

│   │   └── rag_chain.py

│   │

│   ├── reliability/

│   │   ├── __init__.py

│   │   └── safety.py

│   │

│   ├── retrieval/

│   │   ├── __init__.py

│   │   └── retriever.py

│   │

│   ├── utils/

│   │   └── config.py

│   │

│   ├── vectorstore/

│   │   ├── __init__.py

│   │   └── chroma_store.py

│   │

│   └── streamlit_app.py

│

├── data/

│   ├── loader_test/

│   └── test_documents/

│

├── tests/

│

├── .env.example

├── .gitignore

├── README.md

├── PROJECT_DOCUMENTATION.md

├── requirements.txt

└── commands.txt

## 24. Final Submission Contents

The final project ZIP should contain:

app/

tests/

data/loader_test/

data/test_documents/

.env.example

.gitignore

README.md

PROJECT_DOCUMENTATION.md

requirements.txt

commands.txt

The following should not be included:

.env

venv/

.git/

data/chroma/

__pycache__/

.pytest_cache/

*.pyc

The final submission therefore contains the complete application source code, test suite, sample test documents, dependency file, quick-start README, and formal project documentation.

## 25. Future Enhancements

Potential future enhancements include:

Conversation memory.

Query rewriting.

Advanced Agentic RAG.

Multi-agent orchestration.

Tool calling.

Relevance grading.

RAG evaluation metrics.

Automated evaluation datasets.

Additional document formats.

Metadata filtering.

Authentication.

Role-based access control.

Persistent managed vector storage.

Production monitoring.

Usage analytics.

Cloud deployment optimization.

## 26. Conclusion

The GenAI Capstone Project demonstrates an end-to-end Retrieval-Augmented Generation application using modern Generative AI technologies.

The system combines:

Multi-format Document Ingestion

          +

Document Chunking

          +

Gemini Embeddings

          +

ChromaDB Vector Search

          +

Semantic Retrieval

          +

LangChain

          +

LangGraph

          +

Gemini LLM

          +

Source Attribution

          +

Reliability and Safety Controls

          +

Streamlit

The completed application allows users to upload documents, process and index their content, ask questions, retrieve relevant information, and receive grounded answers with source information.

The project also demonstrates how a structured RAG workflow can be organized using LangGraph and provides a foundation for future extensions toward more advanced Agentic RAG and multi-agent systems.

The current implementation successfully demonstrates the core capstone requirements while documenting the system setup, architecture, workflow roles, deployment approach, testing, limitations, security considerations, and challenges encountered during development.