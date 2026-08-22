"""
Module: Grounded Gemini RAG answer chain

Purpose:
Builds prompts from retrieved document context and uses the configured Gemini chat model to generate an answer to a question. It normalizes supported LangChain response content formats and returns a fallback when context or generated text is unavailable. It also exposes a complete question flow that returns the answer and retrieved source metadata.

Responsibilities:
- Configure the Gemini chat model and grounded RAG prompt
- Normalize string and content-block response formats
- Generate answers from supplied retrieved Documents only
- Retrieve context and return answers with source details

Project Role:
This module is the answer-generation layer of the GenAI RAG application. It combines retrieval utilities with the Google Gemini model and is also called by the LangGraph workflow.

Technologies:
- LangChain prompts and Documents
- langchain-google-genai ChatGoogleGenerativeAI
- Google Gemini

Important:
The system prompt instructs the model to use only retrieved context and provides a fallback answer when context or response content is insufficient. The configured model requires both a Google API key and LLM model name, and generation uses temperature 0.0.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.retrieval.retriever import (
    DEFAULT_TOP_K,
    build_context,
    retrieve_documents,
)
from app.utils.config import (
    GOOGLE_API_KEY,
    GOOGLE_LLM_MODEL,
)


# ------------------------------------------
# RAG CONFIGURATION
# ------------------------------------------

DEFAULT_TEMPERATURE = 0.0

FALLBACK_ANSWER = (
    "I don't have enough information in the "
    "uploaded documents to answer that question."
)


# ------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------

RAG_SYSTEM_PROMPT = """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the retrieved context.

Reliability rules:

1. Do not use outside knowledge to answer the question.
2. Do not invent or assume information.
3. If the retrieved context does not contain enough
   information to answer the question, clearly say:

   "I don't have enough information in the uploaded
   documents to answer that question."

4. Keep the answer concise and directly relevant
   to the user's question.
5. When possible, mention the source information
   available in the retrieved context.

Retrieved context:

{context}
"""


# ------------------------------------------
# PROMPT
# ------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            RAG_SYSTEM_PROMPT,
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


# ------------------------------------------
# LLM
# ------------------------------------------

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Create and return the configured Gemini LLM.
    """

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )

    if not GOOGLE_LLM_MODEL:
        raise ValueError(
            "GOOGLE_LLM_MODEL is not configured."
        )

    return ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=DEFAULT_TEMPERATURE,
    )


# ------------------------------------------
# RESPONSE CONTENT NORMALIZATION
# ------------------------------------------

def _extract_response_text(response) -> str:
    """
    Convert the Gemini/LangChain response content
    into a normal string.

    Current langchain-google-genai versions may
    return response.content as either:

    - a string
    - a list of content blocks

    This function handles both formats.
    """

    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content.strip()

    # List of content blocks
    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                block_text = block.get("text")

                if block_text:
                    text_parts.append(
                        str(block_text)
                    )

        return "\n".join(text_parts).strip()

    # Unexpected response type
    if content is None:
        return ""

    return str(content).strip()


# ------------------------------------------
# RAG RESPONSE
# ------------------------------------------

def generate_answer(
    question: str,
    documents: list[Document],
) -> str:
    """
    Generate an answer using only the supplied
    retrieved documents as context.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    if not documents:
        return FALLBACK_ANSWER

    context = build_context(documents)

    if not context.strip():
        return FALLBACK_ANSWER

    prompt = RAG_PROMPT.format_messages(
        context=context,
        question=question.strip(),
    )

    llm = get_llm()

    response = llm.invoke(prompt)

    answer = _extract_response_text(response)

    if not answer:
        return FALLBACK_ANSWER

    return answer


# ------------------------------------------
# COMPLETE RAG QUERY
# ------------------------------------------

def answer_question(
    question: str,
    k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Execute the complete RAG flow:

    Question
        ↓
    Retrieval
        ↓
    Context
        ↓
    LLM
        ↓
    Answer

    Returns the answer together with the
    retrieved source documents.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    documents = retrieve_documents(
        query=question,
        k=k,
    )

    answer = generate_answer(
        question=question,
        documents=documents,
    )

    sources = []

    for document in documents:
        sources.append(
            {
                "source": document.metadata.get(
                    "source",
                    "Unknown source",
                ),
                "file_type": document.metadata.get(
                    "file_type"
                ),
                "page": document.metadata.get(
                    "page"
                ),
                "sheet": document.metadata.get(
                    "sheet"
                ),
                "row": document.metadata.get(
                    "row"
                ),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_documents": documents,
    }