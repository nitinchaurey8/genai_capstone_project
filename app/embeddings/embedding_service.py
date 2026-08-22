from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.utils.config import (
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
)


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Create and return the Google Gemini embedding model.

    The API key and embedding model name are loaded
    from the application configuration.
    """

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )

    if not GOOGLE_EMBEDDING_MODEL:
        raise ValueError(
            "GOOGLE_EMBEDDING_MODEL is not configured."
        )

    return GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def embed_query(text: str) -> list[float]:
    """
    Generate an embedding vector for a single query.
    """

    if not text or not text.strip():
        raise ValueError(
            "Cannot generate an embedding for empty text."
        )

    embedding_model = get_embedding_model()

    return embedding_model.embed_query(text)


def embed_documents(
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embedding vectors for multiple documents.
    """

    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return []

    embedding_model = get_embedding_model()

    return embedding_model.embed_documents(
        cleaned_texts
    )