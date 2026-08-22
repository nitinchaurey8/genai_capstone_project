"""
Module: Input validation and safe output helpers

Purpose:
Centralizes validation rules and user-facing messages for questions, uploaded file extensions, retrieval limits, and retrieved context. It also constrains retrieval settings, sanitizes internal error messages, and extracts a limited set of source metadata for frontend display.

Responsibilities:
- Validate questions, file names, uploaded file collections, and top-k values
- Normalize retrieval counts to configured minimum and maximum limits
- Detect whether retrieved Documents contain usable content
- Sanitize errors and build safe source metadata dictionaries

Project Role:
This module provides reliability and safety helpers used by the Streamlit interface, retriever, and LangGraph workflow. It defines shared validation limits and fallback messages used across the RAG application.

Technologies:
- Python pathlib
- LangChain-compatible Document metadata access
- Application retrieval and LangGraph workflow components

Important:
Supported uploads are PDF, TXT, CSV, and XLSX. Retrieval counts are constrained to 1 through 8 with a default of 4, and API or model-related failures are mapped to a generic LLM error message without exposing internal exception details.
"""

from pathlib import Path


# ------------------------------------------
# SUPPORTED FILE TYPES
# ------------------------------------------

SUPPORTED_FILE_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".xlsx",
}


# ------------------------------------------
# RETRIEVAL LIMITS
# ------------------------------------------

MIN_TOP_K = 1
MAX_TOP_K = 8
DEFAULT_TOP_K = 4


# ------------------------------------------
# SAFE USER MESSAGES
# ------------------------------------------

EMPTY_QUESTION_MESSAGE = (
    "Please enter a question before submitting."
)

NO_DOCUMENTS_MESSAGE = (
    "Please upload at least one supported document "
    "before asking a question."
)

NO_CONTEXT_MESSAGE = (
    "I don't have enough information in the uploaded "
    "documents to answer that question."
)

FALLBACK_ANSWER = NO_CONTEXT_MESSAGE

UNSUPPORTED_FILE_MESSAGE = (
    "This file type is not supported. Please upload "
    "a PDF, TXT, CSV, or XLSX file."
)

LLM_ERROR_MESSAGE = (
    "I’m unable to generate an answer right now. "
    "Please try again."
)

GENERAL_ERROR_MESSAGE = (
    "Something went wrong while processing your request. "
    "Please try again."
)


# ------------------------------------------
# QUESTION VALIDATION
# ------------------------------------------

def validate_question(question: str) -> tuple[bool, str]:
    """
    Validate a user question.

    Returns:
        (True, "") when valid.
        (False, error_message) when invalid.
    """

    if question is None:
        return False, EMPTY_QUESTION_MESSAGE

    if not isinstance(question, str):
        return False, EMPTY_QUESTION_MESSAGE

    if not question.strip():
        return False, EMPTY_QUESTION_MESSAGE

    return True, ""


# ------------------------------------------
# FILE VALIDATION
# ------------------------------------------

def validate_file_extension(
    file_name: str,
) -> tuple[bool, str]:
    """
    Validate whether a file extension is supported.
    """

    if not file_name:
        return False, UNSUPPORTED_FILE_MESSAGE

    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_FILE_EXTENSIONS:
        return False, UNSUPPORTED_FILE_MESSAGE

    return True, ""


def validate_uploaded_files(
    file_names: list[str],
) -> tuple[bool, str]:
    """
    Validate a collection of uploaded files.
    """

    if not file_names:
        return False, NO_DOCUMENTS_MESSAGE

    for file_name in file_names:
        valid, message = validate_file_extension(
            file_name
        )

        if not valid:
            return False, message

    return True, ""


# ------------------------------------------
# RETRIEVAL VALIDATION
# ------------------------------------------

def validate_top_k(
    k: int,
) -> tuple[bool, str]:
    """
    Validate the number of documents requested
    from the retriever.
    """

    if not isinstance(k, int):
        return False, (
            "Retrieval limit must be an integer."
        )

    if k < MIN_TOP_K:
        return False, (
            f"Retrieval limit must be at least "
            f"{MIN_TOP_K}."
        )

    if k > MAX_TOP_K:
        return False, (
            f"Retrieval limit cannot exceed "
            f"{MAX_TOP_K}."
        )

    return True, ""


def normalize_top_k(k: int) -> int:
    """
    Safely constrain the retrieval count to the
    configured range.
    """

    if not isinstance(k, int):
        return DEFAULT_TOP_K

    return max(
        MIN_TOP_K,
        min(k, MAX_TOP_K),
    )


# ------------------------------------------
# RETRIEVAL RESULT VALIDATION
# ------------------------------------------

def has_retrieved_context(
    documents: list,
) -> bool:
    """
    Check whether retrieval returned usable
    documents with actual content.
    """

    if not documents:
        return False

    for document in documents:
        if not document:
            continue

        content = getattr(
            document,
            "page_content",
            "",
        )

        if isinstance(content, str) and content.strip():
            return True

    return False


# ------------------------------------------
# ERROR SANITIZATION
# ------------------------------------------

def sanitize_error_message(
    error: Exception | str,
) -> str:
    """
    Convert internal errors into safe user-facing
    messages.

    Internal exception details are deliberately not
    exposed to the user because they may contain
    provider information, request details, paths,
    or other implementation details.
    """

    if not error:
        return GENERAL_ERROR_MESSAGE

    error_text = str(error).lower()

    # API / model related failures.
    api_indicators = [
        "api",
        "quota",
        "rate limit",
        "resource exhausted",
        "authentication",
        "permission",
        "google",
        "gemini",
        "model",
    ]

    if any(
        indicator in error_text
        for indicator in api_indicators
    ):
        return LLM_ERROR_MESSAGE

    return GENERAL_ERROR_MESSAGE


# ------------------------------------------
# SAFE SOURCE INFORMATION
# ------------------------------------------

def build_safe_source(
    document,
) -> dict:
    """
    Extract only safe source metadata for display
    in the frontend.
    """

    metadata = getattr(
        document,
        "metadata",
        {},
    )

    return {
        "source": metadata.get(
            "source",
            "Unknown source",
        ),
        "file_type": metadata.get(
            "file_type"
        ),
        "page": metadata.get(
            "page"
        ),
        "sheet": metadata.get(
            "sheet"
        ),
        "row": metadata.get(
            "row"
        ),
    }