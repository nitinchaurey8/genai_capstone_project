from langchain_core.documents import Document

from app.reliability.safety import (
    DEFAULT_TOP_K,
    EMPTY_QUESTION_MESSAGE,
    GENERAL_ERROR_MESSAGE,
    LLM_ERROR_MESSAGE,
    MAX_TOP_K,
    MIN_TOP_K,
    NO_CONTEXT_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    SUPPORTED_FILE_EXTENSIONS,
    UNSUPPORTED_FILE_MESSAGE,
    build_safe_source,
    has_retrieved_context,
    normalize_top_k,
    sanitize_error_message,
    validate_file_extension,
    validate_question,
    validate_top_k,
    validate_uploaded_files,
)


def main():
    print("Starting reliability and safety test...\n")

    # ------------------------------------------
    # TEST 1: VALID QUESTION
    # ------------------------------------------

    valid, message = validate_question(
        "What is ChromaDB?"
    )

    assert valid is True
    assert message == ""

    print("Valid question: PASSED")

    # ------------------------------------------
    # TEST 2: EMPTY QUESTION
    # ------------------------------------------

    valid, message = validate_question("")

    assert valid is False
    assert message == EMPTY_QUESTION_MESSAGE

    print("Empty question validation: PASSED")

    # ------------------------------------------
    # TEST 3: WHITESPACE QUESTION
    # ------------------------------------------

    valid, message = validate_question("   ")

    assert valid is False
    assert message == EMPTY_QUESTION_MESSAGE

    print("Whitespace question validation: PASSED")

    # ------------------------------------------
    # TEST 4: SUPPORTED FILES
    # ------------------------------------------

    for extension in SUPPORTED_FILE_EXTENSIONS:
        valid, message = validate_file_extension(
            f"test{extension}"
        )

        assert valid is True
        assert message == ""

    print("Supported file validation: PASSED")

    # ------------------------------------------
    # TEST 5: UNSUPPORTED FILE
    # ------------------------------------------

    valid, message = validate_file_extension(
        "test.docx"
    )

    assert valid is False
    assert message == UNSUPPORTED_FILE_MESSAGE

    print("Unsupported file validation: PASSED")

    # ------------------------------------------
    # TEST 6: EMPTY UPLOAD
    # ------------------------------------------

    valid, message = validate_uploaded_files([])

    assert valid is False
    assert message == NO_DOCUMENTS_MESSAGE

    print("Empty upload validation: PASSED")

    # ------------------------------------------
    # TEST 7: MULTIPLE VALID FILES
    # ------------------------------------------

    valid, message = validate_uploaded_files(
        [
            "document.pdf",
            "notes.txt",
            "data.csv",
            "customers.xlsx",
        ]
    )

    assert valid is True
    assert message == ""

    print(
        "Multiple supported files validation: PASSED"
    )

    # ------------------------------------------
    # TEST 8: MIXED VALID/INVALID FILES
    # ------------------------------------------

    valid, message = validate_uploaded_files(
        [
            "document.pdf",
            "malicious.exe",
        ]
    )

    assert valid is False
    assert message == UNSUPPORTED_FILE_MESSAGE

    print(
        "Mixed file validation: PASSED"
    )

    # ------------------------------------------
    # TEST 9: VALID TOP-K
    # ------------------------------------------

    valid, message = validate_top_k(4)

    assert valid is True
    assert message == ""

    print("Valid top-k validation: PASSED")

    # ------------------------------------------
    # TEST 10: INVALID TOP-K
    # ------------------------------------------

    valid, message = validate_top_k(0)

    assert valid is False

    valid, message = validate_top_k(
        MAX_TOP_K + 1
    )

    assert valid is False

    print("Invalid top-k validation: PASSED")

    # ------------------------------------------
    # TEST 11: TOP-K NORMALIZATION
    # ------------------------------------------

    assert normalize_top_k(4) == 4

    assert normalize_top_k(0) == MIN_TOP_K

    assert (
        normalize_top_k(MAX_TOP_K + 10)
        == MAX_TOP_K
    )

    assert (
        normalize_top_k("invalid")
        == DEFAULT_TOP_K
    )

    print("Top-k normalization: PASSED")

    # ------------------------------------------
    # TEST 12: VALID RETRIEVED CONTEXT
    # ------------------------------------------

    documents = [
        Document(
            page_content=(
                "ChromaDB stores embeddings and "
                "supports similarity search."
            ),
            metadata={
                "source": "chromadb.txt",
                "file_type": "txt",
            },
        )
    ]

    assert has_retrieved_context(
        documents
    ) is True

    print(
        "Retrieved context validation: PASSED"
    )

    # ------------------------------------------
    # TEST 13: EMPTY RETRIEVED CONTEXT
    # ------------------------------------------

    assert has_retrieved_context([]) is False

    empty_document = Document(
        page_content="",
        metadata={
            "source": "empty.txt",
        },
    )

    assert has_retrieved_context(
        [empty_document]
    ) is False

    print(
        "Empty retrieved context validation: PASSED"
    )

    # ------------------------------------------
    # TEST 14: ERROR SANITIZATION
    # ------------------------------------------

    safe_message = sanitize_error_message(
        Exception("Google API quota exceeded")
    )

    assert safe_message == LLM_ERROR_MESSAGE

    print(
        "API error sanitization: PASSED"
    )

    # ------------------------------------------
    # TEST 15: GENERAL ERROR
    # ------------------------------------------

    safe_message = sanitize_error_message(
        Exception("Internal database failure")
    )

    assert safe_message == GENERAL_ERROR_MESSAGE

    print(
        "General error sanitization: PASSED"
    )

    # ------------------------------------------
    # TEST 16: SOURCE SANITIZATION
    # ------------------------------------------

    source_document = Document(
        page_content="Example content.",
        metadata={
            "source": "example.pdf",
            "file_type": "pdf",
            "page": 3,
            "internal_secret": "DO NOT EXPOSE",
        },
    )

    safe_source = build_safe_source(
        source_document
    )

    assert safe_source["source"] == "example.pdf"
    assert safe_source["file_type"] == "pdf"
    assert safe_source["page"] == 3

    assert (
        "internal_secret"
        not in safe_source
    )

    print(
        "Source metadata sanitization: PASSED"
    )

    # ------------------------------------------
    # FINAL RESULT
    # ------------------------------------------

    print(
        "\nRELIABILITY AND SAFETY TEST: PASSED"
    )


if __name__ == "__main__":
    main()