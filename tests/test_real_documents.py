from pathlib import Path

from app.ingestion.document_loader import load_document


TEST_FOLDER = Path("data/loader_test")


def test_file(file_name: str, expected_type: str):
    file_path = TEST_FOLDER / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {file_path}"
        )

    documents = load_document(str(file_path))

    if not documents:
        raise AssertionError(
            f"No documents were loaded from {file_name}"
        )

    for document in documents:
        if document.metadata.get("file_type") != expected_type:
            raise AssertionError(
                f"Incorrect file type metadata for {file_name}"
            )

    print(
        f"{file_name}: PASSED "
        f"({len(documents)} document(s) loaded)"
    )

    print(
        f"  Source: {documents[0].metadata.get('source')}"
    )

    print(
        f"  Preview: "
        f"{documents[0].page_content[:150].replace(chr(10), ' ')}"
    )


def main():
    print("Starting real document loader test...\n")

    test_file(
        "test.txt",
        "txt",
    )

    test_file(
        "test.csv",
        "csv",
    )

    test_file(
        "test.xlsx",
        "xlsx",
    )

    print("\nREAL DOCUMENT LOADER TEST: PASSED")


if __name__ == "__main__":
    main()