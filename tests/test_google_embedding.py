from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.utils.config import (
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
)


def main():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    text = "This is a test document for the GenAI Capstone RAG project."

    vector = embeddings.embed_query(text)

    print("EMBEDDING MODEL:", GOOGLE_EMBEDDING_MODEL)
    print("VECTOR TYPE:", type(vector))
    print("VECTOR LENGTH:", len(vector))
    print("FIRST 5 VALUES:", vector[:5])


if __name__ == "__main__":
    main()