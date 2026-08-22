from langchain_google_genai import ChatGoogleGenerativeAI

from app.utils.config import GOOGLE_API_KEY, GOOGLE_LLM_MODEL


def main():
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    response = llm.invoke(
        "Reply with exactly: Google Gemini connection successful."
    )

    print("MODEL:", GOOGLE_LLM_MODEL)
    print("RESPONSE:", response.content)


if __name__ == "__main__":
    main()