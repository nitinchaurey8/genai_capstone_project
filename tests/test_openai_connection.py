from langchain_openai import ChatOpenAI

from app.utils.config import OPENAI_API_KEY, OPENAI_LLM_MODEL


def main():
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_LLM_MODEL,
        temperature=0,
    )

    response = llm.invoke(
        "Reply with exactly: OpenAI connection successful."
    )

    print("MODEL:", OPENAI_LLM_MODEL)
    print("RESPONSE:", response.content)


if __name__ == "__main__":
    main()