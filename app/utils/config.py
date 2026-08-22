import os

from dotenv import load_dotenv


load_dotenv()


# Google Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_LLM_MODEL = os.getenv(
    "GOOGLE_LLM_MODEL",
    "gemini-3.1-flash-lite",
)
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL",
    "gemini-embedding-2",
)


# OpenAI configuration kept as a backup provider
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv(
    "OPENAI_LLM_MODEL",
    "gpt-5.6-luna",
)
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not configured. "
        "Please add GOOGLE_API_KEY to the .env file."
    )