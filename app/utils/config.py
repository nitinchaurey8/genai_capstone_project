"""
Module: Environment-based model configuration

Purpose:
Loads API keys and model names from environment variables using the dotenv package. It defines Google Gemini settings used by the application and OpenAI settings retained as backup provider configuration. The module requires a configured Google API key when it is imported.

Responsibilities:
- Load values from the project's environment configuration
- Define Google LLM and embedding model settings
- Define backup OpenAI API and model settings
- Fail early when the Google API key is unavailable

Project Role:
This module supplies shared configuration values to the embedding service and RAG chain. Those components use the Google settings to construct Gemini embedding and chat model clients.

Technologies:
- python-dotenv
- Python os environment access
- Google Gemini and OpenAI model configuration values

Important:
Google model names have defaults when their environment variables are absent, but importing this module raises a ValueError if `GOOGLE_API_KEY` is not configured. The OpenAI values are loaded but are not used by the code in this module.
"""

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