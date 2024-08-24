"""
This module configures and creates instances of language models from OpenAI and Ollama.

Environment Variables:
    - OPENAI_API_KEY: The API key for accessing the OpenAI services, retrieved from environment variables.
    - OLLAMA_USER: The username for accessing Ollama, retrieved from environment variables.
    - OLLAMA_PASSWORD: The password for accessing Ollama, retrieved from environment variables.

Functions:
    - create_llm_openai: Creates and returns an instance of the ChatOpenAI model configured for GPT-4o.
    - create_llm_ollama: Creates and returns an instance of the ChatOllama model configured for the Mixtral model.
"""

import base64
import os

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Securely setting up the OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Securely setting up Ollama Header
username = os.environ.get("OLLAMA_USER")
password = os.environ.get("OLLAMA_PASSWORD")
headers = {"Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")}


def create_llm_openai():
    """
    Creates an instance of the ChatOpenAI model with predefined settings for GPT-4o.
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY
    )
    return llm


def create_llm_ollama():
    """
    Creates an instance of the ChatOllama model with predefined settings for the Mixtral model.

    The model is configured with:
        - Temperature: 0 (for deterministic outputs)
        - No token limit
        - No timeout
        - 2 max retries in case of failure
        - Custom base URL and authorization headers for Ollama API access

    Returns:
        An instance of ChatOllama ready for use.
    """
    llm = ChatOllama(
        model="llama3:8b",
        temperature=0,
        max_tokens=None,  # 4096?
        timeout=None,
        max_retries=2,
        base_url="https://ollama.vdl.sdq.kastel.kit.edu",
        headers=headers
    )
    return llm


def create_embedder():
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return embeddings_model
