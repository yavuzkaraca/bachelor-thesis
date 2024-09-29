"""
This module configures and creates instances of language models from OpenAI and Ollama.

"""

import base64
import os

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ORGANIZATION_ID = "org-JOZZVGILnxuXT7NdjxlKF7g8"
BASE_URL = "https://ollama.vdl.sdq.kastel.kit.edu"

# Securely setting up the OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Securely setting up Ollama Header
username = os.environ.get("OLLAMA_USER")
password = os.environ.get("OLLAMA_PASSWORD")
headers = {"Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")}


def create_llm_openai() -> ChatOpenAI:
    """
    Creates an instance of the ChatOpenAI model with predefined settings for GPT-4o.
    """
    llm = ChatOpenAI(
        model="gpt-4o-2024-05-13",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY,
        organization=ORGANIZATION_ID
    )
    return llm


def create_llm_ollama() -> ChatOllama:
    """
    Creates an instance of the ChatOllama model with predefined settings for the llama3:8b model.
    """
    llm = ChatOllama(
        model="llama3:8b",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        base_url=BASE_URL,
        headers=headers
    )
    return llm
