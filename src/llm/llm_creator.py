"""
This module configures and creates instances of language models from OpenAI and Ollama.

"""

import base64
import os

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Securely setting up the OpenAI Parameters
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ORGANIZATION_ID = os.environ.get("ORGANIZATION_ID")

# Securely setting up Ollama Parameters
username = os.environ.get("OLLAMA_USER")
password = os.environ.get("OLLAMA_PASSWORD")
headers = {"Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")}
BASE_URL = "https://ollama.vdl.sdq.kastel.kit.edu"


def create_llm_openai(model="gpt-4o-2024-05-13") -> ChatOpenAI:
    """
    Creates an instance of the ChatOpenAI model with predefined settings for GPT-4o.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY,
        organization=ORGANIZATION_ID
    )
    return llm


def create_llm_ollama(model="llama3:8b") -> ChatOllama:
    """
    Creates an instance of the ChatOllama model with predefined settings for the llama3:8b model.
    """
    llm = ChatOllama(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        base_url=BASE_URL,
        headers=headers
    )
    return llm
