import base64
import os

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

# Securely setting up the OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Securely setting up LLama Header
username = os.environ.get("OLLAMA_USER")
password = os.environ.get("OLLAMA_PASSWORD")
headers = {"Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode('utf-8')).decode("ascii")}


def create_llm_openai():
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
    llm = ChatOllama(
        model="mixtral",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        base_url="https://ollama.vdl.sdq.kastel.kit.edu",
        headers=headers
    )
    return llm
