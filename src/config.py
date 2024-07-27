import os
from langchain_openai import ChatOpenAI

# Securely setting up the OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def create_llm_openai():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=OPENAI_API_KEY
        # base_url="...",
        # organization="...",
        # other params...
    )
    return llm

