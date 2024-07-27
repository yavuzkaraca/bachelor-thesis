import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_and_split_pdf(pdf_path):
    # TODO: Fix pdf loading
    absolute_pdf_path = os.path.abspath("dataset/2001_esa.pdf")  # Maybe convert to absolute path??
    loader = PyPDFLoader(absolute_pdf_path)
    pages = loader.load_and_split()
    print(pages)
    print(pages[0])
    return pages


def create_faiss_index(pages):
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    return faiss_index


def search_documents(faiss_index, query, k=2):
    docs = faiss_index.similarity_search(query, k=k)
    return docs
