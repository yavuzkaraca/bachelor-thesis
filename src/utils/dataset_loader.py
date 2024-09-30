"""
This module provides functions to retrieve PDF file paths from a specified directory and to paginate PDF files by
splitting them into individual pages.
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def get_pdf_file_paths() -> list[str]:
    """
    Recursively searches for PDF files in the '../dataset' directory and returns a list of their file paths.
    """
    pdf_paths = []
    for root, dirs, files in os.walk("../dataset/"):
        # check if dirs can be ignored using "_"
        for file in files:
            if file.endswith('modified.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def paginate_pdf(pdf_path) -> list[Document]:
    """
    Loads a PDF file and splits it into individual pages.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages
