"""
This module provides functions to retrieve PDF file paths from a specified directory and to paginate PDF files by splitting them into individual pages.

Functions:
    - get_pdf_file_paths: Recursively searches a directory for PDF files and returns their file paths.
    - paginate_pdf: Loads and splits a PDF file into individual pages using the PyPDFLoader.
"""

import os
from langchain_community.document_loaders import PyPDFLoader


def get_pdf_file_paths():
    """
    Recursively searches for PDF files in the '../dataset' directory and returns a list of their file paths.

    Returns:
        A list of strings, each representing the file path of a PDF file found in the directory.
    """
    pdf_paths = []
    for root, dirs, files in os.walk("../dataset/PSE"):
        for file in files:
            if file.endswith('modified.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def paginate_pdf(pdf_path):
    """
    Loads a PDF file and splits it into individual pages.

    Parameters:
        - pdf_path: The file path of the PDF to be paginated.

    Returns:
        A list of pages, where each page is an object representing a portion of the PDF content.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages
