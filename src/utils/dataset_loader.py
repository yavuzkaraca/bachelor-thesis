import os
from langchain_community.document_loaders import PyPDFLoader


def get_pdf_file_paths():
    pdf_paths = []
    for root, dirs, files in os.walk("../dataset"):
        for file in files:
            if file.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths


def paginate_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages
