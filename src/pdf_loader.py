from langchain_community.document_loaders import PyPDFLoader


def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages
