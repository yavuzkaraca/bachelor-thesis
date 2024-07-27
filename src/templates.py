from pdf_loader import load_and_split_pdf, create_faiss_index, search_documents
from config import create_llm_openai


def prepare_llm(llm):
    preparation_message = {
        "role": "system",
        "content": "You are an assistant that identifies incompleteness in Software Specification Documents."
    }
    return llm.invoke([preparation_message])


def analyze_pdf_incompleteness(llm, docs):
    messages = [
        {"role": "system",
         "content": "You are an assistant that identifies incompleteness in Software Specification Documents."},
        {"role": "user", "content": "Analyze the following sections for incompleteness:\n" + "\n".join(
            [doc.page_content for doc in docs])}
    ]

    response = llm.invoke(messages)
    return response.content
