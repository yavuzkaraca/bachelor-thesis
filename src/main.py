import config
import templates
from pdf_loader import load_and_split_pdf, create_faiss_index, search_documents
import os  # Imported os


def basic_test(pdf_paths):
    llm = config.create_llm_openai()
    templates.prepare_llm(llm)

    results = {}
    for pdf_path in pdf_paths:
        absolute_pdf_path = os.path.abspath(pdf_path)  # Convert to absolute path to solve the error?
        pages = load_and_split_pdf(absolute_pdf_path)
        faiss_index = create_faiss_index(pages)
        docs = search_documents(faiss_index, "Identify incompleteness in this document.")
        analysis_result = templates.analyze_pdf_incompleteness(llm, docs)
        results[absolute_pdf_path] = analysis_result
    return results


if __name__ == '__main__':
    pdf_paths = ["dataset/2001_esa.pdf", "dataset/2005_nenios.pdf"]
    results = basic_test(pdf_paths)

    for pdf, result in results.items():
        print(f"Results for {pdf}:")
        print(result)
        print("\n" + "=" * 50 + "\n")
