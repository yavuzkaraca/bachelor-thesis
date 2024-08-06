from utils import csv_writer
from llm import llm_creator, invoker
from utils.dataset_loader import paginate_pdf

test_path = "../dataset/PURE/2001_esa.pdf"


def test_memory_and_pagination(pdf_path):
    llm = llm_creator.create_llm_openai()

    test_message = [{"role": "user", "content": "Do you remember your task?"}]
    initial_msg = llm.invoke(test_message)
    print(initial_msg.content)

    # pdf_path = "../dataset/PURE/" + pdf_path
    pages = paginate_pdf(pdf_path)

    print("\nPAGE 6")
    print(pages[6])


def test_single_document(pdf_path):
    llm = llm_creator.create_llm_ollama()

    pages = paginate_pdf(pdf_path)

    result = invoker.analyze_pdf_incompleteness(llm, pages)
    print(result)
    csv_writer.save_results_to_csv(result, "2001_esa_test.cv")
    return result


if __name__ == '__main__':
    test_single_document(test_path)
