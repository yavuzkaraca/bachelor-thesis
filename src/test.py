from src.utils import csv_writer
from src.llm import llm_creator, invoker
from src.utils.dataset_loader import paginate_pdf

test_path_esa = "../dataset/PURE/2001_esa.pdf"
test_path_neutero = "../dataset/PSE/2022_neutero.pdf"


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
    llm = llm_creator.create_llm_openai()

    pages = paginate_pdf(pdf_path)

    result = invoker.validate_instructions_only(llm, pages)
    print(result)
    csv_writer.save_results_to_csv(result, "2022_esa_test_semicolon.csv")
    return result


if __name__ == '__main__':
    test_single_document(test_path_esa)
