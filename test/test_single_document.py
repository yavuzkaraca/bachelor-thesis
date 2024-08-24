from src.utils import csv_writer
from src.llm import llm_creator, invoker
from src.utils.dataset_loader import paginate_pdf

test_path_esa = "../dataset/PURE/2001_esa/2001_esa_modified.pdf"
test_path_neutero = "../dataset/PSE/2022_neutero/2022_neutero_modified.pdf"
test_path_home = "../dataset/PURE/2010_home_1.3/2010_home_1.3.pdf"
test_path_init = "../dataset/PSE/2021_init-v/2021_init-v_modified.pdf"
test_path_octo = "../dataset/PSE/2023_octo/2023_octo_modified.pdf"

def test_single_document(pdf_path):
    """
    This is just for testing a single document to not consume unnecessary API Calls
    """
    llm = llm_creator.create_llm_ollama()

    pages = paginate_pdf(pdf_path)

    result = invoker.few_shot(llm, pages)
    print(result)

    csv_writer.save_results_to_csv(result, "2010_home_llama3_csv", "test_out/")
    return result


if __name__ == '__main__':
    test_single_document(test_path_home)
