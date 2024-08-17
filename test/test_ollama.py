from src.utils import csv_writer
from src.llm import llm_creator
from src.utils.dataset_loader import paginate_pdf

test_path_esa = "../dataset/PURE/2001_esa.pdf"
test_path_neutero = "../dataset/PSE/2022_neutero.pdf"
test_path_home = "../dataset/PURE/2010_home_1.3.pdf"


def test_ollama_csv():
    """
    Is ollama capable of producing a CSV file at all? => YES!
    """
    llm = llm_creator.create_llm_ollama()

    message = [{"role": "user", "content": "Produce CSV file from top 10 things to do in paris. Put ```csv before "
                                           "starting and end it ```. Also use semicolon as delimiter."
                                           "So, Output format is like this: \n"
                                           "```csv\n"
                                           "<<content>>> \n"
                                           "```"}]
    answer = llm.invoke(message)

    csv_writer.save_results_to_csv(answer.content, "ollama_paris_tour_semicolon.csv")

    print(answer.content)


def test_ollama_document_read(pdf_path):
    """
    Is ollama capable of reading the entire document? => I guess not?
    """
    llm = llm_creator.create_llm_ollama()
    pages = paginate_pdf(pdf_path)

    print("\n".join([page.page_content for page in pages]))

    message = [{"role": "user", "content": "Summarize each page. Output format should be like: \n"
                                           "Page number X:\n"
                                           "<<page summary>>.\n"
                                           "here is the document:"
                                           + "\n".join([page.page_content for page in pages])}]
    answer = llm.invoke(message)

    csv_writer.save_results_to_csv(answer.content, "ollama_home_fq_summary.csv")

    print(answer.content)


if __name__ == '__main__':
    test_ollama_document_read(test_path_home)
