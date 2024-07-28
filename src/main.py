import config
import templates
from retrieve_dataset import get_pdf_file_paths
from pdf_loader import paginate_pdf


def main():
    llm = config.create_llm_openai()
    templates.prepare_llm(llm)

    pdf_paths = get_pdf_file_paths()

    results = {}
    for pdf_path in pdf_paths:
        pages = paginate_pdf(pdf_path)
        analysis_result = templates.analyze_pdf_incompleteness(llm, pages)
        results[pdf_path] = analysis_result

    print_results(results)
    return results


def test_single_document(pdf_path):
    llm = config.create_llm_openai()
    templates.prepare_llm(llm)

    test_message = [{"role": "user", "content": "Do you remember your task?"}]
    initial_msg = llm.invoke(test_message)
    print(initial_msg.content)

    pdf_path = "../dataset/" + pdf_path
    pages = paginate_pdf(pdf_path)

    print("\nPAGE 6")
    print(pages[6])

    result = templates.analyze_pdf_incompleteness(llm, pages)
    print(result)
    return result


def print_results(results):
    for pdf, result in results.items():
        print(f"Results for {pdf}:")
        print(result)
        print("\n" + "=" * 50 + "\n")


if __name__ == '__main__':
    # main()
    test_single_document("2001_esa.pdf")

