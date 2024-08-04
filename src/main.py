import invoker
import llm_creator
import templates
from dataset_retriever import get_pdf_file_paths, paginate_pdf


def main():
    llm = llm_creator.create_llm_openai()
    pdf_paths = get_pdf_file_paths()

    results = {}
    for pdf_path in pdf_paths:
        pages = paginate_pdf(pdf_path)
        analysis_result = invoker.analyze_pdf_incompleteness(llm, pages)
        results[pdf_path] = analysis_result

    print_results(results)
    return results


def print_results(results):
    for pdf, result in results.items():
        print(f"Results for {pdf}:")
        print(result)
        print("\n" + "=" * 50 + "\n")


if __name__ == '__main__':
    main()

