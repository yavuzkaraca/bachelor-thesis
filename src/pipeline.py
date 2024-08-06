from llm import invoker, llm_creator
from utils import csv_writer
from utils.csv_writer import generate_filename, OPEN_AI_IDENTIFIER, LLAMA_IDENTIFIER
from utils.dataset_loader import paginate_pdf, get_pdf_file_paths


def gpt4o_process_all():
    llm = llm_creator.create_llm_openai()
    pdf_paths = get_pdf_file_paths()

    print(pdf_paths)

    results = {}
    for pdf_path in pdf_paths:
        pages = paginate_pdf(pdf_path)
        analysis_result = invoker.analyze_pdf_incompleteness(llm, pages)
        csv_writer.save_results_to_csv(analysis_result, generate_filename(pdf_path, OPEN_AI_IDENTIFIER))
        results[pdf_path] = analysis_result

    print_results(results)
    return results


def ollama_process_all():
    llm = llm_creator.create_llm_ollama()
    pdf_paths = get_pdf_file_paths()

    results = {}
    for pdf_path in pdf_paths:
        pages = paginate_pdf(pdf_path)
        analysis_result = invoker.analyze_pdf_incompleteness(llm, pages)
        csv_writer.save_results_to_csv(analysis_result, generate_filename(pdf_path, LLAMA_IDENTIFIER))
        results[pdf_path] = analysis_result

    print_results(results)
    return results


def print_results(results):
    for pdf, result in results.items():
        print(f"Results for {pdf}:")
        print(result)
        print("\n" + "=" * 50 + "\n")
