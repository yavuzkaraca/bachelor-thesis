"""
This module provides functions to process PDF documents using different language models and save the validation results to CSV files.
"""

import os

from llm import invoker, llm_creator
from utils import csv_writer
from utils.csv_writer import generate_filename
from utils.dataset_loader import paginate_pdf, get_pdf_file_paths


def do(func, llm, filename, output_base_dir, identifier):
    pdf_paths = get_pdf_file_paths()

    for pdf_path in pdf_paths:
        pages = paginate_pdf(pdf_path)

        result = func(llm, pages)
        csv_writer.save_results_to_csv(result, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, filename))


def process_all_pdfs_explore(llm, identifier):
    """
    Processes all PDF files in the dataset using all prompt variants.
    """
    output_base_dir = "../out/exploration"

    do(invoker.zero_shot, llm, "zero_shot", output_base_dir, identifier)
    do(invoker.few_shot, llm, "few_shot", output_base_dir, identifier),
    do(invoker.chain_of_thought_zero_shot, llm, "chain_of_thought_zero_shot", output_base_dir, identifier)
    do(invoker.chain_of_thought_few_shot, llm, "result_chain_of_thought_few_shot", output_base_dir, identifier)
    do(invoker.generated_knowledge, llm, "generated_knowledge", output_base_dir, identifier)
    do(invoker.engineer_persona, llm, "engineer_persona", output_base_dir, identifier)
    do(invoker.completeness_types, llm, "completeness_types", output_base_dir, identifier)
    do(invoker.repeated_instructions, llm, "repeated_instructions", output_base_dir, identifier)
    do(invoker.combined_generated_knowledge_completeness_types, llm,
       "combined_generated_knowledge_completeness_types", output_base_dir, identifier)
    do(invoker.combined_chain_of_thought_repeated_instructions, llm,
       "combined_chain_of_thought_repeated_instructions", output_base_dir, identifier)
    do(invoker.combined_all, llm, "combined_all", output_base_dir, identifier)


def process_all_pdfs_advanced(llm, identifier):
    """
    Processes all PDF files in the dataset using only one prompt: Combined GK + CT.
    """
    pdf_paths = get_pdf_file_paths()
    output_base_dir = "../out/"

    do(invoker.combined_generated_knowledge_completeness_types, llm,
       "combined_generated_knowledge_completeness_types", output_base_dir, identifier)


def openai_process_all(model="gpt-4o-2024-05-13"):
    """
    Processes all PDF files in the dataset using an OpenAI model.
    """
    llm = llm_creator.create_llm_openai(model)
    return process_all_pdfs_advanced(llm, "openai")


def ollama_process_all():
    """
    Processes all PDF files in the dataset using an Ollama model.
    """
    llm = llm_creator.create_llm_ollama()
    return process_all_pdfs_advanced(llm, "ollama")
