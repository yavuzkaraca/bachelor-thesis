"""
This module provides functions to process PDF documents using different language models and save the validation results to CSV files.

Functions:
    - process_all_pdfs_all_prompt_variants: Processes all PDFs using a specified language model with various prompts and saves the results to CSV files.
    - gpt4o_process_all: Processes all PDFs using the GPT-4o model with various prompts and saves the results to CSV files.
    - ollama_process_all: Processes all PDFs using the Ollama model with various prompts and saves the results to CSV files.
"""

import os

from llm import invoker, llm_creator
from utils import csv_writer
from utils.csv_writer import generate_filename
from utils.dataset_loader import paginate_pdf, get_pdf_file_paths


def process_all_pdfs_all_prompt_variants(llm, identifier, output_base_dir):
    """

    Processes all PDF files in the dataset using the specified language model.

    Parameters:
    - llm: The language model instance to use for processing.
    - identifier: The model identifier to include in the filenames.
    - output_base_dir: The base directory where the CSV files will be saved.

    The function:
        1. Loads the specified language model.
        2. Retrieves all PDF file paths from the dataset directory.
        3. For each PDF:
            - Splits the PDF into individual pages.
            - Runs three validation prompts (full prompt, instructions with IEEE guidelines, and instructions only).
            - Saves the results to CSV files in corresponding directories.

    Returns:
        A dictionary containing the results of the validations for each PDF, keyed by PDF path and prompt type.
    """
    pdf_paths = get_pdf_file_paths()

    print(pdf_paths)

    results = {}
    for pdf_path in pdf_paths:
        pages = paginate_pdf(pdf_path)

        result_full_prompt = invoker.generated_knowledge_all(llm, pages)
        csv_writer.save_results_to_csv(result_full_prompt, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "full_prompt"))
        results[pdf_path, 0] = result_full_prompt

        result_instructions_ieee = invoker.generated_knowledge_ieee(llm, pages)
        csv_writer.save_results_to_csv(result_instructions_ieee, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "instructions_ieee"))
        results[pdf_path, 1] = result_instructions_ieee

        result_instructions_only = invoker.few_shot(llm, pages)
        csv_writer.save_results_to_csv(result_instructions_only, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "instructions_only"))
        results[pdf_path, 2] = result_instructions_only

    return results


def openai_process_all(output_base_dir="../out"):
    """
    Processes all PDF files in the dataset using the GPT-4o model.

    Parameters:
    - output_base_dir: The base directory where the CSV files will be saved.

    Returns:
        A dictionary containing the results of the validations for each PDF.
    """
    llm = llm_creator.create_llm_openai()
    return process_all_pdfs_all_prompt_variants(llm, "gpt4o", output_base_dir)


def ollama_process_all(output_base_dir="../out"):
    """
    Processes all PDF files in the dataset using the Ollama model.

    Parameters:
    - output_base_dir: The base directory where the CSV files will be saved.

    Returns:
        A dictionary containing the results of the validations for each PDF.
    """
    llm = llm_creator.create_llm_ollama()
    return process_all_pdfs_all_prompt_variants(llm, "ollama", output_base_dir)
