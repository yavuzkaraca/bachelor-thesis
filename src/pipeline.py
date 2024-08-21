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


def process_all_pdfs_all_prompt_variants(llm, identifier, output_base_dir="../out/exploration_second_phase"):
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

        """
        result_zero_shot = invoker.zero_shot(llm, pages)
        csv_writer.save_results_to_csv(result_zero_shot, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "zero_shot"))

        result_few_shot = invoker.few_shot(llm, pages)
        csv_writer.save_results_to_csv(result_few_shot, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "few_shot"))

        result_chain_of_thought_zero_shot = invoker.chain_of_thought_zero_shot(llm, pages)
        csv_writer.save_results_to_csv(result_chain_of_thought_zero_shot, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "chain_of_thought_zero_shot"))

        result_chain_of_thought_few_shot = invoker.chain_of_thought_few_shot(llm, pages)
        csv_writer.save_results_to_csv(result_chain_of_thought_few_shot, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "chain_of_thought_few_shot"))

        result_generated_knowledge = invoker.generated_knowledge(llm, pages)
        csv_writer.save_results_to_csv(result_generated_knowledge, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "generated_knowledge"))

        result_engineer_persona = invoker.engineer_persona(llm, pages)
        csv_writer.save_results_to_csv(result_engineer_persona, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "engineer_persona"))

        result_completeness_types = invoker.provided_completeness_types(llm, pages)
        csv_writer.save_results_to_csv(result_completeness_types, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "completeness_types"))

        result_repeated_instructions = invoker.repeated_instructions(llm, pages)
        csv_writer.save_results_to_csv(result_repeated_instructions, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "repeated_instructions"))
        """

        result_combined_gk_types = invoker.combined_gk_types(llm, pages)
        csv_writer.save_results_to_csv(result_combined_gk_types, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "combined_gk_types"))

        result_combined_cot_ri_few = invoker.combined_cot_ri(llm, pages)
        csv_writer.save_results_to_csv(result_combined_cot_ri_few, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "combined_cot_ri_few"))

        result_combined_all = invoker.combined_all(llm, pages)
        csv_writer.save_results_to_csv(result_combined_all, generate_filename(pdf_path, identifier),
                                       os.path.join(output_base_dir, "combined_all"))


def openai_process_all():
    """
    Processes all PDF files in the dataset using the GPT-4o model.

    Parameters:
    - output_base_dir: The base directory where the CSV files will be saved.

    Returns:
        A dictionary containing the results of the validations for each PDF.
    """
    llm = llm_creator.create_llm_openai()
    return process_all_pdfs_all_prompt_variants(llm, "gpt4o")


def ollama_process_all():
    """
    Processes all PDF files in the dataset using the Ollama model.

    Parameters:
    - output_base_dir: The base directory where the CSV files will be saved.

    Returns:
        A dictionary containing the results of the validations for each PDF.
    """
    llm = llm_creator.create_llm_ollama()
    return process_all_pdfs_all_prompt_variants(llm, "ollama")
