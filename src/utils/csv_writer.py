"""
This module provides functions for processing analysis results and saving them to CSV files. It includes functionality for extracting CSV data from text and generating appropriate filenames for output files.

Constants:
    - OPEN_AI_IDENTIFIER: Identifier for the OpenAI model used.
    - LLAMA_IDENTIFIER: Identifier for the LLama model used.

Functions:
    - save_results_to_csv: Extracts CSV data from a result string and saves it to a CSV file in the specified output directory.
    - generate_filename: Generates a filename for the output CSV file based on the input PDF path and the model identifier.
"""

import csv
import os
from io import StringIO


def save_results_to_csv(result, filename, output_dir="../out") -> None:
    """
    Saves the analysis results to a CSV file.

    Parameters:
    - result: A string containing the analysis results, which includes a CSV section.
    - filename: The name of the file where the results will be saved.
    - output_dir: Directory where the CSV file will be saved (default is "../out").

    The function:
    1. Ensures the output directory exists.
    2. Extracts the CSV section from the result string.
    3. Parses the CSV data.
    4. Writes the CSV data to the specified file.

    Outputs:
    - A CSV file saved in the specified directory containing the analysis results.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find the start of the CSV section
    csv_start = result.find("```csv\n") + len("```csv\n")
    csv_end = result.find("```\n", csv_start)

    # Extract the CSV portion
    csv_data = result[csv_start:csv_end].strip()

    # Use StringIO to simulate a file object for the csv reader
    csv_reader = csv.reader(StringIO(csv_data), delimiter=',')

    # Define the full path to the output file
    output_file = os.path.join(output_dir, filename)

    # Write the extracted CSV data to the output file
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # Write the CSV data directly to the file
        for row in csv_reader:
            writer.writerow(row)

    print(f"Results have been saved to {output_file}")


def generate_filename(pdf_path, identifier) -> str:
    """
    Generates a filename for the output CSV file based on the PDF path and model identifier.
    """
    basename = os.path.basename(pdf_path)

    name_without_extension, _ = os.path.splitext(basename)  # removes file extension (.pdf or .html)

    if name_without_extension.endswith('_modified'):
        name_without_extension = name_without_extension[:-9]  # removes the last 9 characters ('_modified')

    filename = name_without_extension + "_report_" + identifier + ".csv"
    return filename
