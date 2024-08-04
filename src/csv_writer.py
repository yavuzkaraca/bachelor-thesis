import csv
import os
from io import StringIO


def save_results_to_csv(result, filename, output_dir="../out"):
    """
    Save the analysis results to a CSV file.

    Parameters:
    - results: List of dictionaries containing analysis results with keys "Label", "Issue", and "Suggestion".
    - output_dir: Directory where the CSV file will be saved.
    - filename: Name of the PDF file.
    """
    filename += "_report.csv"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find the start of the CSV section
    csv_start = result.find("```csv\n") + len("```csv\n")
    csv_end = result.find("```\n", csv_start)

    # Extract the CSV portion
    csv_data = result[csv_start:csv_end].strip()

    # Parse the CSV data
    csv_lines = csv_data.splitlines()

    # Use StringIO to simulate a file object for the csv reader
    csv_reader = csv.reader(StringIO(csv_data))
    headers = next(csv_reader)

    # Define the full path to the output file
    output_file = os.path.join(output_dir, filename)

    # Write the extracted CSV data to the output file
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write the headers
        writer.writerow(headers)

        # Write the data rows
        for row in csv_reader:
            writer.writerow(row)

    print(f"Results have been saved to {output_file}")
