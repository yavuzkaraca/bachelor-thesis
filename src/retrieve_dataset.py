import os


def get_pdf_file_paths():
    pdf_paths = []
    for root, dirs, files in os.walk("../dataset"):
        for file in files:
            if file.endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    return pdf_paths

