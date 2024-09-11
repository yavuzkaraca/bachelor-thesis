# Validating Completeness in Software Requirements Specifications using Large Language Models 
Copyright (c) 2024 Yavuz Karaca


## Overview
This research explores the application of Large Language Models (LLMs) to automate the validation of Software Requirements Specifications (SRS) for completeness. By leveraging models such as GPT-4 and LLaMA3, the study aims to detect incomplete elements in SRS documents. The process involves sequentially feeding SRS documents to LLMs, along with specific prompts instructing them to identify instances of incompleteness. The LLMs then generate validation reports in CSV format. Experiments were conducted using various datasets and prompting techniques, with the F1 score serving as the primary performance metric.

This repository holds all the artifacts produced during the study except the paper itself. To read the paper: 
[Add the link for the paper](https://place-holder.com)


## Structure
- **Dataset** 
  - Original
  - Modified
  - Gold Standard
- **Out**: Holds the following experiments, each with results, evaluation, and plots
  - Exploration
  - Advanced
  - Delimiter Difference
- **Source**
  - LLM: Creation and invoking of the LLMs, including prompts
  - Result Processing: Tools for calculating scores and plotting
  - Utilities: Tools for loading PDFs and writing CSV files
- **Test**: Used as a playground during development


## Acknowledgments
This study was supervised by M.Sc. Lars König


## Getting Started

### Prerequisites
Ensure you have Python 3.x installed on your system. You may also need to install additional packages:

```bash
pip install numpy matplotlib pandas langchain
```

#### Setting Up API Keys
Before running the project, ensure you have set up the required API keys in your environment variables.

1. **OpenAI API Key**: Securely set up your OpenAI API key by adding it to your environment variables.

   - `OPENAI_API_KEY='your-openai-api-key'`

2. **Ollama API Headers**: Set up your Ollama credentials (username and password) in your environment variables.

   - `OLLAMA_USER='your-ollama-username'`
   - `OLLAMA_PASSWORD='your-ollama-password'`

You can add these lines to your shell configuration file (e.g., `.bashrc`, `.zshrc`) to automatically set them when you start a new session.

Make sure to replace `'your-openai-api-key'`, `'your-ollama-username'`, and `'your-ollama-password'` with your actual API keys and credentials.


### Installation
Clone this repository to your local machine using:
```bash
git clone https://gitlab.kit.edu/kit/kastel/sdq/stud/abschlussarbeiten/masterarbeiten/yavuzkaraca
```

### Configuring Parameters
- **Model Selection**: Only the "gpt-4o" model is currently invoked. To include LLaMA3, uncomment line 6 in `main.py`
- **Output Directory**: Results are saved to the "out" directory. To change this, update the `output_base_dir` parameter in `pipeline.py`
- **Prompt Selection**: Only the default prompt is used. To modify it, update the method called by the invoker in `pipeline.py`
- **Dataset Processing**: The script processes all files in the dataset ending with "modified.pdf." To change this, edit line 25 in `dataset_loader.py`

### Generating Reports
After configuring, run the following:
```bash
python main.py
```
