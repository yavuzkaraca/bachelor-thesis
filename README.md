# Validating Completeness in Software Requirements Specifications using Large Language Models 
Copyright (c) 2024 Yavuz Karaca


## Overview
Traditional validation of SRS completeness is a manual, complex, and time-consuming
process that often results in overlooked requirements, potentially increasing costs. Large
Language Models (LLMs) offer a potential solution by automating the detection of incom-
pleteness in SRS documents. This work defines completeness types and evaluates state of
the art LLMs’ ability to correctly identify violations of them. The evaluation used various
prompt strategies, including generated knowledge, few-shot, and chain-of-thought, with
performance measured across multiple levels of completeness. Results showed GPT-4o
achieved an overall F1 score of 0.3, with precision consistently higher than recall across
all types. Validating completeness by understanding the document’s content had lowest
precision, while detecting gaps such as missing labels and references exhibited the lowest
recall. Overall, LLMs provide a useful supporting tool, but thorough validation still requires
human oversight.

## Structure
- **Dataset** 
  - Original
  - Modified
  - Gold Standard
- **Misc**: Holds few minor experiments relating to Ollama's capabilities
- **Out**: Holds the following experiments, each with results, evaluation, and plots
  - Exploration
  - Advanced
  - Delimiter Difference
- **Source**
  - LLM: Creation and invoking of the LLMs, including prompts
  - Result Processing: Tools for calculating scores and plotting
  - Utilities: Tools for loading PDFs and writing CSV files
- **Thesis**: Holds presentation and thesis along with the diagrams used in them


## Acknowledgments
This study was supervised by M.Sc. Lars König


## Getting Started

### Prerequisites
Ensure you have Python 3.12.4 installed on your system. You may also need to install additional packages:

```bash
pip install -r requirements.txt
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
- **Model Selection**: To change the model, adjust the model parameters in `llm_creator.py`. Also, only the "gpt-4o" model is currently invoked. To include LLaMA3, uncomment line 6 in `main.py`
- **Output Directory**: Results are saved to the "out" directory. To change this, update the `output_base_dir` parameter in `pipeline.py`
- **Prompt Selection**: Only the default prompt is used. To modify it, update the method called by the invoker in `pipeline.py`
- **Dataset Processing**: The script processes all files in the dataset ending with "modified.pdf." To change this, edit line 25 in `dataset_loader.py`

### Generating Reports
After configuring, run the following:
```bash
python main.py
```
