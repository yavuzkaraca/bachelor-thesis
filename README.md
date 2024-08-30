# Validating Completeness in Software Requirements Specifications using Large Language Models 
Copyright (c) 2024 Yavuz Karaca


## Overview
This research investigates the use of Large Language Models (LLMs) to automate the validation of Software Requirements Specifications (SRS) for completeness. By leveraging LLMs, specifically GPT-4 and LLaMA3, this study aims to identify incomplete SRS elements. Experiments were conducted using various datasets and prompting techniques, with the F1 score as the primary performance metric.

This repository holds all the artifacts produced during the study except the paper itself. To read the paper: 
[Read the paper](https://journals.biologists.com/dev/article/139/2/335/45409/Balancing-of-ephrin-Eph-forward-and-reverse)


## Structure
- Dataset 
  - Original
  - Modified
  - Gold Standard
- Out: holds following experiments each with results, evaluation and plots
  - Exploration
  - Advanced
  - Delimiter Difference
- Source
  - LLM: includes creation and invoking of the LLMs as well as prompts
  - Result Processing: basic tools for calculating scores and plotting
  - Utilities: basic tools for loading PDFs and writing CSV files



## Acknowledgments
This study was supervised by M.Sc. Lars König


## Getting Started

### Prerequisites
Ensure you have Python 3.x installed on your system. You may also need to install additional packages:

```bash
pip install numpy matplotlib pandas langchain
```

### Installation
Clone this repository to your local machine using:
```bash
git clone https://gitlab.kit.edu/kit/kastel/sdq/stud/abschlussarbeiten/masterarbeiten/yavuzkaraca
```

### Configuring the Parameters
To current state is
- Invoking only "gpt-4o" model. To include llama3, uncomment the line 6 in main.py
- Saving results to "out". To change output directory, change output_base_dir parameter in pipeline.py
- Using only the default prompt. To change this, change the method called from invoker in pipeline.py
- Processing all files in dataset ending with "modified.pdf". To change this, change line 25 in dataset_loader.py

### Generating Reports
After configuring, simply run by:
```bash
python main.py
```
