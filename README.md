# Cross-Language Code Clone Detection System Documentation

This project is a complete system for cross-language code clone detection. By building an automated data preprocessing pipeline, the system can parse the source code of multiple programming languages into Abstract Syntax Trees (ASTs) and extract key control flow subtrees. Subsequently, it utilizes a deep learning model to learn structural features, ultimately achieving similarity discrimination of cross-language code.

## Operating Environment and Configuration

To ensure the model's efficiency when processing long-sequence intensive tensor calculations, it is recommended to conduct the training, validation, and testing of this project in a high-performance computing environment.

### Hardware Environment
* **Operating System**: Ubuntu
* **GPU**: 2 × NVIDIA A16

### Software and Dependencies
* **Programming Language**: Python 3.8.3
* **Deep Learning Framework**: PyTorch 1.11.0
* **Compute Acceleration Library**: CUDA 11.3 
* **Code Parsing Library**: `tree-sitter` 

## Directory Structure

The following is a complete directory tree description of the project's core data and source code:

```text
├── source_data/            # Stores the original training, validation, and test sets
├── datasets/               # Stores the structured datasets for each language pair generated after pipeline preprocessing
├── results/                # Stores the best model weight files saved during the model training process
├── prepare_data.py         # Global execution entry point for the data preprocessing stage
├── train.py                # Global execution entry point for model training, validation, and testing
├── preprocess/             # Data preprocessing module
│   ├── __init__.py         # Initialization file for the preprocessing module
│   ├── pipeline.py         # Core preprocessing pipeline
│   ├── python_cpp.py       # Processing logic for Python and C++ code pairs
│   ├── java_cpp.py         # Processing logic for Java and C++ code pairs
│   ├── java_python.py      # Processing logic for Java and Python code pairs
│   ├── python_cs.py        # Processing logic for Python and C# code pairs
│   ├── cpp_cs.py           # Processing logic for C++ and C# code pairs
│   └── java_cs.py          # Processing logic for Java and C# code pairs
└── model/                  # Deep learning model and training support module
    ├── __init__.py         # Initialization file for the model module
    ├── data.py             # Vocabulary construction, dataset encapsulation, and batch data assembly logic
    ├── metric.py           # Model evaluation metric calculation module
    └── model.py            # Neural network architecture definition module
```