# Automatic Machine Translation of Online Chinese Literature

This project includes the source datasets, code, and results for the work done by G1 Group 3 & G2 Group 5:

- Huang Donghao
- Teo Nicole
- Yan Guanru Linsay
- Yu Allen
- Zhao Xingyu Jasper

## Datasets

The datasets are located in the [datasets/mac](datasets/mac) folder:

- `mac.tsv`: Full dataset curated from [MAC](https://github.com/bfsujason/mac)
- `mac-train.tsv`: Training split
- `mac-test.tsv`: Testing split

## Source Code

The source code is located in the [llm_translation](llm_translation) folder and contains the following Python files:

- [tune.py](llm_translation/tune.py): Main script to evaluate and train models. It depends on:
  - [translation_engine_v3.py](llm_translation/translation_engine_v3.py)
  - [translation_utils.py](llm_translation/translation_utils.py)
- [translation_engine.py](llm_translation/translation_engine.py): Used by Jupyter [notebooks](notebooks) only

## Scripts

There are two shell scripts in the [scripts](scripts) folder to set up environment variables and invoke [tune.py](../llm_translation/tune.py):

- [tune-small.sh](scripts/tune-small.sh): Evaluates and tunes Qwen-0.5B and Qwen-1.5B (executed on a laptop)
- [tune-medium.sh](scripts/tune-medium.sh): Evaluates and tunes Qwen-7B, Mistral-7B, and Llama-3-8B (executed in SMU GPU cluster)

## Jupyter Notebooks

The [notebooks](notebooks) folder contains the following Jupyter notebook files:

1. [01_Qwen2-0.5B_Unsloth_train.ipynb](notebooks/01_Qwen2-0.5B_Unsloth_train.ipynb): Evaluate, train, and re-evaluate Qwen2-0.5B-Instruct
2. [02_Qwen2-1.5B_Unsloth_train.ipynb](notebooks/02_Qwen2-1.5B_Unsloth_train.ipynb): Evaluate, train, and re-evaluate Qwen2-1.5B-Instruct
3. [03_Qwen2-0.5B_1.5B-4bit.ipynb](notebooks/03_Qwen2-0.5B_1.5B-4bit.ipynb): Evaluate, train, and re-evaluate Qwen2-1.5B-Instruct (4-bit)
4. [04_Data_Analysis.ipynb](notebooks/04_Data_Analysis.ipynb): Data visualization

## Results

The results files are located in the [results](results) folder:

- [mac-results.csv](results/mac-results.csv): Raw results for experiment 1
- [mac-results_v3.csv](results/mac-results_v3.csv): Raw results for experiment 2
- [experiment-1-results.csv](results/experiment-1-results.csv): Experiment 1 visualization
- [experiment-2-results.csv](results/experiment-2-results.csv): Experiment 2 visualization

## Running Locally

### Pre-requisites

1. Ensure you have Python version 3.10 or above by running:

```
python –version
```

2. Verify that you have a CUDA-enabled GPU.
3. Use Linux or WSL2 for the best compatibility.

### Setup Instructions

1. Clone the repository:

```
git clone https://github.com/smu-ai/novel-translation
```

2. Set up the Unsloth environment by following the instructions at [Unsloth Setup](https://github.com/unslothai/unsloth).

3. Install the required packages:

```
pip install -r requirements.txt
```

4. Set up your environment variables:

- By default, environment variables are loaded from the `.env.example` file.
- To customize settings, copy `.env.example` to `.env` and update it for your local runs.

5. Run Jupyter notebooks or the automated script:

```
./scripts/tune-small.sh
```
