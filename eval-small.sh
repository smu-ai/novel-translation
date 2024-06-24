#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

export RESULTS_PATH=results/train-results.csv
export MODEL_NAME=unsloth/Qwen2-0.5B-Instruct
echo Tuning $MODEL_NAME
python tune.py

export RESULTS_PATH=results/eval-results.csv
export MODEL_NAME=inflaton/Qwen2-0.5B-Instruct-MAC-lora
echo Evaluating $MODEL_NAME
python eval.py

# export MODEL_NAME=inflaton/Qwen2-7B-Instruct-MAC-lora
# echo Evaluating $MODEL_NAME
# python eval.py

# export MODEL_NAME=inflaton/mistral-7b-instruct-v0.3-MAC-lora
# echo Evaluating $MODEL_NAME
# python eval.py
