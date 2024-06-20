#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

# pip install -r requirements.txt
# FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --upgrade flash-attn

export MODEL_NAME=unsloth/Qwen2-7B-Instruct-bnb-4bit
echo Evaluating $MODEL_NAME
python eval.py

export MODEL_NAME=gradientai/Llama-3-8B-Instruct-Gradient-1048k
echo Evaluating $MODEL_NAME
python eval.py
