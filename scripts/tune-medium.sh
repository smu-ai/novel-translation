#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

# pip install -r requirements.txt
# FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --upgrade flash-attn

export MODEL_NAME=unsloth/Qwen2-7B-Instruct
echo Tuning $MODEL_NAME
python llm_translation/tune.py

export MODEL_NAME=unsloth/mistral-7b-instruct-v0.3
echo Tuning $MODEL_NAME
python llm_translation/tune.py

export MODEL_NAME=gradientai/Llama-3-8B-Instruct-Gradient-1048k
echo Tuning $MODEL_NAME
python llm_translation/tune.py
