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

export MODEL_NAME=unsloth/llama-3-70b-Instruct-bnb-4bit
echo Tuning $MODEL_NAME
python tune.py

