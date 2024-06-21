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

#export MODEL_NAME=THUDM/chatglm-6b
export MODEL_NAME=THUDM/glm-4-9b-chat-1m
echo Tuning $MODEL_NAME
python tune.py

