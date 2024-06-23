#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR
echo Current Directory:
pwd

export MODEL_NAME=unsloth/Qwen2-0.5B-Instruct
echo Tuning $MODEL_NAME
python tune.py

export MODEL_NAME=unsloth/Qwen2-1.5B-Instruct
echo Tuning $MODEL_NAME
python tune.py