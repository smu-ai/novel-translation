#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export MODEL_NAME=unsloth/Qwen2-0.5B-Instruct-bnb-4bit
echo Tuning $MODEL_NAME
python llm_translation/tune.py

export MODEL_NAME=unsloth/Qwen2-1.5B-Instruct-bnb-4bit
echo Tuning $MODEL_NAME
python llm_translation/tune.py
