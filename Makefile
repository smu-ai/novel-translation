.PHONY: start
start:
	python tune.py

.PHONY: format
format:
	black .

setup:
	# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
	# pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
	# FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --upgrade flash-attn
	conda create --name unsloth_env \
		python=3.10 \
		pytorch-cuda=12.1 \
		pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
			-y
	
	# conda activate unsloth_env

install:
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
	pip install -r requirements.txt
