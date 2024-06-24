.PHONY: start
start:
	python tune.py

.PHONY: format
format:
	black .

unsloth_env:
	conda create --name unsloth_env \
		python=3.10 \
		pytorch-cuda=12.1 \
		pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
			-y

install:
	pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
	pip install flash-attn --no-build-isolation
	pip install -r requirements.txt

build-flash-attn:
	FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --upgrade flash-attn
