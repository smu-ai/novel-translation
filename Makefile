.PHONY: start
start:
	python tune.py

.PHONY: format
format:
	black .

setup:
	# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
	pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
	FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --upgrade flash-attn
	

install:
	pip install -r requirements.txt

