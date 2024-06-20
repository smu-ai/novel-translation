import os
import re
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    TextStreamer,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer


def load_model(model_id, max_seq_length=2048, dtype=None, load_in_4bit=False):
    # load the quantized settings, we're doing 4 bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        # use the gpu
        device_map={"": 0},
    )

    # don't use the cache
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_model(model, tokenizer, prompt):
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)

    _ = model.generate(
        **inputs, max_new_tokens=128, streamer=text_streamer, use_cache=True
    )


def load_trainer(
    model,
    tokenizer,
    dataset,
    num_train_epochs,
    fp16=False,
    bf16=False,
    output_dir="./outputs",
):
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,  # uses the number of epochs earlier
        per_device_train_batch_size=2,  # 2 seems reasonable (made smaller due to CUDA memory issues)
        gradient_accumulation_steps=2,  # 2 is fine, as we're a small batch
        optim="paged_adamw_32bit",  # default optimizer
        save_steps=0,  # we're not gonna save
        logging_steps=10,  # same value as used by Meta
        learning_rate=2e-4,  # standard learning rate
        weight_decay=0.001,  # standard weight decay 0.001
        fp16=fp16,  # set to true for A100
        bf16=bf16,  # set to true for A100
        max_grad_norm=0.3,  # standard setting
        max_steps=-1,  # needs to be -1, otherwise overrides epochs
        warmup_ratio=0.03,  # standard warmup ratio
        group_by_length=True,  # speeds up the training
        lr_scheduler_type="cosine",  # constant seems better than cosine
        # report_to="tensorboard",
    )

    # Load LoRA configuration
    peft_config = LoraConfig(
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set supervised fine-tuning parameters
    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,  # use our lora peft config
        dataset_text_field="text",
        max_seq_length=None,  # no max sequence length
        tokenizer=tokenizer,  # use the llama tokenizer
        args=training_arguments,  # use the training arguments
        packing=False,  # don't need packing
    )


def load_translation_dataset(data_path, tokenizer=None):
    train_data_file = data_path.replace(".tsv", "-train.tsv")
    test_data_file = data_path.replace(".tsv", "-test.tsv")

    if not os.path.exists(train_data_file):
        print("generating train/test data files")
        dataset = load_dataset(
            "csv", data_files=data_path, delimiter="\t", split="train"
        )
        print(len(dataset))
        dataset = dataset.filter(lambda x: x["chinese"] and x["english"])

        datasets = dataset.train_test_split(test_size=0.2)
        print(len(dataset))

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(datasets["train"])
        test_df = pd.DataFrame(datasets["test"])

        # Save to TSV
        train_df.to_csv(train_data_file, sep="\t", index=False)
        test_df.to_csv(test_data_file, sep="\t", index=False)

    print("loading train/test data files")
    datasets = load_dataset(
        "csv",
        data_files={"train": train_data_file, "test": test_data_file},
        delimiter="\t",
    )

    if tokenizer:
        translation_prompt = "Please translate the following Chinese text into English and provide only the translated content, nothing else.\n{}"

        def formatting_prompts_func(examples):
            inputs = examples["chinese"]
            outputs = examples["english"]

            texts = []
            prompts = []
            for input, output in zip(inputs, outputs):
                prompt = translation_prompt.format(input)
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert in translating Chinese to English.",
                    },
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)
                texts.append(prompt + output + tokenizer.eos_token)
            return {"text": texts, "prompt": prompts}

        datasets = datasets.map(
            formatting_prompts_func,
            batched=True,
        )

    print(datasets)
    return datasets


def extract_answer(text, debug=False):
    if text:
        # Remove the begin and end tokens
        text = re.sub(r".*?assistant.+?\b", "", text, flags=re.DOTALL | re.MULTILINE)
        if debug:
            print("--------\nstep 1:", text)

        text = re.sub(r"<\|.*?\|>.*", "", text, flags=re.DOTALL | re.MULTILINE)
        if debug:
            print("--------\nstep 2:", text)
    # Return the result
    return text


def eval_model(model, tokenizer, eval_dataset):
    total = len(eval_dataset)
    predictions = []
    for i in tqdm(range(total)):
        inputs = tokenizer(
            eval_dataset["prompt"][i : i + 1],
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=4096, use_cache=False)
        decoded_output = tokenizer.batch_decode(outputs)
        decoded_output = [extract_answer(output) for output in decoded_output]
        predictions.extend(decoded_output)

    return predictions


def save_model(model, tokenizer, save_method, gguf=False, publish=True):
    model_name = os.getenv("MODEL_NAME")
    token = os.getenv("HF_TOKEN") or None
    hub_model = model_name.split("/")[-1] + "-MAC-"
    local_model = "models/" + hub_model

    if gguf:
        quantization_method = save_method
        model.save_pretrained_gguf(
            local_model + quantization_method,
            tokenizer,
            quantization_method=quantization_method,
        )

        if publish:
            model.push_to_hub_gguf(
                hub_model + "gguf-" + quantization_method,
                tokenizer,
                quantization_method=quantization_method,
                token=token,
            )
    else:
        model.save_pretrained_merged(
            local_model + save_method,
            tokenizer,
            save_method=save_method,
        )

        if publish:
            model.push_to_hub_merged(
                hub_model + save_method,
                tokenizer,
                save_method=save_method,
                token=token,
            )
