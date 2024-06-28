import os
import pandas as pd
from datasets import load_dataset
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from llm_translation.transformers import TrainingArguments, TextStreamer
from tqdm import tqdm
from llm_translation.translation_utils import *

print(f"loading {__file__}")


def get_model_names(
    model_name, save_method="merged_4bit_forced", quantization_method="q5_k_m"
):
    hub_model = model_name.split("/")[-1] + "-MAC-"
    local_model = "models/" + hub_model

    return {
        "local": local_model + save_method,
        "local-gguf": local_model + quantization_method,
        "hub": hub_model + save_method,
        "hub-gguf": hub_model + "gguf-" + quantization_method,
    }


def load_model(
    model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
):
    print(f"loading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
    )
    FastLanguageModel.for_inference(model)

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
    max_seq_length=2048,
    fp16=False,
    bf16=False,
    output_dir="./outputs",
):
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=100,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
        ),
    )

    return trainer


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

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in translating Chinese to English.",
                },
                None,
            ]

            model_name = os.getenv("MODEL_NAME")

            if "mistral" in model_name.lower():
                messages = messages[1:]

            texts = []
            prompts = []
            for input, output in zip(inputs, outputs):
                prompt = translation_prompt.format(input)
                messages[-1] = {"role": "user", "content": prompt}

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
        debug = i == 0
        decoded_output = [
            extract_answer(output, debug=debug) for output in decoded_output
        ]
        predictions.extend(decoded_output)

    return predictions


def save_model(
    model,
    tokenizer,
    include_gguf=True,
    include_merged=True,
    publish=True,
):
    try:
        token = os.getenv("HF_TOKEN") or None
        model_name = os.getenv("MODEL_NAME")

        save_method = "lora"
        quantization_method = "q5_k_m"

        model_names = get_model_names(
            model_name, save_method=save_method, quantization_method=quantization_method
        )

        model.save_pretrained(model_names["local"])
        tokenizer.save_pretrained(model_names["local"])

        if publish:
            model.push_to_hub(
                model_names["hub"],
                token=token,
            )
            tokenizer.push_to_hub(
                model_names["hub"],
                token=token,
            )

        if include_merged:
            model.save_pretrained_merged(
                model_names["local"] + "-merged", tokenizer, save_method=save_method
            )
            if publish:
                model.push_to_hub_merged(
                    model_names["hub"] + "-merged",
                    tokenizer,
                    save_method="lora",
                    token="",
                )

        if include_gguf:
            model.save_pretrained_gguf(
                model_names["local-gguf"],
                tokenizer,
                quantization_method=quantization_method,
            )

            if publish:
                model.push_to_hub_gguf(
                    model_names["hub-gguf"],
                    tokenizer,
                    quantization_method=quantization_method,
                    token=token,
                )
    except Exception as e:
        print(e)
