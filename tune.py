import os
import sys
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from translation_engine import *
import torch

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

model_name = os.getenv("MODEL_NAME")
token = os.getenv("HF_TOKEN") or None
load_in_4bit = os.getenv("LOAD_IN_4BIT") == "true"
eval_base_model = os.getenv("EVAL_BASE_MODEL") == "true"
eval_fine_tuned = os.getenv("EVAL_FINE_TUNED") == "true"
save_fine_tuned_model = os.getenv("SAVE_FINE_TUNED") == "true"
num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS") or 0)
data_path = os.getenv("DATA_PATH")
results_path = os.getenv("RESULTS_PATH")

hub_model = model_name.split("/")[-1] + "-MAC-"
local_model = "models/" + hub_model

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)

print(
    model_name,
    load_in_4bit,
    local_model,
    hub_model,
    max_seq_length,
    num_train_epochs,
    dtype,
    data_path,
    results_path,
    eval_base_model,
    eval_fine_tuned,
    save_fine_tuned_model,
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"(1) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

model, tokenizer = load_model(model_name, load_in_4bit)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"(2) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

datasets = load_translation_dataset(data_path, tokenizer)

if eval_base_model:
    print("Evaluating base model: " + model_name)
    predictions = eval_model(model, tokenizer, datasets["test"])

    calc_metrics(datasets["test"]["english"], predictions, debug=True)

    save_results(
        model_name,
        results_path,
        datasets["test"],
        predictions,
        debug=True,
    )

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"(3) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=datasets["train"],
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
        output_dir="outputs",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"(4) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"(5) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

if eval_fine_tuned:
    print("Evaluating fine-tuned model: " + model_name)
    predictions = eval_model(model, tokenizer, datasets["test"])
    calc_metrics(datasets["test"]["english"], predictions, debug=True)

    save_results(
        model_name + "(finetuned)",
        results_path,
        datasets["test"],
        predictions,
        debug=True,
    )

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"(6) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


def save_model(model, tokenizer, save_method, publish=True):
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


def save_model_gguf(model, tokenizer, quantization_method, publish=True):
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


if save_fine_tuned_model:
    save_model(model, tokenizer, "merged_4bit_forced")
    save_model_gguf(model, tokenizer, quantization_method="q5_k_m")
