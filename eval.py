import os
import torch
from dotenv import find_dotenv, load_dotenv
from translation_engine_v3 import *
from translation_utils import *

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

model_name = os.getenv("MODEL_NAME")
load_in_4bit = os.getenv("LOAD_IN_4BIT") == "true"
eval_base_model = os.getenv("EVAL_BASE_MODEL") == "true"
eval_fine_tuned = os.getenv("EVAL_FINE_TUNED") == "true"
save_fine_tuned_model = os.getenv("SAVE_FINE_TUNED") == "true"
num_train_epochs = int(os.getenv("NUM_TRAIN_EPOCHS") or 0)
data_path = os.getenv("DATA_PATH")
results_path = os.getenv("RESULTS_PATH")

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)

print(
    model_name,
    load_in_4bit,
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

model, tokenizer = load_model(model_name, load_in_4bit=load_in_4bit)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"(2) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

datasets = load_translation_dataset(data_path, tokenizer)

print("Evaluating model: " + model_name)
predictions = eval_model(model, tokenizer, datasets["test"])

# calc_metrics(datasets["test"]["english"], predictions, debug=True)

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
