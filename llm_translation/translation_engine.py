import os
import re
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from llm_translation.transformers import TextStreamer
from tqdm import tqdm
from llm_translation.transformers import TextStreamer

print(f"loading {__file__}")


def load_model(model_name, max_seq_length=2048, dtype=None, load_in_4bit=False):
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
