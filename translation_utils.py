import os
import re
import pandas as pd
import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
accuracy = evaluate.load("accuracy")


def extract_answer(text, debug=False):
    if text:
        # Remove the begin and end tokens
        text = re.sub(r".*?assistant.+?\b", "", text, flags=re.DOTALL | re.MULTILINE)
        if debug:
            print("--------\nstep 1:", text)

        text = re.sub(r"<\|.*?\|>.*", "", text, flags=re.DOTALL | re.MULTILINE)
        if debug:
            print("--------\nstep 2:", text)

        text = re.sub(
            r".*?end_header_id\|>\n\n", "", text, flags=re.DOTALL | re.MULTILINE
        )
        if debug:
            print("--------\nstep 3:", text)

    return text


def calc_metrics(references, predictions, debug=False):
    assert len(references) == len(
        predictions
    ), f"lengths are difference: {len(references)} != {len(predictions)}"

    predictions = [extract_answer(text) for text in predictions]

    correct = [1 if ref == pred else 0 for ref, pred in zip(references, predictions)]
    accuracy = sum(correct) / len(references)

    results = {"accuracy": accuracy}
    if debug:
        correct_ids = [i for i, c in enumerate(correct) if c == 1]
        results["correct_ids"] = correct_ids

    results["meteor"] = meteor.compute(predictions=predictions, references=references)[
        "meteor"
    ]

    results["bleu_scores"] = bleu.compute(
        predictions=predictions, references=references, max_order=4
    )
    results["rouge_scores"] = rouge.compute(
        predictions=predictions, references=references
    )
    return results


def save_results(model_name, results_path, dataset, predictions, debug=False):
    if not os.path.exists(results_path):
        # Get the directory part of the file path
        dir_path = os.path.dirname(results_path)

        # Create all directories in the path (if they don't exist)
        os.makedirs(dir_path, exist_ok=True)
        df = dataset.to_pandas()
        df.drop(columns=["text", "prompt"], inplace=True)
    else:
        df = pd.read_csv(results_path, on_bad_lines="warn")

    df[model_name] = predictions

    if debug:
        print(df.head(1))

    df.to_csv(results_path, index=False)
