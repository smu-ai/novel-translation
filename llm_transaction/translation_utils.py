import os
import re
import pandas as pd
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
accuracy = evaluate.load("accuracy")


def extract_answer(text, debug=False):
    if text:
        # Remove the begin and end tokens
        text = re.sub(
            r".*?(assistant|\[/INST\]).+?\b", "", text, flags=re.DOTALL | re.MULTILINE
        )
        if debug:
            print("--------\nstep 1:", text)

        text = re.sub(r"<.+?>.*", "", text, flags=re.DOTALL | re.MULTILINE)
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


def get_metrics(df):
    metrics_df = pd.DataFrame(df.columns.T)[2:]
    metrics_df.rename(columns={0: "model"}, inplace=True)
    metrics_df.reset_index(inplace=True)
    metrics_df = metrics_df.drop(columns=["index"])

    accuracy = []
    meteor = []
    bleu_1 = []
    rouge_l = []
    all_metrics = []
    for col in df.columns[2:]:
        metrics = calc_metrics(df["english"], df[col], debug=True)
        print(f"{col}: {metrics}")

        accuracy.append(metrics["accuracy"])
        meteor.append(metrics["meteor"])
        bleu_1.append(metrics["bleu_scores"]["bleu"])
        rouge_l.append(metrics["rouge_scores"]["rougeL"])
        all_metrics.append(metrics)

    metrics_df["accuracy"] = accuracy
    metrics_df["meteor"] = meteor
    metrics_df["bleu_1"] = bleu_1
    metrics_df["rouge_l"] = rouge_l
    metrics_df["all_metrics"] = all_metrics

    return metrics_df


def plot_metrics(metrics_df, figsize=(14, 5)):
    plt.figure(figsize=figsize)

    # Assuming `df` is your DataFrame and 'model', 'accuracy', 'meteor', 'bleu_1', 'rouge_l' are columns in `df`
    df_melted = pd.melt(
        metrics_df, id_vars="model", value_vars=["meteor", "bleu_1", "rouge_l"]
    )

    barplot = sns.barplot(x="variable", y="value", hue="model", data=df_melted)
    barplot.set_xticklabels(["METEOR", "BLEU-1", "ROUGE-L"])

    # show numbers on the bars
    for p in barplot.patches:
        if p.get_height() == 0:
            continue

        barplot.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    # set ylimit
    barplot.set(ylim=(0, 0.44), ylabel="Scores", xlabel="Metrics")

    # Move the legend to an empty part of the plot
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center")
    # plt.tight_layout()

    plt.show()
