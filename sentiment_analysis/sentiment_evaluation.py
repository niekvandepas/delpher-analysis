import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from constants import FIETJE_SENTIMENT_RESULTS_PATH, OLLAMA_SENTIMENT_RESULTS_PATH, ROBBERT_SENTIMENT_RESULTS_PATH
import json
from typing import TypedDict
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import curses

from delpher_types import SentimentResult

SentimentResults = TypedDict('SentimentResults', {'robbert': list[SentimentResult], 'fietje': list[SentimentResult], 'ollama': list[SentimentResult]})

def get_evaluation_results() -> SentimentResults:
    """Loads sentiment results from all three models."""
    with open(FIETJE_SENTIMENT_RESULTS_PATH, "r") as f:
        fietje_results = json.load(f)

    with open(ROBBERT_SENTIMENT_RESULTS_PATH, "r") as f:
        robbert_results = json.load(f)

    with open(OLLAMA_SENTIMENT_RESULTS_PATH, "r") as f:
        ollama_results = json.load(f)

    return {
        'fietje': fietje_results,
        'robbert': robbert_results,
        'ollama': ollama_results,
    }

def annotate_entries_curses(df: pd.DataFrame) -> pd.DataFrame:
    annotations = []
    records = df.to_dict("records")

    def main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.clear()

        total = len(records)

        for i, row in enumerate(records):
            while True:
                stdscr.erase()
                height, width = stdscr.getmaxyx()

                header = f"Item {i+1}/{total}"
                snippet = row["text"]
                snippet = snippet[: width * (height - 6)]  # crude wrap-cut

                stdscr.addstr(0, 0, header)
                stdscr.addstr(2, 0, "TEXT:")
                stdscr.addstr(3, 0, snippet)

                stdscr.addstr(height - 3, 0,
                    "Sentiment? (p = positive, n = negative, o = neutral, q = quit)"
                )
                stdscr.refresh()

                ch = stdscr.getch()
                if ch in (ord("p"), ord("P")):
                    annotations.append("POSITIEF")
                    break
                elif ch in (ord("n"), ord("N")):
                    annotations.append("NEGATIEF")
                    break
                elif ch in (ord("o"), ord("O")):
                    annotations.append("NEUTRAAL")
                    break
                elif ch in (ord("q"), ord("Q")):
                    return

    curses.wrapper(main)

    result = df.iloc[:len(annotations)].copy()
    result["true_label"] = annotations
    return result

def print_metrics(df: pd.DataFrame):
    if "true_label" not in df.columns:
        return

    y_true = df["true_label"]
    y_pred = df["sentiment_label"]

    print("\n" + "="*30)
    print("SENTIMENT EVALUATION")
    print("="*30)

    labels = sorted(df["true_label"].unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    for i, label in enumerate(labels):
        print(f"{label:>10}: {cm[i]}")


    print("\nDetailed Metrics:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

evaluation_results = get_evaluation_results()

model_results = evaluation_results["fietje"]  # or "robbert"

df = pd.DataFrame(model_results)

labels = df["sentiment_label"].unique()

sampled = []

for label in labels:
    subset = df[df["sentiment_label"] == label]
    sampled.append(subset.sample(n=min(5, len(subset)), random_state=42))

eval_set = pd.concat(sampled).sample(frac=1, random_state=42)

annotated_set = annotate_entries_curses(eval_set)
print_metrics(annotated_set)

annotated_set.to_csv("annotated_sentiment_evaluation.csv", index=False)
