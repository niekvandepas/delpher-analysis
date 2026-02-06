import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from constants import (
    FIETJE_SENTIMENT_RESULTS_PATH,
    OLLAMA_SENTIMENT_RESULTS_PATH,
    ROBBERT_SENTIMENT_RESULTS_PATH,
    ANNOTATED_DATA_DIR,
)
import json
from typing import TypedDict
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import curses

from delpher_types import SentimentResult

SentimentResults = TypedDict(
    "SentimentResults",
    {
        "robbert": list[SentimentResult],
        "fietje": list[SentimentResult],
        "ollama": list[SentimentResult],
    },
)


def get_sentiment_results() -> SentimentResults:
    """Loads sentiment results from all three models."""
    with open(FIETJE_SENTIMENT_RESULTS_PATH, "r") as f:
        json_results = json.load(f)
        fietje_results: list[SentimentResult] = []
        for res in json_results:
            fietje_results.append(
                SentimentResult(
                    text=res["text"],
                    identifier=res["identifier"],
                    sentiment_label=res["sentiment_label"],
                )
            )

    with open(ROBBERT_SENTIMENT_RESULTS_PATH, "r") as f:
        json_results = json.load(f)
        robbert_results: list[SentimentResult] = []
        for res in json_results:
            robbert_results.append(
                SentimentResult(
                    text=res["text"],
                    identifier=res["identifier"],
                    sentiment_label=res["sentiment_label"],
                )
            )

    with open(OLLAMA_SENTIMENT_RESULTS_PATH, "r") as f:
        json_results = json.load(f)
        ollama_results: list[SentimentResult] = []
        for res in json_results:
            ollama_results.append(
                SentimentResult(
                    text=res["text"],
                    identifier=res["identifier"],
                    sentiment_label=res["sentiment_label"],
                )
            )

    return {
        "fietje": fietje_results,
        "robbert": robbert_results,
        "ollama": ollama_results,
    }


def construct_review_set(entries: list[SentimentResult]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "id": i,
                "predicted_sentiment": entry.sentiment_label,
                "original_text": entry.text,
            }
            for i, entry in enumerate(entries)
        ]
    )

    # Separate the classes
    positive_df = df[df["predicted_sentiment"].str.lower() == "positive"]
    negative_df = df[df["predicted_sentiment"].str.lower() == "negative"]
    neutral_df = df[df["predicted_sentiment"].str.lower() == "neutral"]

    # Sample 40 from each (or all if less than 40)
    sample_positive = positive_df.sample(n=min(40, len(positive_df)), random_state=42)
    sample_negative = negative_df.sample(n=min(40, len(negative_df)), random_state=42)
    sample_neutral = neutral_df.sample(n=min(40, len(neutral_df)), random_state=42)

    if len(positive_df) < 40:
        logging.warning(
            f"Only {len(positive_df)} entries with 'positive' sentiment, sampled all."
        )
    if len(negative_df) < 40:
        logging.warning(
            f"Only {len(negative_df)} entries with 'negative' sentiment, sampled all."
        )
    if len(neutral_df) < 40:
        logging.warning(
            f"Only {len(neutral_df)} entries with 'neutral' sentiment, sampled all."
        )

    return pd.concat([sample_positive, sample_negative, sample_neutral]).sample(
        frac=1, random_state=42
    )


def remove_previously_annotated_entries(
    full_set: pd.DataFrame, previously_annotated_df: pd.DataFrame
) -> pd.DataFrame:
    if previously_annotated_df.empty:
        return full_set

    # HACK: (As below) match entries and the annotated dataframe on fulltext, because I was stupid enough not to return IDs from the Ollama classifier...
    annotated_texts = set(previously_annotated_df["original_text"])
    remaining = full_set[~full_set["original_text"].isin(annotated_texts)]

    return remaining


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
                snippet = row["original_text"]
                snippet = snippet[: width * (height - 6)]  # crude wrap-cut

                stdscr.addstr(0, 0, header)
                stdscr.addstr(2, 0, "TEXT:")
                stdscr.addstr(3, 0, snippet)

                stdscr.addstr(
                    height - 3,
                    0,
                    "Sentiment? (p = positive, n = negative, o = neutral, q = quit)",
                )
                stdscr.refresh()

                ch = stdscr.getch()
                if ch in (ord("p"), ord("P")):
                    annotations.append("POSITIVE")
                    break
                elif ch in (ord("n"), ord("N")):
                    annotations.append("NEGATIVE")
                    break
                elif ch in (ord("o"), ord("O")):
                    annotations.append("NEUTRAL")
                    break
                elif ch in (ord("q"), ord("Q")):
                    return

    curses.wrapper(main)

    result = df.iloc[: len(annotations)].copy()
    result["true_label"] = annotations
    return result


def print_metrics(df: pd.DataFrame):
    if "true_label" not in df.columns:
        return

    y_true = df["true_label"]
    y_pred = df["sentiment_label"]

    print("\n" + "=" * 30)
    print("SENTIMENT EVALUATION")
    print("=" * 30)

    labels = sorted(df["true_label"].unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    for i, label in enumerate(labels):
        print(f"{label:>10}: {cm[i]}")

    print("\nDetailed Metrics:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


def evaluate_model_performance(
    model_name: str,
    entries: list[SentimentResult],
    annotated_df: pd.DataFrame,
) -> None:
    df = annotated_df.copy()

    # map text -> predicted label
    text_to_label = {e.text: e.sentiment_label for e in entries}
    df["predicted_sentiment"] = df["original_text"].map(text_to_label)

    unmatched = df[df["predicted_sentiment"].isna()]
    if not unmatched.empty:
        print(f"[{model_name}] Warning: {len(unmatched)} unmatched rows")

    print(f"\n=== {model_name} ===")
    print_metrics(df)


sentiment_results = get_sentiment_results()
model_name = "robbert"

model_results = sentiment_results[model_name]

df = pd.DataFrame(model_results)

labels = df["sentiment_label"].unique()

review_set = construct_review_set(model_results)

annotated_data_path = Path(ANNOTATED_DATA_DIR) / "sentiment_annotated.csv"

try:
    previously_annotated_df = pd.read_csv(annotated_data_path)
except FileNotFoundError:
    previously_annotated_df = pd.DataFrame()

entries_to_annotate = remove_previously_annotated_entries(
    review_set, previously_annotated_df
)
newly_annotated_df = annotate_entries_curses(entries_to_annotate)
all_annotated_entries_df = pd.concat(
    [previously_annotated_df, newly_annotated_df], ignore_index=True
)
all_annotated_entries_df.to_csv(annotated_data_path, index=False)

evaluate_model_performance(model_name, model_results, all_annotated_entries_df)
