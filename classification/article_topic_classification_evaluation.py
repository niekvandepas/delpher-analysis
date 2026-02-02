from pathlib import Path
import sys
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import curses
import json

# HACK: allow importing from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from delpher_types import OllamaClassificationResult
from constants import ANNOTATED_DATA_DIR

CATEGORIES = ["Duurzaamheid", "Gezondheid", "Economie", "Smaak", "Traditie"]
CATEGORY_KEYS = {ord(str(i + 1)): cat for i, cat in enumerate(CATEGORIES)}


def has_valid_category_value(line: str):
    return any(line.startswith(f"Predicted Category: {c}") for c in CATEGORIES)


def get_classification_results(file_path: str) -> list[OllamaClassificationResult]:
    with open(file_path, "r") as f:
        values: list[dict] = json.load(f)
        return [OllamaClassificationResult(**v) for v in values]


def annotate_entries_curses(df: pd.DataFrame) -> pd.DataFrame:
    annotations = []
    records = df.to_dict("records")

    def main(stdscr):
        curses.curs_set(0)
        stdscr.clear()
        total = len(records)

        for i, row in enumerate(records):
            while True:
                stdscr.erase()
                height, width = stdscr.getmaxyx()

                header = f"Item {i+1}/{total}"
                snippet = row["original_text"][: width * (height - 10)]

                stdscr.addstr(0, 0, header)
                stdscr.addstr(2, 0, "TEXT:")
                stdscr.addstr(3, 0, snippet)

                stdscr.addstr(height - 8, 0, "Select category:")
                for idx, cat in enumerate(CATEGORIES):
                    stdscr.addstr(height - 7 + idx, 2, f"{idx+1} = {cat}")

                stdscr.addstr(height - 1, 0, "q = quit")
                stdscr.refresh()

                ch = stdscr.getch()

                if ch in CATEGORY_KEYS:
                    annotations.append(CATEGORY_KEYS[ch])
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
    y_pred = df["predicted_category"]

    print("\n" + "=" * 30)
    print("EVALUATION RESULTS")
    print("=" * 30)

    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES))

    print("\nDetailed Metrics:")
    print(
        classification_report(
            y_true, y_pred, labels=CATEGORIES, target_names=CATEGORIES, zero_division=0
        )
    )


entries = get_classification_results(file_path="output/article_topics_llama_8b.json")

df = pd.DataFrame(
    [
        {
            "id": i,
            "predicted_category": entry.label,
            "original_text": entry.text,
        }
        for i, entry in enumerate(entries)
    ]
)

# ---- Sampling (balanced across predicted classes) ----

samples = []
for category in CATEGORIES:
    subset = df[df["predicted_category"] == category]
    if len(subset) > 0:
        samples.append(subset.sample(n=min(50, len(subset)), random_state=42))

eval_set = pd.concat(samples).sample(frac=1, random_state=42)

annotated_set = annotate_entries_curses(eval_set)
print_metrics(annotated_set)

save_path = Path(ANNOTATED_DATA_DIR) / "article_topics_annotated.csv"
annotated_set.to_csv(save_path, index=False)
