from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import curses
import json
from pathlib import Path
import sys

# HACK: allow importing from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from constants import ANNOTATED_DATA_DIR
from delpher_types import OllamaClassificationResult


def has_valid_category_value(line: str):
    if line.startswith("Predicted Category: Is about food"):
        return True
    if line.startswith("Predicted Category: Is not about food"):
        return True
    return False


def get_classification_results(file_path: str) -> list[OllamaClassificationResult]:
    with open(file_path, "r") as f:
        values: list[dict] = json.load(f)
        return [OllamaClassificationResult(**v) for v in values]


def construct_review_set(entries: list[OllamaClassificationResult]) -> pd.DataFrame:
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

    # Separate the classes
    food_df = df[df["predicted_category"] == "Is about food"]
    not_food_df = df[df["predicted_category"] != "Is about food"]

    # Sample 50 from each (or all if less than 50)
    sample_food = food_df.sample(n=50, random_state=42)
    sample_not_food = not_food_df.sample(n=min(50, len(not_food_df)), random_state=42)

    return pd.concat([sample_food, sample_not_food]).sample(frac=1, random_state=42)


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

                # Prepare the text
                header = f"Item {i+1}/{total}"
                snippet = row["original_text"]

                stdscr.addstr(0, 0, header)
                stdscr.addstr(2, 0, "TEXT:")
                stdscr.addstr(3, 0, snippet)

                stdscr.addstr(
                    height - 3, 0, "Is this about food? (y = yes, n = no, q = quit)"
                )
                stdscr.refresh()

                ch = stdscr.getch()
                if ch in (ord("y"), ord("Y")):
                    annotations.append("Is about food")
                    break
                elif ch in (ord("n"), ord("N")):
                    annotations.append("Is not about food")
                    break
                elif ch in (ord("q"), ord("Q")):
                    # quit early
                    return

    curses.wrapper(main)

    # Build output df (partial or full)
    result = df.iloc[: len(annotations)].copy()
    result["true_label"] = annotations
    return result


def evaluate_model_performance(
    entries: list[OllamaClassificationResult], annotated_df: pd.DataFrame
) -> None:
    # Substitute dinges
    df = annotated_df.copy()

    # HACK: Match entries and the annotated dataframe on fulltext, because I was stupid enough not to return IDs from the Ollama classifier...
    text_to_label = {e.text: e.label for e in entries}
    df["predicted_category"] = df["original_text"].map(text_to_label)

    unmatched = df[df["predicted_category"].isna()]
    if not unmatched.empty:
        # If this point has been reached, the text mapping did not work perfectly, probably due to unicode conversion or some other lossy reading... again, this is why IDs exist :)
        print(f"Warning: {len(unmatched)} rows did not match any new label")

    print_metrics(df)


def print_metrics(df: pd.DataFrame):
    """Calculates and prints confusion matrix and classification report."""
    if "true_label" not in df.columns:
        return

    y_true = df["true_label"]
    y_pred = df["predicted_category"]

    print("\n" + "=" * 30)
    print("EVALUATION RESULTS")
    print("=" * 30)

    # Confusion Matrix
    labels = ["Is about food", "Is not about food"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(f"True Pos (Food): {cm[0][0]} | False Neg (Missed Food): {cm[0][1]}")
    print(
        f"False Pos (Wrongly Food): {cm[1][0]} | True Neg (Correct Non-Food): {cm[1][1]}"
    )

    # Classification Report
    print("\nDetailed Metrics:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


entries = get_classification_results(
    file_path="output/about_food_ollama3_70b_1000.json"
)
review_set = construct_review_set(entries)

annotated_data_path = Path(ANNOTATED_DATA_DIR) / "is_about_food_annotated.csv"

annotated_df = annotate_entries_curses(review_set)
# annotated_df.to_csv(annotated_data_path, index=False)

annotated_df = pd.read_csv(annotated_data_path)
evaluate_model_performance(entries, annotated_df)
