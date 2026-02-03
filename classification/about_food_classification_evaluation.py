from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import curses

def has_valid_category_value(line: str):
    if line.startswith("Predicted Category: Is about food"):
        return True
    if line.startswith("Predicted Category: Is not about food"):
        return True
    return False


def get_valid_entries():
    valid_line_starts = [
      "ID",
      "Predicted Category",
      "Original Text",
      "--------------------------------------------------"
    ]

    with open("ollama_out.txt", "r") as f:
        lines = f.readlines()
        valid_lines = []

        # Ignore lines that don't start with ID, Predicted Category, or Original Text
        for line in lines:
            for valid_line_start in valid_line_starts:
                if line.startswith(valid_line_start):
                    if line.startswith("Predicted Category"):
                        if has_valid_category_value(line):
                            valid_lines.append(line)
                    else:
                        valid_lines.append(line)

        entries = []
        current_entry = []
        for i, line in enumerate(valid_lines):
            if i == 304:
                ...
            if line.strip() == "--------------------------------------------------":
                if current_entry:
                    entries.append(current_entry)
                    current_entry = []
            else:
                current_entry.append(line.strip())

        valid_entries = []
        for entry in entries:
            if len(entry) == 3:
                valid_entries.append(entry)

        return valid_entries


entries = get_valid_entries()

df = pd.DataFrame([
    {
        'id': entry[0].replace("ID:", "").strip(),
        'predicted_category': entry[1].replace("Predicted Category:", "").strip(),
        'original_text': entry[2].replace("Original Text:", "").strip()
    }
    for entry in entries
])

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
                snippet = snippet[: width * (height - 6)]  # very crude wrap-cut

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


# Separate the classes
food_df = df[df["predicted_category"] == "Is about food"]
not_food_df = df[df["predicted_category"] != "Is about food"]

# Sample 50 from each (or all if less than 50)
sample_food = food_df.sample(n=50, random_state=42)
sample_not_food = not_food_df.sample(n=min(50, len(not_food_df)), random_state=42)

# Combine and shuffle
eval_set = pd.concat([sample_food, sample_not_food]).sample(frac=1, random_state=42)

annotated_set = annotate_entries_curses(eval_set)
print_metrics(annotated_set)

annotated_set.to_csv("annotated_evaluation_set.csv", index=False)
