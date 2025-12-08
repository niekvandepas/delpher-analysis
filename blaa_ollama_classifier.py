from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- Configuration ---
# IMPORTANT: Ensure your manually labeled CSV file is saved as this name:
INPUT_FILE = 'annotated_evaluation_set.csv'
TARGET_COLUMN = 'true_label'
TEXT_COLUMN = 'original_text'
# ---------------------

def load_data(file_path):
    """Loads and cleans the annotated data."""
    try:
        # Tries to load assuming a standard CSV format (Pandas is robust)
        df = pd.read_csv(file_path)
    except Exception:
        # If standard load fails (e.g., due to Excel/semicolon issues)
        print("Attempting load with semicolon delimiter...")
        df = pd.read_csv(file_path, sep=';')

    # Ensure necessary columns exist and handle missing values
    if TARGET_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing '{TARGET_COLUMN}' or '{TEXT_COLUMN}' column in the input file.")

    df = df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN])
    print(f"Loaded {len(df)} annotated samples.")
    return df

def train_and_evaluate_model(df):
    """
    Trains a Logistic Regression model using TF-IDF vectorization
    and evaluates its performance.
    """
    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    # Split data: 80% for training the new model, 20% for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Feature Engineering (TF-IDF Vectorization)
    # This converts text into numerical features weighted by importance.
    stop_words = stopwords.words('dutch')

    vectorizer = TfidfVectorizer(max_features=2000, stop_words=stop_words)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Note: TF-IDF is the process that turns words into numbers
    # the model can understand. This is a crucial step!
    #

    # 2. Model Training (Logistic Regression)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vectorized, y_train)

    # 3. Prediction and Evaluation
    y_pred = model.predict(X_test_vectorized)

    print("\n" + "="*50)
    print("NEW CLASSIFIER PERFORMANCE (Logistic Regression)")
    print("="*50)

    # Compare performance to the LLM baseline
    print(f"Test Set Size: {len(X_test)} samples (20% of annotations)")
    print("\nConfusion Matrix (True Labels vs. Predicted Labels):")
    print(confusion_matrix(y_test, y_pred, labels=['Is about food', 'Is not about food']))

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model, vectorizer

# --- Execution ---

if __name__ == "__main__":
    annotated_data = load_data(INPUT_FILE)

    if len(annotated_data) < 20:
        print("Warning: Insufficient data for training/testing. Need at least 20 samples.")
    else:
        trained_model, trained_vectorizer = train_and_evaluate_model(annotated_data)

        # Next step: Use the trained_model and trained_vectorizer to classify
        # the remaining 2,800+ documents from your original ollama_out.txt corpus.
        print("\nModel training complete. You now have a fast, specialized classifier.")
