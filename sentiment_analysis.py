from typing import TypedDict
from sklearn.pipeline import Pipeline  # type: ignore
from transformers import TranslationPipeline, pipeline  # type: ignore
from typing import Optional


class SentimentResult(TypedDict):
    text: str
    label: str
    score: float


def analyze_sentiments(texts: list[str]) -> list[SentimentResult]:
    # English-language model
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Try to detect tokenizer/model max length; fall back to 512
    model_max_length: Optional[int] = None
    try:
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None:
            # common attribute name
            model_max_length = getattr(tokenizer, "model_max_length", None)
            if model_max_length is None:
                model_max_length = getattr(tokenizer, "max_len", None)
    except Exception:
        model_max_length = None

    if model_max_length is None:
        model_max_length = 512

    sentiment_results = sentiment_analysis(pipe, texts, max_text_length=10000, model_max_length=model_max_length)  # type: ignore
    return sentiment_results


def sentiment_analysis(
    analysis_pipeline: Pipeline,
    texts: list[str],
    max_text_length: int = 10000,
    model_max_length: int = 512,
) -> list[SentimentResult]:
    """Run sentiment analysis on texts with safe truncation.

    - `max_text_length` is an optional character-length filter before calling the model.
    - `model_max_length` (in tokens) is forwarded to the tokenizer via the pipeline
      as `max_length` and `truncation=True` to avoid tensor-size mismatches.
    """
    results: list[SentimentResult] = []
    skipped_counter = 0

    for i, text in enumerate(texts, start=1):
        print(f"Analyzing text #{i}, skipped: {skipped_counter}", end="\r")

        # Optional coarse-grained character-length filter
        if len(text) > max_text_length:
            skipped_counter += 1
            continue

        try:
            # Pass truncation and max_length to the pipeline so the tokenizer
            # truncates long inputs instead of producing oversized tensors.
            result = analysis_pipeline(text, truncation=True, max_length=model_max_length)[0]  # type: ignore
        except Exception as e:
            # Log and skip problematic texts rather than crashing.
            print(f"\nWarning: pipeline failed for text #{i}: {e}")
            skipped_counter += 1
            continue

        results.append(SentimentResult(text=text, label=result["label"], score=result["score"]))

    return results

def test_deberta():
    # Use a pipeline as a high-level helper
    from transformers import pipeline

    pipe = pipeline("fill-mask", model="microsoft/deberta-v3-base")
    result = pipe("I am a <mask> dog")
    print(result)
