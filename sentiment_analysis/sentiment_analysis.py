from time import time
from sklearn.pipeline import Pipeline  # type: ignore
from transformers import TranslationPipeline, pipeline, AutoTokenizer, AutoModelForCausalLM # type: ignore
from typing import Any, List, Optional, cast
import torch


from delpher_types import PlainTextSearchResult, SentimentLabel, SentimentResult, TranslatedSearchResult



def analyze_sentiments_robbert(texts: list[PlainTextSearchResult]) -> list[SentimentResult]:
    sentiment_analysis_pipeline = pipeline(
        "text-classification", model="DTAI-KULeuven/robbert-v2-dutch-sentiment", device=0
    )

    sentiment_results = sentiment_analysis_dutch(sentiment_analysis_pipeline, texts)
    return sentiment_results


def sentiment_analysis_dutch(
    analysis_pipeline: Pipeline, texts: list[PlainTextSearchResult]
) -> list[SentimentResult]:

    plain_texts = [t.plain_text for t in texts]

    outputs = cast(
        list[dict[str, Any]],
        analysis_pipeline(
            plain_texts,
            truncation=True,
            max_length=512,
            batch_size=32,
        ),
    )


    results: list[SentimentResult] = []
    for search_result, out in zip(texts, outputs):
        if out["label"] == "Positive":
            normalized_label = SentimentLabel.POSITIVE
        elif out["label"] == "Negative":
            normalized_label = SentimentLabel.NEGATIVE
        elif out["label"] == "Neutral":
            normalized_label = SentimentLabel.NEUTRAL

        results.append(SentimentResult(
            text=search_result.plain_text,
            identifier=search_result.identifier or "",
            sentiment_label=normalized_label,
            sentiment_score=out["score"],
        ))

    return results

def analyze_sentiments_english(texts: list[TranslatedSearchResult]) -> list[SentimentResult]:
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

    sentiment_results = sentiment_analysis_english(pipe, texts, max_text_length=10000, model_max_length=model_max_length)  # type: ignore
    return sentiment_results


def sentiment_analysis_english(
    analysis_pipeline: Pipeline,
    texts: list[TranslatedSearchResult],
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

    for i, search_result in enumerate(texts, start=1):
        print(f"Analyzing text #{i}, skipped: {skipped_counter}", end="\r")
        text = search_result.english_translated_text

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

        sentiment_result = SentimentResult(
            text=text,
            identifier=search_result.identifier or "",
            sentiment_label=result["label"],
            sentiment_score=result["score"],
        )

        results.append(sentiment_result)

    return results

def test_deberta():
    # Use a pipeline as a high-level helper
    from transformers import pipeline

    pipe = pipeline("fill-mask", model="microsoft/deberta-v3-base")
    result = pipe("I am a <mask> dog")
    print(result)















def analyze_sentiments_dutch_fietje(
    search_results: List[PlainTextSearchResult]
) -> List[SentimentResult]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("BramVanroy/fietje-2")
    model = AutoModelForCausalLM.from_pretrained(
        "BramVanroy/fietje-2",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    def classify_text(search_result: PlainTextSearchResult) -> SentimentResult:
        label, score = classify_sentiment_fietje(model, tokenizer, search_result.plain_text[:1500])
        if label == "POSITIEF":
            normalized_label = SentimentLabel.POSITIVE
        elif label == "NEGATIEF":
            normalized_label = SentimentLabel.NEGATIVE
        else:
            normalized_label = SentimentLabel.NEUTRAL

        return SentimentResult(
            text=search_result.plain_text,
            identifier=search_result.identifier or "",
            sentiment_label=normalized_label,
            sentiment_score=score,
        )

    results = []

    for i, search_result in enumerate(search_results):
        start_time = time()
        print(f"Analyzing text #{i+1}") # Cannot use \r because torch warnings would overwrite the line
        classification = classify_text(search_result)
        results.append(classification)
        end_time = time()
        print(f" Done in {end_time - start_time:.2f} seconds.")

    return results

def classify_sentiment_fietje(model, tokenizer, text: str) -> tuple[str, float]:
    prompt = (
        "Je bent een sentimentanalyse-model. "
        "Classificeer de volgende tekst als POSITIEF, NEUTRAAL of NEGATIEF.\n\n"
        f"Tekst: \"{text}\"\n\n"
        "Antwoord exact in het volgende JSON-formaat:\n"
        "{\"label\": \"POS/NEUTRAAL/NEG\", \"score\": 0.xx}\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.0
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract JSON from the tail of the output
    # We keep this simple and robust
    import json
    import re

    match = re.search(r'\{.*\}', output_text, flags=re.DOTALL)
    if not match:
        return "NEUTRAAL", 0.50

    try:
        data = json.loads(match.group(0))
        label = data.get("label", "NEUTRAAL")
        score = float(data.get("score", 0.50))
    except Exception:
        return "NEUTRAAL", 0.50

    return label.upper(), score
