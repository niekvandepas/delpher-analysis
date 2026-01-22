from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from time import time
import ollama
from sklearn.pipeline import Pipeline  # type: ignore
from transformers import TranslationPipeline, pipeline, AutoTokenizer, AutoModelForCausalLM # type: ignore
from typing import Any, List, Optional, cast
import torch


from delpher_types import PlainTextSearchResult, SentimentLabel, SentimentResult, TranslatedSearchResult



def analyze_sentiments_robbert(texts: list[PlainTextSearchResult]) -> list[SentimentResult]:
    """
    Performs Dutch sentiment analysis using RoBERTa model,
    utilizing multithreading for increased throughput.
    """
    NUM_WORKERS_ROBBERT = os.environ.get("NUM_WORKERS_ROBBERT")
    if NUM_WORKERS_ROBBERT:
        NUM_WORKERS_ROBBERT = int(NUM_WORKERS_ROBBERT)
    else:
        print("Environment variable NUM_WORKERS_ROBBERT not set, defaulting to 1.")
        NUM_WORKERS_ROBBERT = 1

    def process_single_text(search_result: PlainTextSearchResult) -> SentimentResult:
        """Processes a single text in a dedicated thread."""
        start_time = time()

        # Each thread gets its own pipeline
        sentiment_pipeline = pipeline(
            "text-classification",
            model="DTAI-KULeuven/robbert-v2-dutch-sentiment",
            device=0
        )

        try:
            output = sentiment_pipeline(
                search_result.plain_text,
                truncation=True,
                max_length=512,
            )[0]

            if output["label"] == "Positive":
                normalized_label = SentimentLabel.POSITIVE
            elif output["label"] == "Negative":
                normalized_label = SentimentLabel.NEGATIVE
            elif output["label"] == "Neutral":
                normalized_label = SentimentLabel.NEUTRAL
            else:
                normalized_label = SentimentLabel.NEUTRAL

            result = SentimentResult(
                text=search_result.plain_text,
                identifier=search_result.identifier or "",
                sentiment_label=normalized_label,
            )

        except Exception as e:
            print(f"Error analyzing item {search_result.identifier}: {e}")
            result = SentimentResult(
                text=search_result.plain_text,
                identifier=search_result.identifier or "",
                sentiment_label=SentimentLabel.NEUTRAL,
            )

        end_time = time()
        print(f"Processed item {search_result.identifier} in {end_time - start_time:.2f} seconds.")

        return result

    results: list[SentimentResult] = []
    print(f"Starting RoBERTa sentiment analysis on {len(texts)} items with {NUM_WORKERS_ROBBERT} workers...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS_ROBBERT) as executor:
        future_to_item = {
            executor.submit(process_single_text, item): item
            for item in texts
        }

        completed_count = 0
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)

            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(texts):
                print(f"Processed {completed_count}/{len(texts)}", end='\r')

    print(f"\nAnalysis complete. Processed {len(results)} items.")
    return results


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
    """
    Performs Dutch sentiment analysis using Fietje-2 model,
    utilizing multithreading for increased throughput.
    """
    NUM_WORKERS_FIETJE = os.environ.get("NUM_WORKERS_FIETJE")
    if NUM_WORKERS_FIETJE:
        NUM_WORKERS_FIETJE = int(NUM_WORKERS_FIETJE)
    else:
        print("Environment variable NUM_WORKERS_FIETJE not set, defaulting to 1.")
        NUM_WORKERS_FIETJE = 1

    def process_single_text(search_result: PlainTextSearchResult) -> SentimentResult:
        """Processes a single text in a dedicated thread."""
        start_time = time()

        # Each thread gets its own model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("BramVanroy/fietje-2")
        model = AutoModelForCausalLM.from_pretrained(
            "BramVanroy/fietje-2",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        try:
            label = classify_sentiment_fietje(model, tokenizer, search_result.plain_text[:1500])
            if label == "POSITIEF":
                normalized_label = SentimentLabel.POSITIVE
            elif label == "NEGATIEF":
                normalized_label = SentimentLabel.NEGATIVE
            else:
                normalized_label = SentimentLabel.NEUTRAL

            result = SentimentResult(
                text=search_result.plain_text,
                identifier=search_result.identifier or "",
                sentiment_label=normalized_label,
            )

        except Exception as e:
            print(f"Error analyzing item {search_result.identifier}: {e}")
            result = SentimentResult(
                text=search_result.plain_text,
                identifier=search_result.identifier or "",
                sentiment_label=SentimentLabel.NEUTRAL,
            )

        end_time = time()
        print(f"Processed item {search_result.identifier} in {end_time - start_time:.2f} seconds.")

        return result

    results: List[SentimentResult] = []
    print(f"Starting Fietje sentiment analysis on {len(search_results)} items with {NUM_WORKERS_FIETJE} workers...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS_FIETJE) as executor:
        future_to_item = {
            executor.submit(process_single_text, item): item
            for item in search_results
        }

        completed_count = 0
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)

            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(search_results):
                print(f"Processed {completed_count}/{len(search_results)}", end='\r')

    print(f"\nAnalysis complete. Processed {len(results)} items.")
    return results

def classify_sentiment_fietje(model, tokenizer, text: str) -> str:
    prompt = (
        "Je bent een sentimentanalyse-model. "
        "Classificeer de volgende tekst als POSITIEF, NEUTRAAL of NEGATIEF.\n\n"
        f"Tekst: \"{text}\"\n\n"
        "Antwoord exact in het volgende JSON-formaat:\n"
        "{\"label\": \"POSITIEF/NEUTRAAL/NEGATIEF\"}\n"
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
    import json
    import re

    match = re.search(r'\{.*\}', output_text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")

    data = json.loads(match.group(0))
    label = data.get("label", "NEUTRAAL")

    return label.upper()

def analyze_sentiments_dutch_ollama(
    search_results: List[PlainTextSearchResult]
) -> List[SentimentResult]:
    """
    Performs Dutch sentiment analysis using a local Ollama model,
    utilizing multithreading for increased throughput.
    """

    # Configuration (aligned with your classification script)
    MODEL_NAME = 'llama3:8b'
    NUM_WORKERS_OLLAMA = os.environ.get("NUM_WORKERS_OLLAMA")
    if NUM_WORKERS_OLLAMA:
        NUM_WORKERS_OLLAMA = int(NUM_WORKERS_OLLAMA)
    else:
        print("Environment variable NUM_WORKERS_OLLAMA not set, defaulting to 1.")
        NUM_WORKERS_OLLAMA = 1

    SYSTEM_PROMPT = (
        "Je bent een expert in sentimentanalyse voor Nederlandse teksten. "
        "Je taak is om het sentiment van de gegeven tekst te bepalen. "
        "De mogelijke categorieën zijn: POSITIEF, NEGATIEF of NEUTRAAL "
        "Antwoord UITSLUITEND met een JSON-object in het volgende formaat: "
        "{\"label\": \"POSITIEF\"}. "
        "Geen andere tekst of uitleg."
    )

    def process_single_sentiment(item: PlainTextSearchResult) -> SentimentResult:
        """Processes a single item in a dedicated thread."""
        start_time = time()

        client = ollama.Client()  # Each thread gets its own client

        # Truncate text to respect context limits
        text_content = item.plain_text[:2000] if item.plain_text else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text: {text_content}"}
        ]

        label = SentimentLabel.NEUTRAL
        score = 0.5

        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=messages,
                options={"temperature": 0.0},
                format="json"  # Force JSON output from Ollama
            )

            content = response['message']['content']
            data = json.loads(content)

            raw_label = data.get("label", "NEUTRAL").upper()

            if "POS" in raw_label:
                label = SentimentLabel.POSITIVE
            elif "NEG" in raw_label:
                label = SentimentLabel.NEGATIVE
            else:
                label = SentimentLabel.NEUTRAL

        except Exception as e:
            # Fallback to neutral on error
            print(f"Error analyzing item {item.identifier}: {e}")
            label = SentimentLabel.NEUTRAL

        end_time = time()
        print(f"Processed item {item.identifier} in {end_time - start_time:.2f} seconds.")

        return SentimentResult(
            text=item.plain_text,
            identifier=item.identifier or "",
            sentiment_label=label,
        )

    results: List[SentimentResult] = []
    print(f"Starting Ollama sentiment analysis on {len(search_results)} items with {NUM_WORKERS_OLLAMA} workers...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS_OLLAMA) as executor:
        future_to_item = {
            executor.submit(process_single_sentiment, item): item
            for item in search_results
        }

        completed_count = 0
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)

            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(search_results):
                print(f"Processed {completed_count}/{len(search_results)}", end='\r')

    print(f"\nAnalysis complete. Processed {len(results)} items.")
    return results
