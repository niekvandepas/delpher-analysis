from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from time import time
import ollama
from sklearn.pipeline import Pipeline  # type: ignore
from transformers import TranslationPipeline, pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification # type: ignore
from typing import Any, List, Optional, cast
import torch
import numpy as np
from scipy.special import softmax

from delpher_types import (
    PlainTextSearchResult,
    SentimentLabel,
    SentimentResult,
    TranslatedSearchResult,
)


def analyze_sentiments_robbert(
    texts: list[PlainTextSearchResult],
) -> list[SentimentResult]:
    """
    Robust Dutch sentiment analysis using RobBERT with Sliding Window for long texts.
    """
    MODEL_NAME = "DTAI-KULeuven/robbert-v2-dutch-sentiment"

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Move to GPU/MPS if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    def get_long_text_sentiment(text: str) -> tuple[SentimentLabel, float]:
        """
        Splits long text into 512-token chunks, processes them, and averages the scores.
        """
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"][0]

        CHUNK_SIZE = 510
        STRIDE = 256  # Overlap between chunks to preserve context

        # Split into chunks
        chunks = []
        if len(input_ids) <= CHUNK_SIZE:
            chunks.append(input_ids)
        else:
            # Create overlapping chunks
            for i in range(0, len(input_ids), STRIDE):
                chunk = input_ids[i : i + CHUNK_SIZE]
                chunks.append(chunk)
                if i + CHUNK_SIZE >= len(input_ids):
                    break

        all_probs = []

        with torch.no_grad():
            for chunk in chunks:
                # Add [CLS] and [SEP] tokens back manually
                full_chunk = torch.cat([
                    torch.tensor([tokenizer.cls_token_id], device=device),
                    chunk.to(device),
                    torch.tensor([tokenizer.sep_token_id], device=device)
                ])

                outputs = model(full_chunk.unsqueeze(0))
                logits = outputs.logits.cpu().numpy()[0]

                probs = softmax(logits)
                all_probs.append(probs)

        # Average the probabilities across all chunks
        avg_probs = np.mean(all_probs, axis=0)

        prediction_idx = np.argmax(avg_probs)
        score = float(avg_probs[prediction_idx])

        if prediction_idx == 1:
            return SentimentLabel.POSITIVE, score
        else:
            return SentimentLabel.NEGATIVE, score

    results = []
    print(f"Analyzing {len(texts)} texts with RobBERT (Sliding Window)...")

    for i, item in enumerate(texts):
        start = time()
        try:
            label, score = get_long_text_sentiment(item.plain_text)

            # Optional: Heuristic for Neutral
            # If the confidence score is very low (e.g. 0.51 vs 0.49), it might be neutral
            if 0.45 < score < 0.55:
                label = SentimentLabel.NEUTRAL

        except Exception as e:
            print(f"Error on item {i}: {e}")
            label = SentimentLabel.NEUTRAL
            score = 0.0

        results.append(SentimentResult(
            text=item.plain_text,
            identifier=item.identifier or "",
            sentiment_label=label
        ))

        print(f"Processed {i+1}/{len(texts)}: {label.value} ({score:.2f}) - {time()-start:.2f}s")

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

        results.append(
            SentimentResult(
                text=search_result.plain_text,
                identifier=search_result.identifier or "",
                sentiment_label=normalized_label,
            )
        )

    return results


def analyze_sentiments_english(
    texts: list[TranslatedSearchResult],
) -> list[SentimentResult]:
    # English-language model
    pipe = pipeline(
        "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

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
    search_results: List[PlainTextSearchResult],
) -> List[SentimentResult]:
    """
    Performs Dutch sentiment analysis using Fietje-2 model.
    """
    # Each thread gets its own model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BramVanroy/fietje-2b-instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "BramVanroy/fietje-2b-instruct", torch_dtype=torch.float32, device_map="auto"
    )

    # Set model to evaluation mode rather than training mode
    model.eval()

    def process_single_text(search_result: PlainTextSearchResult) -> SentimentResult:
        """Processes a single text."""
        start_time = time()
        try:
            label = classify_sentiment_fietje(
                model, tokenizer, search_result.plain_text[:1500]
            )
        except ValueError as e:
            raise ValueError(e)
        if label == "POSITIEF":
            normalized_label = SentimentLabel.POSITIVE
        elif label == "NEGATIEF":
            normalized_label = SentimentLabel.NEGATIVE
        elif label == "NEUTRAAL":
            normalized_label = SentimentLabel.NEUTRAL
        else:
            logging.warning(
                f"Unexpected label '{label}' from Fietje, defaulting to NEUTRAAL."
            )

        result = SentimentResult(
            text=search_result.plain_text,
            identifier=search_result.identifier or "",
            sentiment_label=normalized_label,
        )

        end_time = time()
        print(
            f"Processed item {search_result.identifier} in {end_time - start_time:.2f} seconds."
        )

        return result

    results: List[SentimentResult] = []
    print(f"Starting Fietje sentiment analysis on {len(search_results)} items...")
    start_time = time()


    results = []
    for search_result in search_results:
        try:
            result = process_single_text(search_result)
            results.append(result)
        except ValueError as e:
            logging.warning(f"Skipping item {search_result.identifier} due to error: {e}")

    end_time = time()

    print(
        f"\nAnalysis complete. Processed {len(results)} items in {end_time - start_time:.2f} seconds."
    )
    return results


def classify_sentiment_fietje(model, tokenizer, text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Je bent een sentimentanalyse-model. "
                "Classificeer de tekst. Antwoord uitsluitend met een geldige JSON string in het formaat: "
                '{"label": "CATEGORIE"}. '
                "Kies voor CATEGORIE uit: POSITIEF, NEUTRAAL, NEGATIEF."
            ),
        },
        {"role": "user", "content": f'Tekst: "{text}"'},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=40,
            temperature=0.1,
            do_sample=True,
            pad_token_id=pad_token_id,
        )

    # Slice to keep only new tokens
    new_tokens = outputs[0][input_ids.shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    import json
    import re

    match = re.search(r"\{.*?\}", output_text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data.get("label", "NEUTRAAL").upper()
        except:
            pass

    clean = output_text.upper()
    if "POSITIEF" in clean:
        return "POSITIEF"
    if "NEGATIEF" in clean:
        return "NEGATIEF"
    if "NEUTRAAL" in clean:
        return "NEUTRAAL"
    raise ValueError(f"Could not classify sentiment from Fietje output: {output_text}")


def analyze_sentiments_dutch_ollama(
    search_results: List[PlainTextSearchResult],
) -> List[SentimentResult]:
    """
    Performs Dutch sentiment analysis using a local Ollama model,
    utilizing multithreading for increased throughput.
    """

    # Configuration (aligned with your classification script)
    MODEL_NAME = "llama3:8b"
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
        '{"label": "POSITIEF"}. '
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
            {"role": "user", "content": f"Text: {text_content}"},
        ]

        label = SentimentLabel.NEUTRAL

        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=messages,
                options={"temperature": 0.0},
                format="json",  # Force JSON output from Ollama
            )

            content = response["message"]["content"]
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
        print(
            f"Processed item {item.identifier} in {end_time - start_time:.2f} seconds."
        )

        return SentimentResult(
            text=item.plain_text,
            identifier=item.identifier or "",
            sentiment_label=label,
        )

    results: List[SentimentResult] = []
    print(
        f"Starting Ollama sentiment analysis on {len(search_results)} items with {NUM_WORKERS_OLLAMA} workers..."
    )

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
                print(f"Processed {completed_count}/{len(search_results)}", end="\r")

    print(f"\nAnalysis complete. Processed {len(results)} items.")
    return results
