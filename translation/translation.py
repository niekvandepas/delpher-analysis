from transformers import TranslationPipeline, pipeline
import os
import json
import ollama
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import asdict
from typing import Optional


from delpher_types import PlainTextSearchResult, TranslatedSearchResult
from utils import truncate_text

def translate_texts_helsinki(search_results: list[PlainTextSearchResult]) -> list[TranslatedSearchResult]:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-nl-en", device="cpu")
    search_results_len = len(search_results)

    import time
    start_time = time.time()
    translated_texts: list[TranslatedSearchResult] = []
    for i, t in enumerate(search_results):
        print(f"translating text {i}/{search_results_len}")
        translated_text = translate_text(t.plain_text, translator)

        translated_texts.append(
            TranslatedSearchResult(
                publication_date=t.publication_date,
                title=t.title,
                ocr_url=t.ocr_url,
                paper_title=t.paper_title,
                spatial_creation=t.spatial_creation,
                identifier=t.identifier,
                ocr_xml=t.ocr_xml,
                plain_text=t.plain_text,
                english_translated_text=translated_text,
            )
        )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Translation took {duration} seconds")

    return translated_texts

def translate_text(t: str, translator: TranslationPipeline) -> str:
    # TODO check these params
    out = translator(t, max_length=512, truncation=True)
    return out[0]["translation_text"]














































MODEL_NAME = 'llama3:8b'
# Specific System Prompt for Translation
TRANSLATION_SYSTEM_PROMPT = """
You are a professional translator. Your task is to translate the provided Dutch text into English.
Output ONLY the English translation. Do not include any introductory text, notes, or explanations.
If the text is empty or unintelligible, output an empty string.
"""

def translate_single_text_with_ollama(client: ollama.Client, text: str) -> str:
    """Sends a translation request to the local Ollama model."""
    messages = [
        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]
    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": 0.0} # Deterministic output
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"ERROR: {e}"

def process_single_translation(
    item: PlainTextSearchResult,
    client: ollama.Client,
    file_lock: Lock,
    jsonl_output_path: str, # The data storage (Full results)
    processed_ids: set
) -> Optional[dict]:
    """Processes a single translation task and saves immediately."""

    # Skip if already processed
    if item.identifier in processed_ids:
        return None

    start_time = time()

    # Truncate to avoid context window overflow.
    # Llama 3 8B has an 8k context window. 4000 chars is roughly 1000 tokens,
    # leaving plenty of room for the English output.
    text_to_translate = truncate_text(item.plain_text, 4000)

    translation = translate_single_text_with_ollama(client, text_to_translate)

    translated_result = TranslatedSearchResult(
        publication_date=item.publication_date,
        title=item.title,
        ocr_url=item.ocr_url,
        paper_title=item.paper_title,
        spatial_creation=item.spatial_creation,
        identifier=item.identifier,
        ocr_xml=item.ocr_xml,
        plain_text=item.plain_text,
        english_translated_text=translation
    )

    end_time = time()

    with file_lock:
        with open(jsonl_output_path, 'a', encoding='utf-8') as jsonl_file:
            # asdict converts the dataclass to a dictionary
            json_line = json.dumps(asdict(translated_result), ensure_ascii=False)
            jsonl_file.write(json_line + "\n")

        processed_ids.add(item.identifier)

    return {
        "result_object": translated_result,
        "time": end_time - start_time
    }

def translate_texts_llama(search_results: list[PlainTextSearchResult]) -> list[TranslatedSearchResult]:
    """
    Translates texts and saves continuously to a JSONL file.
    """

    # 1. Setup Paths
    jsonl_output_path = os.environ.get("TRANSLATION_JSONL_PATH")

    if not jsonl_output_path:
        raise ValueError("TRANSLATION_JSONL_PATH environment variable is not set.")

    num_workers = os.environ.get("NUM_WORKERS", "4")

    os.makedirs(os.path.dirname(jsonl_output_path), exist_ok=True)

    processed_ids = set()

    # If JSONL exists, read already-processed texts from it to prevent re-doing work
    if os.path.exists(jsonl_output_path):
        print("Checking existing JSONL file for progress...")
        with open(jsonl_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'identifier' in data:
                        processed_ids.add(data['identifier'])
                except json.JSONDecodeError:
                    continue

    texts_to_process = [sr for sr in search_results if sr.identifier not in processed_ids]

    print(f"Total texts: {len(search_results)}")
    print(f"Already translated: {len(processed_ids)}")
    print(f"Remaining: {len(texts_to_process)}")
    print(f"Starting translation with {num_workers} workers...")

    final_results = []
    file_lock = Lock()
    global_start_time = time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
        futures = {
            executor.submit(
                process_single_translation,
                item,
                ollama.Client(),
                file_lock,
                jsonl_output_path, # Pass the new path
                processed_ids
            ): item
            for item in texts_to_process
        }

        for future in as_completed(futures):
            data = future.result()
            if data is not None:
                final_results.append(data["result_object"])
                completed_count += 1

                # Progress logging...
                elapsed = time() - global_start_time
                print(f"✓ Translated {completed_count}/{len(texts_to_process)} | Time: {data['time']:.2f}s")

    total_time = time() - global_start_time
    print(f"\n✅ Completed in {total_time:.2f}s")

    return final_results
