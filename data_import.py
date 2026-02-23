from dataclasses import asdict
import html
import json
import logging
import os
import re
import unicodedata
from pathlib import Path

from constants import DUTCH_WORDS_FILE_PATH, ENGLISH_WORDS_FILE_PATH
from delpher_types import OcredSearchResult, PlainTextSearchResult
from ftfy import fix_text


def normalize_unicode(text: str) -> str:
    # Decode HTML entities (&gt; -> >, etc.)
    unescaped = html.unescape(text)

    # Fix common encoding errors (Ã© -> é, √¢ -> ‘, etc.)
    fixed = fix_text(unescaped)

    # Normalize composed/decomposed Unicode forms and fold ligatures
    normalized = unicodedata.normalize("NFKC", fixed)

    return normalized


def import_search_results(
    path: str, limit: int | None = None
) -> list[OcredSearchResult]:
    with open(path, "r") as f:
        data = json.load(f)

    search_results = []
    for i, item in enumerate(data):
        search_result = OcredSearchResult(
            publication_date=item.get("publication_date"),
            title=item.get("title"),
            ocr_url=item.get("ocr_url"),
            paper_title=item.get("paper_title"),
            spatial_creation=item.get("spatial_creation"),
            identifier=item.get("identifier"),
            ocr_xml=item.get("ocr_xml"),
        )
        search_results.append(search_result)
    return search_results


def import_search_results_ndjson(
    path: str, limit: int | None = None
) -> list[OcredSearchResult]:
    search_results = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            item = json.loads(line)
            search_result = OcredSearchResult(
                publication_date=item.get("publication_date"),
                title=item.get("title"),
                ocr_url=item.get("ocr_url"),
                paper_title=item.get("paper_title"),
                spatial_creation=item.get("spatial_creation"),
                identifier=item.get("identifier"),
                ocr_xml=item.get("ocr_xml"),
            )
            search_results.append(search_result)
    return search_results


def strip_xml_tags(xml: str) -> str:
    # Remove XML tags using regex
    text = re.sub(r"<[^>]+>", " ", xml)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def import_plain_texts(dir: str) -> list[str]:
    texts = []
    results = import_search_results(dir)

    for result in results:
        text = strip_xml_tags(result.ocr_xml)
        texts.append(text)

    return texts


def count_non_dutch_words(input_str: str) -> dict[str, int]:
    """Count non-Dutch words in the given string.

    Args:
      string: A strings.

    Returns:
      A dictionary containing the number of Dutch words in the string, the number of English words in the string, the number of unknown words in the string, and the number of non-words in the string (e.g. punctuation-only or numeric words).
    """
    known_dutch_words = open(DUTCH_WORDS_FILE_PATH).read().splitlines()
    known_english_words = open(ENGLISH_WORDS_FILE_PATH).read().splitlines()
    words = input_str.split()
    total_words = len(words)

    found_dutch_words = 0
    found_english_words = 0
    found_proper_nouns = 0
    found_unknown_words = 0
    found_non_words = 0

    for i, word in enumerate(words):
        remove_chars = '.,!?;:"\'()[]{}<>-–—_/\\|@#$%^&*~`+=\n\r\t"'
        table = str.maketrans("", "", remove_chars)
        clean_word = word.translate(table)

        print(
            f"Processing {i}/{total_words}. Current: {found_dutch_words} Dutch words, {found_proper_nouns} proper nouns, {found_english_words} English words, {found_unknown_words} unknown words, {found_non_words} non-words.",
            end="\r",
        )

        if clean_word.lower() in known_dutch_words:
            found_dutch_words += 1
        # Assume all title-cased words are proper nouns.
        # This is not exactly true, but it's a reasonable heuristic that passes a quick sanity check, and it provides more information than lumping them into 'unknown words'.
        # The reason this clause comes before detecting English words is because the English word list contains a lot of names and proper nouns,
        # Which would otherwise unfairly get lumped into the 'English words' category.
        elif clean_word.istitle():
            found_proper_nouns += 1
        elif clean_word.lower() in known_english_words:
            found_english_words += 1
        elif not clean_word.isnumeric() and not clean_word.isspace():
            found_unknown_words += 1
        else:
            found_non_words += 1

    return {
        "dutch": found_dutch_words,
        "english": found_english_words,
        "proper_nouns": found_proper_nouns,
        "unknown": found_unknown_words,
        "non_words": found_non_words,
    }


def data_dir_to_single_json_file(data_dir: str, out_file_path: str) -> None:
    absolute_paths: list[str] = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Skip metadata files on macOS
            if file.startswith("._"):
                continue
            absolute_paths.append(os.path.join(root, file))

    search_results = []
    total_files = len(absolute_paths)
    errors = 0
    progress = 0

    for path in absolute_paths:
        with open(path, "r", encoding="utf-8") as f:
            try:
                search_result = json.load(f)
            except UnicodeDecodeError:
                print(f"Error decoding file: {path}")
                errors += 1
            progress += 1
            print(
                f"Processing file {progress}/{total_files}. Errors so far: {errors}.",
                end="\r",
            )

        search_results.append(OcredSearchResult(**search_result))

    results = [json.dumps(asdict(search_result)) for search_result in search_results]
    # As .ndjson
    out_value = "\n".join(results)

    with open(out_file_path, "w", encoding="utf-8") as out_file:
        out_file.write(out_value)


def remove_advertisements(
    search_results: list[PlainTextSearchResult],
) -> list[PlainTextSearchResult]:
    search_results_without_ads = []

    for search_result in search_results:
        if not search_result.plain_text.startswith("Advertentie "):
            search_results_without_ads.append(search_result)

    return search_results_without_ads


def preprocess_plain_texts(
    search_results: list[OcredSearchResult],
) -> list[PlainTextSearchResult]:
    plain_text_search_results: list[PlainTextSearchResult] = []

    for i, search_result in enumerate(search_results):
        logging.info(f"Processing search result #{i+1}")
        plain_text = normalize_unicode(strip_xml_tags(search_result.ocr_xml))
        search_result_with_plain_text = PlainTextSearchResult(
            publication_date=search_result.publication_date,
            title=search_result.title,
            ocr_url=search_result.ocr_url,
            paper_title=search_result.paper_title,
            spatial_creation=search_result.spatial_creation,
            identifier=search_result.identifier,
            ocr_xml=search_result.ocr_xml,
            plain_text=plain_text,
        )

        plain_text_search_results.append(search_result_with_plain_text)

    return plain_text_search_results
