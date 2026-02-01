#!/usr/bin/env python3
from dataclasses import asdict
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from dotenv import load_dotenv

from translation.translation import translate_texts_llama

# ruff: noqa: E402 # Ignore import positioning for this file
logging.info("Importing libraries...")

from typing import Any

from classification.article_topic_classification import (
    classify_articles_dutch,
    classify_articles_english,
)
from co_occurence import document_co_occurence
from constants import (
    ARTICLE_TOPIC_CLASSIFICATION_RESULTS_PATH,
    DOTENV_PATH,
    PROJECT_DIR,
)
from data_import import (
    import_search_results,
    import_search_results_ndjson,
    normalize_unicode,
    remove_advertisements,
    strip_xml_tags,
)
from delpher_types import (
    DistilbertClassificationResult,
    EmbeddingModel,
    IndoAuthenticityResults,
    OcredSearchResult,
    OllamaClassificationResult,
    PlainTextSearchResult,
    TranslatedSearchResult,
    enum_serializer,
)
from embeddings import (
    get_most_similar_sentence_transformer_documents,
    get_most_similar_word_embedding_words,
    train_fasttext_model,
    train_sentence_transformer_model,
    train_word2vec_model,
)
from sentiment_analysis.sentiment_analysis import (
    analyze_sentiments_dutch_ollama,
    analyze_sentiments_robbert,
    analyze_sentiments_dutch_fietje,
    analyze_sentiments_english,
)
from topic_modelling import run_bertopic
from utils import time_function


# - Is Dutch cuisine portrayed as boring as compared to Indo and Indonesian cuisines?
# - To what extent is a pragmatic attitude to Dutch cuisine vs a ~culturalist, thick, identity and heritage-focused attitude to Indonesian food visible in the newspaper archives?
def main() -> None:
    logging.info("Loading dotenv")
    load_dotenv(DOTENV_PATH)

    DATA_IMPORT_LIMIT = os.environ.get("DATA_IMPORT_LIMIT")
    if not DATA_IMPORT_LIMIT:
        logging.info(
            "Environment variable DATA_IMPORT_LIMIT not set, loading all data."
        )
        DATA_IMPORT_LIMIT = None
    else:
        logging.info(
            f"Environment variable DATA_IMPORT_LIMIT set to {DATA_IMPORT_LIMIT}, limiting imported data."
        )
        DATA_IMPORT_LIMIT = int(DATA_IMPORT_LIMIT)

    logging.info("Importing search results")
    search_results = import_search_results_ndjson(
        limit=DATA_IMPORT_LIMIT,
        path="data/dutch_and_food_terms_query_with_plain_texts.ndjson",
    )
    logging.info("Importing search results done")

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

    plain_text_search_results_without_ads = remove_advertisements(
        plain_text_search_results
    )

    # sentiment_analysis(plain_text_search_results_without_ads)
    classification_results = technocratism(plain_text_search_results_without_ads)

    with open(ARTICLE_TOPIC_CLASSIFICATION_RESULTS_PATH, "r") as f:
        classification_results_dict = [asdict(r) for r in classification_results]
        json.dump(
            classification_results_dict,
            f,
            ensure_ascii=False,
            indent=2,
            default=enum_serializer,
        )


def sentiment_analysis(texts: list[PlainTextSearchResult]):
    logging.info("Analyzing sentiments with RobBERT")
    sentiment_results_robbert = analyze_sentiments_robbert(texts)
    logging.info("Analyzing sentiments with fietje")
    sentiment_results_fietje = analyze_sentiments_dutch_fietje(texts)
    logging.info("Analyzing sentiments with ollama")
    sentiment_results_ollama = analyze_sentiments_dutch_ollama(texts)

    robbert_dict = [asdict(r) for r in sentiment_results_robbert]
    fietje_dict = [asdict(r) for r in sentiment_results_fietje]
    ollama_dict = [asdict(r) for r in sentiment_results_ollama]

    robbert_path = "output/robbert_sentiment.json"
    fietje_path = "output/fietje_sentiment.json"
    ollama_path = "output/ollama_sentiment.json"

    logging.info("Writing sentiment analysis results to JSON files in output/")
    with open(robbert_path, "w", encoding="utf-8") as f:
        json.dump(
            robbert_dict, f, ensure_ascii=False, indent=2, default=enum_serializer
        )

    with open(fietje_path, "w", encoding="utf-8") as f:
        json.dump(fietje_dict, f, ensure_ascii=False, indent=2, default=enum_serializer)

    with open(ollama_path, "w", encoding="utf-8") as f:
        json.dump(ollama_dict, f, ensure_ascii=False, indent=2, default=enum_serializer)

    ...


def assign_document_topics(
    data: list[OcredSearchResult],
) -> dict[OcredSearchResult, Any]:
    texts = [strip_xml_tags(t.ocr_xml) for t in data]

    topics = run_bertopic(texts)
    # TODO this doesn't return the right thing yet
    return topics


def indifference(texts: list[TranslatedSearchResult]) -> Any:
    print("Running indifference tests...")
    sentiment_results = analyze_sentiments_english(texts)

    return sentiment_results


def technocratism(
    texts: list[PlainTextSearchResult],
) -> list[DistilbertClassificationResult] | list[OllamaClassificationResult]:
    print("Running technocratism tests...")
    # technocratic tendencies, which is explored through topic modeling or classification to test if most articles focus on sustainability, health, and economics;

    # return classify_articles_english(translated_texts)
    return classify_articles_dutch(texts)


def semantic_search(texts: list[str], query_words: list[str], model: str) -> Any:
    tokenized_texts = [text.split() for text in texts]

    if model.lower() == "fasttext":
        trained_model = train_fasttext_model(tokenized_texts)
        return get_most_similar_word_embedding_words(trained_model, query_words)
    elif model.lower() == "word2vec":
        trained_model = train_word2vec_model(tokenized_texts)
        return get_most_similar_word_embedding_words(trained_model, query_words)
    elif model.lower() == "sentence-transformers":
        trained_model, corpus_embeddings = time_function(
            train_sentence_transformer_model, texts
        )
        return get_most_similar_sentence_transformer_documents(
            trained_model, corpus_embeddings, texts, query_words
        )
    else:
        raise ValueError(f"Unknown model type: {model}")


def indo_authenticity(
    docs: list[str], embedding_model: EmbeddingModel
) -> IndoAuthenticityResults:
    print("Running Indo-authenticity tests...")
    terms = ["authentisch", "traditioneel", "erfgoed", "modern"]
    pattern = "(indo|indones\w*|rijsttafel|sumatra|java*)"
    co_occurence = document_co_occurence(docs, terms, pattern)

    # TODO this should be a good list
    query_words = [
        "Indonesië",
        "Indonesiërs",
        "Indisch",
        "Indonesisch",
        "rijsttafel",
        "nasi",
        "sate",
        "gado gado",
        "sambal",
        "tempeh",
        "toko",
        "loempia",
        "bami",
        "ayam",
        "krupuk",
        "perkedel",
        "serundeng",
        "soto",
        "rendang",
        "sambal goreng",
        "ayam goreng",
        "bakmi",
        "bakso",
        "es cendol",
        "es teler",
    ]

    most_similar_words = get_most_similar_words(embedding_model, docs, query_words)
    return IndoAuthenticityResults(
        co_occurence=co_occurence, most_similar_words=most_similar_words
    )


def get_most_similar_words(
    embedding_model: EmbeddingModel, texts: list[str], query_words: list[str]
):
    tokenized_texts = [text.split() for text in texts]

    if embedding_model == EmbeddingModel.FASTTEXT:
        trained_model = train_fasttext_model(tokenized_texts)
        return get_most_similar_word_embedding_words(trained_model, query_words)
    elif embedding_model == EmbeddingModel.WORD2VEC:
        trained_model = train_word2vec_model(tokenized_texts)
        return get_most_similar_word_embedding_words(trained_model, query_words)
    elif embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
        raise ValueError(
            "Sentence transformers cannot be used to get most similar words."
        )


if __name__ == "__main__":
    main()
