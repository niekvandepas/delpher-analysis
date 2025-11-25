# ruff: noqa: E402 # Ignore import positioning for this file
print("Importing libraries...")
from dataclasses import asdict
import json
import os
from typing import Any

from classifier import classify_articles
from constants import REGIONAL_DISHES_LIST
from co_occurence import document_co_occurence
from data_import import count_non_dutch_words, import_search_results, strip_xml_tags
from delpher_types import EmbeddingModel, IndoAuthenticityResults, OcredSearchResult, PlainTextSearchResult, TranslatedSearchResult
from embeddings import (
    get_most_similar_sentence_transformer_documents,
    get_most_similar_word_embedding_words,
    train_fasttext_model,
    train_sentence_transformer_model,
    train_word2vec_model,
)
from sentiment_analysis import analyze_sentiments
from topic_modelling import run_bertopic
from translation import translate_texts
from utils import time_function

SCRIPT_DIR = os.path.dirname(__file__)
PLAIN_TEXTS_FILE_PATH = f"{SCRIPT_DIR}/data/plain_texts.json"
TRANSLATED_TEXTS_FILE_PATH = f"{SCRIPT_DIR}/data/translated_texts.json"
SEARCH_RESULTS_WITH_PLAIN_TEXTS_FILE_PATH = f"{SCRIPT_DIR}/data/search_results_with_plain_texts.json"


def main() -> None:
    print("Importing data...")
    search_results = time_function(import_search_results, SEARCH_RESULTS_WITH_PLAIN_TEXTS_FILE_PATH)

    with open(SEARCH_RESULTS_WITH_PLAIN_TEXTS_FILE_PATH, "r") as f:
        data = json.load(f)
        search_results_with_plain_texts = [PlainTextSearchResult(**d) for d in data]

    translated_search_results: list[TranslatedSearchResult] = translate_texts(search_results_with_plain_texts)

    with open(f"{SCRIPT_DIR}/search_results_with_translations.json", "w") as f:
        json.dump(translated_search_results, f)

    with open(PLAIN_TEXTS_FILE_PATH, "r") as f:
        plain_texts: list[str] = json.load(f)

    # Make list of tuples (length, text)
    length_text_tuples = [(len(text), text) for text in plain_texts]

    # Sort by length
    length_text_tuples.sort(key=lambda x: x[0])

    one_big_string = " ".join(plain_texts)

    word_proportions = count_non_dutch_words(one_big_string)

    print(word_proportions)

    translated_texts = translate_texts(search_results_with_plain_texts)
    # write translated texts to JSON
    with open(TRANSLATED_TEXTS_FILE_PATH, "w") as f:
        json.dump(translated_texts, f)

    # Read translated texts from JSON
    # with open(TRANSLATED_TEXTS_FILE_PATH, "r") as f:
    #     translated_texts = json.load(f)
    # translated_texts = translated_texts[:1000]

    # print("Assigning document topics...")
    topics = assign_document_topics(search_results)
    print(topics)

    indifference_results = indifference(translated_texts)
    print(indifference_results)
    positive_results = [r for r in indifference_results if r["label"] == "positive"]
    neutral_results =  [r for r in indifference_results if r["label"] == "neutral"]
    negative_results = [r for r in indifference_results if r["label"] == "negative"]

    technocratic_results = technocratism(translated_texts[:1000])

    print(technocratic_results)

    regional_dishes_most_similar_fasttext = semantic_search(
        plain_texts, query_words=REGIONAL_DISHES_LIST, model="fasttext"
    )
    print(regional_dishes_most_similar_fasttext)

    regional_dishes_most_similar_sentence_transformers = semantic_search(
        plain_texts, query_words=REGIONAL_DISHES_LIST, model="sentence-transformers"
    )
    print(regional_dishes_most_similar_sentence_transformers)

    indo_authenticity_results = indo_authenticity(plain_texts, embedding_model=EmbeddingModel.FASTTEXT)
    print(indo_authenticity_results)
    results_dict = asdict(indo_authenticity_results)
    # Convert the DataFrame to a dict
    results_dict["co_occurence"] = results_dict["co_occurence"].to_dict()

    with open(f"{SCRIPT_DIR}/indo_authenticity_fasttext.json", "w") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    indo_authenticity_results = indo_authenticity(plain_texts, embedding_model=EmbeddingModel.WORD2VEC)
    print(indo_authenticity_results)
    results_dict = asdict(indo_authenticity_results)

    # Convert the DataFrame to a dict
    results_dict["co_occurence"] = results_dict["co_occurence"].to_dict()

    with open(f"{SCRIPT_DIR}/indo_authenticity_word2vec.json", "w") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

def assign_document_topics(
    data: list[OcredSearchResult],
) -> dict[OcredSearchResult, Any]:
    texts = [strip_xml_tags(t.ocr_xml) for t in data]

    topics = run_bertopic(texts)
    # TODO this doesn't return the right thing yet
    return topics


def indifference(texts: list[TranslatedSearchResult]) -> Any:
    print("Running indifference tests...")
    sentiment_results = analyze_sentiments(texts)

    return sentiment_results


def technocratism(english_texts: list[TranslatedSearchResult]) -> list[dict]:
    print("Running technocratism tests...")
    # technocratic tendencies, which is explored through topic modeling or classification to test if most articles focus on sustainability, health, and economics;

    return classify_articles(english_texts)


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

def indo_authenticity(docs: list[str], embedding_model: EmbeddingModel) -> IndoAuthenticityResults:
    print("Running Indo-authenticity tests...")
    terms = ["authentisch", "traditioneel", "erfgoed", "modern"]
    pattern = '(indo|indones\w*|rijsttafel|sumatra|java*)'
    co_occurence = document_co_occurence(docs, terms, pattern)

    # TODO this should be a good list
    query_words = ["Indonesië", "Indonesiërs", "Indisch", "Indonesisch", "rijsttafel", "nasi", "sate", "gado gado", "sambal", "tempeh", "toko", "loempia", "bami", "ayam", "krupuk", "perkedel", "serundeng", "soto", "rendang", "sambal goreng", "ayam goreng", "bakmi", "bakso", "es cendol", "es teler"]

    most_similar_words = get_most_similar_words(embedding_model, docs, query_words)
    return IndoAuthenticityResults(
        co_occurence=co_occurence,
        most_similar_words=most_similar_words
    )

def get_most_similar_words(embedding_model: EmbeddingModel, texts: list[str], query_words: list[str]):
    tokenized_texts = [text.split() for text in texts]

    if embedding_model == EmbeddingModel.FASTTEXT:
        trained_model = train_fasttext_model(tokenized_texts)
        return get_most_similar_word_embedding_words(trained_model, query_words)
    elif embedding_model == EmbeddingModel.WORD2VEC:
        trained_model = train_word2vec_model(tokenized_texts)
        return get_most_similar_word_embedding_words(trained_model, query_words)
    elif embedding_model == EmbeddingModel.SENTENCE_TRANSFORMER:
        raise ValueError("Sentence transformers cannot be used to get most similar words.")

if __name__ == "__main__":
    main()
