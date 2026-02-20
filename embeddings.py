from enum import Enum
from gensim.models import FastText, Word2Vec
from gensim.models import Word2Vec
import nltk
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from sklearn.metrics.pairwise import cosine_similarity

from constants import RANDOM_SEED


def train_fasttext_model(tokenized_texts: list[list[str]]) -> FastText:
    model = FastText(seed=RANDOM_SEED, min_count=20)
    model.build_vocab(tokenized_texts)
    model.train(
        epochs=10, total_examples=model.corpus_count, corpus_iterable=tokenized_texts
    )
    return model


def train_word2vec_model(tokenized_texts: list[list[str]]) -> Word2Vec:
    word2vec_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        seed=RANDOM_SEED,
    )

    return word2vec_model


def train_sentence_transformer_model(
    texts: list[str],
) -> tuple[SentenceTransformer, Tensor]:
    """Loads a pretrained SentenceTransformer and encodes the texts."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(texts, convert_to_tensor=True)
    return model, corpus_embeddings


def load_fasttext_model(model_path: str) -> FastText:
    return FastText.load(model_path)  # type: ignore


def load_word2vec_model(model_path: str) -> Word2Vec:
    return Word2Vec.load(model_path)


def get_most_similar_fasttext_words(
    fasttext_model: FastText, query_words: list[str]
) -> dict[str, str]:
    top_similar_words = {}

    for word in query_words:
        try:
            similar_words = [
                w for w, _ in fasttext_model.wv.similar_by_word(word, topn=20)
            ]
            top_similar_words[word] = similar_words
        except KeyError:
            pass  # Skip words not in vocabulary

    return top_similar_words


def get_most_similar_word_embedding_words(
    word2vec_model: Word2Vec, query_words: list[str]
) -> dict[str, str]:
    """
    Returns the top 20 most similar words for each query word using a trained Word2Vec or FastText model. Words not found in the model's vocabulary are skipped.

    Parameters:
        word2vec_model (Word2Vec): A trained gensim Word2Vec or FastText model.
        query_words (list[str]): A list of words to query for similar words.

    Returns:
        dict[str, list[tuple[str, float]]]: A dictionary mapping each query word to a list
        of its top 20 most similar words and their similarity scores.
    """
    top_similar_words_per_word = {}

    for word in query_words:
        try:
            similar_words = word2vec_model.wv.most_similar(word, topn=20)
            top_similar_words_per_word[word] = similar_words
        except KeyError:
            pass  # Skip words not in vocabulary

    return top_similar_words_per_word


def get_most_similar_sentence_transformer_documents(
    model: SentenceTransformer,
    corpus_embeddings: Tensor,
    corpus_texts: list[str],
    query_words: list[str],
    top_k: int = 5,
) -> dict[str, list[tuple[str, float]]]:
    """
    Returns top_k most similar **documents** for each query.
    """
    query_embeddings = model.encode(query_words, convert_to_tensor=True)
    results = {}

    for i, query in enumerate(query_words):
        scores = util.cos_sim(query_embeddings[i], corpus_embeddings)[0]
        top_results = sorted(
            zip(corpus_texts, scores.tolist()), key=lambda x: x[1], reverse=True
        )[:top_k]
        results[query] = top_results

    return results


def compute_document_embeddings(docs: list[str]) -> np.ndarray:
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    sentence_tokenizer = nltk.data.load("tokenizers/punkt/dutch.pickle")

    doc_embeddings = []

    # Split each document into sentences and then average the vectors, in order to overcome 128-token limit
    # https://link.springer.com/article/10.1007/s11227-025-07414-4
    for doc in docs:
        sentences = sentence_tokenizer.tokenize(doc)

        sentence_embeddings: np.ndarray = model.encode(sentences)  # type: ignore
        doc_avg = np.mean(sentence_embeddings, axis=0)

        doc_embeddings.append(doc_avg)

    return np.array(doc_embeddings)


def compute_document_similarity(docs: list[str]) -> np.ndarray:
    document_embeddings = compute_document_embeddings(docs)
    similarity_matrix = cosine_similarity(document_embeddings)

    return similarity_matrix
