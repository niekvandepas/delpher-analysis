from bertopic import BERTopic  # type: ignore
from constants import BERTOPIC_MODEL_PATH
from data_import import import_plain_texts
from delpher_types import OcredSearchResult
from utils import time_function
from umap import UMAP  # type: ignore
from sklearn.datasets import fetch_20newsgroups  # type: ignore
import os

def create_bertopic_model(texts: list[str]) -> BERTopic:
    topic_model = BERTopic(language="dutch")
    topics, probs = time_function(topic_model.fit_transform, texts)
    return topic_model


def get_bertopic_topics_dict(bertopic_model: "BERTopic", topics: list[int]) -> dict[int, list[tuple[str, float]]]:  # type: ignore
    """
    Returns a dict representation of the selected BERTopic topics.
    Keys are topic numbers, values are lists of (word, weight) tuples.
    """
    topic_dict = {}

    for topic_num in set(topics):
        topic_dict[topic_num] = bertopic_model.get_topic(topic_num)

    return topic_dict


# TODO this function SHOULD return a docs->topics dict
def run_bertopic(texts: list[str]) -> dict[int, list[tuple[str, float]]]:
    if os.path.exists(BERTOPIC_MODEL_PATH):
        model = BERTopic.load(BERTOPIC_MODEL_PATH)
    else:
        model = create_bertopic_model(texts)
        model.save(BERTOPIC_MODEL_PATH)
    # topics, _ = model.transform(texts)
    topics = model.get_topics()

    results: list[tuple[str, int, list[str]]] = []

    for text, topic_num in zip(texts, topics):
        if topic_num == -1:
            results.append((text, -1, []))
            continue

        topic_words = model.get_topic(topic_num)
        words = [w for w, _ in topic_words] if topic_words else []
        results.append((text, topic_num, words))

    return results
