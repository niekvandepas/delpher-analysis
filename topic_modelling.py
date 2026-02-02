from bertopic import BERTopic  # type: ignore
from constants import BERTOPIC_MODEL_PATH
from delpher_types import BertopicResult, OcredSearchResult, TopicInfo
from utils import time_function
from nltk.corpus import stopwords
from umap import UMAP  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import os


def create_bertopic_model(texts: list[str]) -> BERTopic:
    dutch_stopwords = stopwords.words("dutch")
    vectorizer_model = CountVectorizer(stop_words=dutch_stopwords)

    topic_model = BERTopic(language="dutch", vectorizer_model=vectorizer_model)
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


def run_bertopic(texts: list[str]) -> list[BertopicResult]:
    if os.path.exists(BERTOPIC_MODEL_PATH):
        model = BERTopic.load(BERTOPIC_MODEL_PATH)
    else:
        model = create_bertopic_model(texts)
        model.save(BERTOPIC_MODEL_PATH)

    # topics, _ = model.transform(texts)
    topics = model.topics_
    info = model.get_topic_info()

    results: list[BertopicResult] = []

    for text, topic_num in zip(texts, topics):
        if topic_num == -1:
            topic_info = TopicInfo(topic_num=-1, topic_words=[])
            bertopic_result = BertopicResult(text=text, topic_info=topic_info)
            results.append(bertopic_result)
            continue

        topic_words = model.get_topic(topic_num)
        words = [w for w, _ in topic_words] if topic_words else []

        topic_info = TopicInfo(topic_num=topic_num, topic_words=words)
        bertopic_result = BertopicResult(text=text, topic_info=topic_info)
        results.append(bertopic_result)

    return results
