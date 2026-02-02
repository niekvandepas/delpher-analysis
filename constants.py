RANDOM_SEED = 42

REGIONAL_DISHES_LIST = [
    "Alkmaars kaasdragers Gilde",
    "Babi Pangang",
    "Boerenkaas maken",
    "Cultuur van het Brabants worstenbroodje",
    "Fijndistillatie van genever en likeur in Amsterdam",
    "Groninger eierbaltraditie",
    "Haringuitreiking in Niekerk",
    "Het bakken van poffertjes",
    "Het stoken van Limburgse stroop",
    "Hollandse gebakskraam",
    "Indische rijsttafel traditie",
    "Kroketten maken met de hand",
    "Olieslaan op ambachtelijke wijze",
    "Pannenkoeken bakken in ijzeren pannen",
    "Siroopwafels bakken",
    "Spekkedikken bakken",
    "Traditie van de Tielsche kermiskoek",
    "Wecken in Roden",
    "Worst maken",
    "Zwolse balletjes",
]

import os

PROJECT_DIR = f"{os.path.dirname(__file__)}"
PLAIN_TEXTS_FILE_PATH = f"{PROJECT_DIR}/data/plain_texts.json"
TRANSLATED_TEXTS_FILE_PATH = f"{PROJECT_DIR}/data/translated_texts.json"
SEARCH_RESULTS_WITH_PLAIN_TEXTS_FILE_PATH = (
    f"{PROJECT_DIR}/data/search_results_with_plain_texts.json"
)
DOTENV_PATH = f"{PROJECT_DIR}/.env"

DUTCH_WORDS_FILE_PATH = f"{PROJECT_DIR}/data/dutch_words.txt"
ENGLISH_WORDS_FILE_PATH = f"{PROJECT_DIR}/data/english_words.txt"

FIETJE_SENTIMENT_RESULTS_PATH = f"{PROJECT_DIR}/output/fietje_sentiment.json"
ROBBERT_SENTIMENT_RESULTS_PATH = f"{PROJECT_DIR}/output/robbert_sentiment.json"
OLLAMA_SENTIMENT_RESULTS_PATH = f"{PROJECT_DIR}/output/ollama_sentiment.json"

ARTICLE_TOPIC_CLASSIFICATION_RESULTS_PATH = f"{PROJECT_DIR}/output/article_topics.json"
BERTOPIC_MODEL_PATH = f"{PROJECT_DIR}/models/bertopic_model"
BERTOPIC_RESULTS_PATH = f"{PROJECT_DIR}/output/bertopic_results.json"
ANNOTATED_DATA_DIR = f"{PROJECT_DIR}/annotated/"
