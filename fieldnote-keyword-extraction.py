from dataclasses import dataclass
from keybert import KeyBERT
import os
from typing import TypeAlias

kw_model = KeyBERT()


@dataclass(frozen=True)
class DocumentKeywords:
    text: str
    keywords: list[tuple[str, float]]


def read_markdown_files(directory: str) -> list[str]:
    """
    Import markdown files from a specified directory.
    Args:
        directory (str): Path to the directory containing markdown files.

    Returns:
        list[str]: List of contents of markdown files.
    """
    fieldnotes: list[str] = []

    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                fieldnotes.append(content)

    return fieldnotes


def extract_keywords(texts: list[str]) -> list[DocumentKeywords]:
    """
    Extract keywords from a list of texts using KeyBERT.
    Args:
        texts (list[str]): List of texts to extract keywords from.

    Returns:
        list[DocumentKeywords]: List of DocumentKeywords dataclasses containing the text and its keywords.
    """
    results: list[DocumentKeywords] = []

    for text in texts:
        keywords: list[tuple[str, float]] = kw_model.extract_keywords(text, stop_words=None, top_n=10)  # type: ignore
        results.append(DocumentKeywords(text=text, keywords=keywords))

    return results


if __name__ == "__main__":
    FIELDNOTES_DIR = os.getenv("FIELDNOTES_DIR")

    if FIELDNOTES_DIR is None:
        raise ValueError("FIELDNOTES_DIR environment variable is not set.")

    fieldnotes = read_markdown_files(FIELDNOTES_DIR)
    results = extract_keywords(fieldnotes)
    for result in results:
        print(result.text[:500])
        print(result.keywords)
        print("\n\n")
