from dataclasses import asdict, replace
from data_import import import_search_results, strip_xml_tags
from delpher_types import LabeledSearchResult, OcredSearchResult

import csv
import re

def filter_texts_by_keyword(search_results: list[OcredSearchResult], keyword_regex: str):
    filtered_texts = []
    pattern = re.compile(keyword_regex, re.IGNORECASE)

    for search_result in search_results:
        plain_text = strip_xml_tags(search_result.ocr_xml)
        if pattern.search(plain_text):
            filtered_texts.append(search_result)

    return filtered_texts

def label(search_results: list[OcredSearchResult]) -> list[LabeledSearchResult]:
    label_results = []

    for search_result in search_results:
        snippet = strip_xml_tags(search_result.ocr_xml)[:2000]
        print("------------------------------------------------------------------------------------------", "\n\n", snippet, "\n")
        label = input("Indonesian food/culture? (y/n) ")

        is_about_indonesia = True if label.lower() == "y" else False

        labeled_result = LabeledSearchResult(
            publication_date=search_result.publication_date,
            title=search_result.title,
            ocr_url=search_result.ocr_url,
            paper_title=search_result.paper_title,
            spatial_creation=search_result.spatial_creation,
            identifier=search_result.identifier,
            ocr_xml=search_result.ocr_xml,
            is_about_indonesia=is_about_indonesia,
            snippet=snippet
        )

        label_results.append(labeled_result)

    return label_results

if __name__ == "__main__":
    search_results = import_search_results()

    filtered_results = filter_texts_by_keyword(search_results, "(Indones|rijsttafel|Javaanse|Sumatra)")[:200]

    label_results = label(filtered_results)

    labeled_dicts = [asdict(r) for r in label_results]

    with open("labeled_articles.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=labeled_dicts[0].keys())
        writer.writeheader()
        writer.writerows(labeled_dicts)
