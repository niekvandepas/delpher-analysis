import os
import json
from pathlib import Path
from dotenv import load_dotenv
import requests
import time

from constants import PROJECT_DIR
from delpher_types import OcredSearchResult, SearchQuery, SearchResult

BASE_URL = "https://jsru.kb.nl/sru/sru"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_QUERY_USED = SearchQuery(
    search_text="dutch_and_food_terms_query",
    start_date="2000-01-01",
    end_date="2026-01-01",
    maximum_records=10,
    start_record=1,
    collection="DDD_artikel",
)

JSON_FILE_PATH = Path(
    f"search_results/{SEARCH_QUERY_USED.search_text}_{SEARCH_QUERY_USED.start_date}_{SEARCH_QUERY_USED.end_date}_search_results.ndjson"
)


def read_ndjson(path: Path) -> list[SearchResult]:
    from datetime import date

    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert publication_date to a date object if it's a string
                if isinstance(data.get("publication_date"), str):
                    data["publication_date"] = date.fromisoformat(
                        data["publication_date"]
                    )
                results.append(SearchResult(**data))
    return results


def fetch_result_xml(search_result: SearchResult) -> str:
    if not search_result.ocr_url:
        raise ValueError("Search result does not contain an OCR URL.")
    try:
      response = requests.get(search_result.ocr_url)
      response.raise_for_status()
      return response.text
    except:
        print("Request timed out. Retrying in 5 seconds...")
        time.sleep(5)
        fetch_result_xml(search_result)


def fetch_and_save_result_texts(
    search_results: list[SearchResult], data_dir_name: str, timeout_between_requests=3.0
) -> None:
    dotenv_path = os.path.join(PROJECT_DIR, '.env')
    load_dotenv(dotenv_path)

    DATA_DIR_PATH = os.environ.get("DATA_DIR_PATH")
    if not DATA_DIR_PATH:
        raise ValueError("DATA_DIR_PATH not set in .env file.")

    for i, result in enumerate(search_results):
        print(
            f"Processing record {i + 1}/{len(search_results)} ({(i + 1) / len(search_results) * 100:.2f}%)"
        )
        if result.identifier is None:
            raise ValueError("Search result identifier is None.")
        safe_identifier = result.identifier.replace("/", "-").replace(":", "-")  # type: ignore
        output_dir = os.path.join(DATA_DIR_PATH, data_dir_name)
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{safe_identifier}.json")

        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"File {file_path} already exists, skipping.")
            continue

        print(f"Processing result with identifier: {result.identifier}")

        xml = fetch_result_xml(result)
        ocred_search_result = OcredSearchResult(
            publication_date=result.publication_date,
            title=result.title,
            ocr_url=result.ocr_url,
            paper_title=result.paper_title,
            spatial_creation=result.spatial_creation,
            identifier=result.identifier,
            ocr_xml=xml,
        )

        # Convert publication_date to isoformat for JSON serialization
        result_dict = {
            **ocred_search_result.__dict__,
            "publication_date": (
                ocred_search_result.publication_date.isoformat()
                if ocred_search_result.publication_date is not None
                else None
            ),
        }
        with open(file_path, "w", encoding="utf-8") as jf:
            json.dump(result_dict, jf, ensure_ascii=False, indent=2)

        time.sleep(timeout_between_requests)


if __name__ == "__main__":
    if not os.path.exists(JSON_FILE_PATH):
        print(f"File {JSON_FILE_PATH} does not exist.")
        exit(1)
    else:
        search_results = read_ndjson(JSON_FILE_PATH)
        print(f"Read {len(search_results)} records from {JSON_FILE_PATH}.")

    # There is a risk of duplicates from stopping and restarting the crawler, so
    deduped_search_results = list(set(search_results))
    fetch_and_save_result_texts(
        search_results,
        data_dir_name=SEARCH_QUERY_USED.search_text,
        timeout_between_requests=0,
    )
