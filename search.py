import sys
from pathlib import Path
import requests
from datetime import datetime
import xml.etree.ElementTree as ET
import os
import json
import time

from delpher_types import SearchQuery, SearchResult

BASE_URL = "https://jsru.kb.nl/sru/sru"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

namespace_map = {
    "ddd": "http://www.kb.nl/ddd",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcx": "http://example.com/dcx",  # replace with real URI if needed
    "srw": "http://www.loc.gov/zing/srw/",
}


def fetch_search_results(search_query: SearchQuery) -> bytes:
    query_params = build_query_params(search_query)
    response = requests.get(BASE_URL, params=query_params)
    response.raise_for_status()
    return response.content


def build_query_params(search_query: SearchQuery) -> dict[str, str]:
    def build_search_query_text(sq: SearchQuery) -> str:
        search_term = sq.search_text
        start_date = sq.start_date
        end_date = sq.end_date

        date_range_text = f'(date within "{start_date} ' f'{end_date}")'
        return f"{search_term} AND {date_range_text}"

    search_query_params = {
        "version": "1.2",
        "maximumRecords": str(search_query.maximum_records),
        "operation": "searchRetrieve",
        "startRecord": str(search_query.start_record),
        "recordSchema": "ddd",
        "x-collection": search_query.collection,
        "query": build_search_query_text(search_query),
    }

    return search_query_params


def extract_total_search_results(root: ET.Element) -> int:
    tagname = "srw:numberOfRecords"
    total_search_results = root.find(tagname, {"srw": "http://www.loc.gov/zing/srw/"})
    if total_search_results is None:
        raise ValueError(f"XML does not contain tag {tagname}")
    return int(total_search_results.text)  # type: ignore


def parse_search_results(xml_bytes) -> list[SearchResult]:
    root = ET.fromstring(xml_bytes)
    search_results: list[SearchResult] = []

    for record_elem in root.findall(".//srw:record", namespaces=namespace_map):
        record_data = record_elem.find("srw:recordData", namespaces=namespace_map)
        if record_data is None:
            continue
        title = record_data.findtext("dc:title", namespaces=namespace_map)
        pub_date = record_data.findtext("dc:date", namespaces=namespace_map)
        paper_title = record_data.findtext("ddd:papertitle", namespaces=namespace_map)
        spatial_creation = record_data.findtext(
            "ddd:spatialCreation", namespaces=namespace_map
        )
        identifier = record_data.findtext("dc:identifier", namespaces=namespace_map)
        ocr_url = identifier  # In the response, the identifier is also the OCR URL

        missing_fields = []
        if not title:
            missing_fields.append("title")
        if not pub_date:
            missing_fields.append("pub_date")
        if not ocr_url:
            missing_fields.append("ocr_url")
        if not paper_title:
            missing_fields.append("paper_title")
        if not spatial_creation:
            missing_fields.append("spatial_creation")
        if not identifier:
            missing_fields.append("identifier")

        if missing_fields:
            with open("missing_fields.log", "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"identifier: {identifier}, missing: {', '.join(missing_fields)}\n"
                )
            continue

        if pub_date is not None:
            publication_date = datetime.strptime(pub_date, "%Y/%m/%d %H:%M:%S").date()
        else:
            publication_date = None

        search_result = SearchResult(
            publication_date=publication_date,
            title=title,
            ocr_url=ocr_url,
            paper_title=paper_title,
            spatial_creation=spatial_creation,
            identifier=identifier,
        )
        search_results.append(search_result)

    return search_results


def fetch_paginated_search_results_stream(
    search_query: SearchQuery, total_search_results: int, offset_file: Path
):
    if offset_file.exists():
        try:
            offset = int(offset_file.read_text().strip())
            print(f"Resuming from offset {offset}")
        except Exception:
            offset = search_query.start_record
    else:
        offset = search_query.start_record

    while offset <= total_search_results:
        print(f"Fetching results from offset {offset}...")
        search_query.start_record = offset
        try:
            xml_bytes = fetch_search_results(search_query)
        except requests.HTTPError as e:
            print(
                f"HTTP error occurred while fetching search results from offset {offset}: {e}"
            )
            raise e
        try:
            next_results = parse_search_results(xml_bytes)
        except ET.ParseError as e:
            print(f"XML parsing error at results offset {offset}: {e}")
            raise e
        for result in next_results:
            yield result
        # Save offset after each batch
        offset_file.write_text(str(offset))
        offset += search_query.maximum_records
        time.sleep(5)


def write_results_stream(search_results_iter, output_file: Path):
    with open(output_file, "a", encoding="utf-8") as f:
        for result in search_results_iter:
            result_dict = {
                **result.__dict__,
                "publication_date": result.publication_date.isoformat(),
            }
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            f.flush()
    print(f"Search results written to {output_file}")


def write_to_json_file(search_results):
    output_file = Path("search_results.json")

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    **result.__dict__,
                    "publication_date": result.publication_date.isoformat(),
                }
                for result in search_results
            ],
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(f"Search results written to {output_file}")


def main():
    query_text = "eten"

    search_query = SearchQuery(
        search_text=query_text,
        # start_date="1940-01-01",
        start_date="2000-01-01",
        # end_date="1945-12-31",
        end_date="2026-01-01",
        maximum_records=10,
        start_record=1,
        collection="DDD_artikel",
    )

    try:
        xml_bytes = fetch_search_results(search_query)
    except requests.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        exit(1)

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        exit(1)

    total_search_results = extract_total_search_results(root)
    print(f"Total search results: {total_search_results}")
    if total_search_results < 1:
        sys.exit(0)

    # print(f"Continue fetching {total_search_results} docs? [Y/n]")

    # response = input()
    # if response.lower() not in ["y", ""]:
    # print("Exiting without fetching documents.")
    # sys.exit(0)

    offset_file = Path(f"offsets/{search_query.search_text}_{search_query.start_date}_{search_query.end_date}_offset.txt")

    results_stream = fetch_paginated_search_results_stream(
        search_query, total_search_results, offset_file
    )

    # {search_query.start_date}_{search_query.end_date}_offset.txt")
    search_results_file_name = Path(f"search_results/{search_query.search_text}_{search_query.start_date}_{search_query.end_date}_search_results.ndjson")
    write_results_stream(results_stream, output_file=search_results_file_name)


if __name__ == "__main__":
    main()
