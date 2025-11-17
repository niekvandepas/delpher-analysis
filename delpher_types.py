from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from pandas import DataFrame


@dataclass
class SearchQuery:
    search_text: str
    start_date: str
    end_date: str
    maximum_records: int
    start_record: int
    collection: str


@dataclass(frozen=True)
class SearchResult:
    publication_date: Optional[date]
    title: Optional[str]
    ocr_url: Optional[str]
    paper_title: Optional[str]
    spatial_creation: Optional[str]
    identifier: Optional[str]


@dataclass(frozen=True)
class OcredSearchResult(SearchResult):
    ocr_xml: str

@dataclass(frozen=True)
class PlainTextSearchResult(OcredSearchResult):
    plain_text: str

@dataclass(frozen=True)
class TranslatedSearchResult(PlainTextSearchResult):
    english_translated_text: str

@dataclass(frozen=True)
class LabeledSearchResult(OcredSearchResult):
    is_about_indonesia: bool
    snippet: str

@dataclass(frozen=True)
class IndoAuthenticityResults():
    co_occurence: DataFrame
    most_similar_words: dict[str, str]


class EmbeddingModel(Enum):
    WORD2VEC = 1
    FASTTEXT = 2
    SENTENCE_TRANSFORMER = 3
