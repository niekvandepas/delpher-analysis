import re
import pandas as pd

def document_co_occurence(docs: list[str], target_terms: list[str], doc_filter_regex: str) -> pd.DataFrame:
    indo_pattern = re.compile(doc_filter_regex, re.IGNORECASE)
    contains_indo = [bool(indo_pattern.search(doc)) for doc in docs]

    n_indo_docs = sum(contains_indo)
    n_non_indo_docs = len(docs) - n_indo_docs

    counts = {term: {"indo_docs": 0, "non_indo_docs": 0} for term in target_terms}

    for doc, is_indo in zip(docs, contains_indo):
        for term in target_terms:
            if re.search(rf'\b{term}\b', doc, re.IGNORECASE):
                if is_indo:
                    counts[term]["indo_docs"] += 1
                else:
                    counts[term]["non_indo_docs"] += 1

    df = pd.DataFrame(counts).T
    df["total_docs"] = len(docs)

    # Percentages (relative frequencies)
    df["indo_doc_percent"] = (df["indo_docs"] / n_indo_docs * 100) if n_indo_docs else 0
    df["non_indo_doc_percent"] = (df["non_indo_docs"] / n_non_indo_docs * 100) if n_non_indo_docs else 0
    df["percent_difference"] = df["indo_doc_percent"] - df["non_indo_doc_percent"]

    return df
