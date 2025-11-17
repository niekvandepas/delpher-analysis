from transformers import TranslationPipeline, pipeline

from delpher_types import PlainTextSearchResult, TranslatedSearchResult

def translate_texts(search_results: list[PlainTextSearchResult]) -> list[TranslatedSearchResult]:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-nl-en", device="cpu")
    search_results_len = len(search_results)

    import time
    start_time = time.time()
    translated_texts: list[TranslatedSearchResult] = []
    for i, t in enumerate(search_results):
        print(f"translating text {i}/{search_results_len}")
        translated_text = translate_text(t.plain_text, translator)

        translated_texts.append(
            TranslatedSearchResult(
                publication_date=t.publication_date,
                title=t.title,
                ocr_url=t.ocr_url,
                paper_title=t.paper_title,
                spatial_creation=t.spatial_creation,
                identifier=t.identifier,
                ocr_xml=t.ocr_xml,
                plain_text=t.plain_text,
                english_translated_text=translated_text,
            )
        )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Translation took {duration} seconds")

    return translated_texts


def translate_text(t: str, translator: TranslationPipeline) -> str:
    # TODO check these params
    out = translator(t, max_length=512, truncation=True)
    return out[0]["translation_text"]
