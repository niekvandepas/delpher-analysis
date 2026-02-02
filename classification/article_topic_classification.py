#!/usr/bin/env python3

import json
import os
from pathlib import Path
import time
from transformers import pipeline  # type: ignore
import ollama
from delpher_types import DistilbertClassificationResult, OllamaClassificationResult, PlainTextSearchResult, TranslatedSearchResult  # type: ignore

classifier = pipeline(
    "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1
)


def classify_articles_dutch(
    texts: list[PlainTextSearchResult],
) -> list[OllamaClassificationResult]:
    MODEL_NAME = os.environ.get("OLLAMA_CLASSIFICATION_MODEL")
    if not MODEL_NAME:
        raise ValueError("Environment variable OLLAMA_CLASSIFICATION_MODEL not set.")

    EXAMPLE_TEXT_1 = "Maastricht – De winst van restaurants loopt terug, maar de populariteit van de buitenlandse keuken neemt in Nederland alleen maar toe. In de periode 1995–2002 is het aantal restaurants met een buitenlandse keuken 19 procent gestegen. De nettowinst voor restaurants is in 10 jaar afgenomen van 11 naar 9 procent in 2001. Dat concludeert het bedrijfschap Horeca en Catering in het sectorrapport Eten in de Nederlandse horeca 2002, dat gisteren werd gepresenteerd tijdens de horecavakbeurs BBB/European Fine Food Fair in Maastricht. Net als de snackbar op de hoek, lijdt ook het restaurant onder de ‘amateurskoks’ die zich – geholpen door de detailhandel – thuis uitsloven. De consument laat vooral de Nederlands-Franse keuken links liggen. Het aantal restaurants met deze keuken is in de afgelopen zeven jaar met 2 procent gedaald. De restaurantsector heeft zijn aandeel in de totale eetomzet binnen de horeca zien afnemen van 49 procent in 1991 naar 44 procent in 2001."
    EXAMPLE_CATEGORY_1 = "Economie"
    EXAMPLE_TEXT_2 = "Kaas van de markt Eten Wie 's zaterdags zijn kaas koopt op een Hollandse markt, sluit aan in een oude traditie. Wat is er nodig? Een kraam, wat kazen en een mes, een handelaar en een klant - net zoals eeuwen geleden. Onze voorouders spendeerden graag geld aan een brok kaas zoals we uit i7e-eeuwse huishoudboekjes en schuldbekentenissen (!) weten. Ook de export van kaas is stokoud. Reeds voor het jaar 1000 moet er handel zijn geweest, maar het eerste 'harde' bewijs is een document uit 1118, waaruit blijkt dat handelaren uit Stavoren op de Rijn tol betaalden, met boter en kaas, Die kaas kwam veelal uit Noord-Holland en werd over de Zuiderzee naar Friesland gebracht. Een andere route liep via Kampen en Deventer (toen een transitostad van betekenis), vanwaar men de kaas verder over de IJssel verscheepte. Een artistiek bewijs van kaashandel is te zien in de dom van Munster: op een wandschildering prijken boeren uit Friese gewesten die boter en kaas als geschenk aan de Duitse bisschop brengen. Het bereik van de Friese handelaren was opvallend groot: boter, kaas, vet, vee en paarden exporteerden ze naar Haarlem en Alkmaar, maar ook naar Vlaanderen, Denemarken en zelfs naar de Baltische staten en Frankrijk. In de 15e eeuw was de kaashandel - naar huidige maatstaven gerekend - van immense omvang: talloze schippers uit Edam en Enkhuizen voeren schepen vol kaas naar verdere be-stemmingen. Zo stak in de herfst van 1439 schipper Vrederik Hillebrantz uit Edam van wal. Hij voer naar Kampen met 15 vaten boter, 3000 pond kaas en 2500 'deyne kasekens'. Vandaag: een selectie Hollandse 'kasekens' uit het vuistje, met boerenbrood, een zoute haring van de kar en een glas bier. Voor ons een middag- of avondmaal, voor een Gouden Eeuwer een 'banketje' of ontbijt! Alma Huisken"
    EXAMPLE_CATEGORY_2 = "Traditie"
    EXAMPLE_TEXT_3 = "Amsterdam – De gemeente heeft aangekondigd dat het stadsarchief de komende maanden gaat digitaliseren. Bezoekers kunnen straks online door oude stadskaarten, foto's en documenten bladeren, waardoor historisch onderzoek eenvoudiger wordt. De inspanning richt zich op het behoud van cultureel erfgoed en het toegankelijk maken van archiefmateriaal voor een breed publiek."
    EXAMPLE_CATEGORY_3 = "Overig"

    CLASSIFICATION_SYSTEM_PROMPT = f"""
    Je bent een tekstclassificatiesysteem. Je taak is om de tekst van de gebruiker te classificeren in één van de volgende categorieën:
    [Duurzaamheid], [Gezondheid], [Economie], [Smaak], [Traditie], [Overig].

    - Kies alleen **één categorie per tekst**.
    - Als de tekst niet duidelijk in Duurzaamheid, Gezondheid, Economie, Smaak of Traditie valt, kies **Overig**.
    - Output uitsluitend de naam van de categorie, zonder extra tekst of leestekens.

    Voorbeelden ter begeleiding:

    Voorbeeldtekst 1:
    {EXAMPLE_TEXT_1}
    Categorie 1:
    {EXAMPLE_CATEGORY_1}

    Voorbeeldtekst 2:
    {EXAMPLE_TEXT_2}
    Categorie 2:
    {EXAMPLE_CATEGORY_2}

    Voorbeeldtekst 3:
    {EXAMPLE_TEXT_3}
    Categorie 3:
    {EXAMPLE_CATEGORY_3}
    """

    def classify_text_with_ollama(client: ollama.Client, text: str) -> str:
        """Sends a classification request to the local Ollama model."""
        messages = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Test Text: {text}\nCategory:"},
        ]
        try:
            response = client.chat(
                model=MODEL_NAME, messages=messages, options={"temperature": 0.0}
            )
            return response["message"]["content"].strip()
        except Exception as e:
            return f"ERROR: {e}"

    results: list[OllamaClassificationResult] = []

    client = ollama.Client()

    total_start_time = time.time()

    for i, original_text in enumerate(texts):
        start_time = time.time()

        category = classify_text_with_ollama(client, original_text.plain_text)
        result = OllamaClassificationResult(
            text=original_text.plain_text, label=category
        )

        end_time = time.time()
        print(
            f"Processed article #{i}: {category} in {end_time - start_time:.2f} seconds"
        )

        results.append(result)

    total_end_time = time.time()
    print(
        f"Processed {len(texts)} articles in {total_end_time - total_start_time} seconds"
    )
    return results


def classify_articles_english(
    english_texts: list[TranslatedSearchResult],
) -> list[DistilbertClassificationResult]:
    candidate_labels = ["sustainability", "health", "economics"]

    results: list[DistilbertClassificationResult] = []
    # https://huggingface.co/typeform/distilbert-base-uncased-mnli/blame/b91e7a74c63c287d22a105a9f050cd26d648879f/config.json
    max_tokens = 512

    for i, search_result in enumerate(english_texts):
        text = search_result.english_translated_text

        truncated_text = " ".join(text.split()[:max_tokens])
        classification = classifier(truncated_text, candidate_labels)
        label = classification["labels"][0]  # type: ignore
        score = classification["scores"][0]  # type: ignore

        results.append(
            DistilbertClassificationResult(
                translated_text=text,
                label=label,  # type: ignore
                score=score,  # type: ignore
            )
        )

        print(f"Processed article #{i}: {label} ({score:.2f})", end="\r")

    # Ensure last line stays visible
    print(f"Processed {len(english_texts)} articles", end="\n")
    return results
