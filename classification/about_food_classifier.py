from random import shuffle
from time import time
import ollama
import json
import os
from dotenv import load_dotenv
from constants import PROJECT_DIR
from data_import import import_search_results_ndjson, normalize_unicode, strip_xml_tags
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from delpher_types import OllamaClassificationResult, PlainTextSearchResult
from utils import truncate_text

MODEL_NAME = "llama3:8b"
CLASSIFICATION_CATEGORIES = "[Is about food], [Is not about food]"
EXAMPLE_TEXT_1 = "'Drugsteams met Fransen Frankrijk en Nederland strijden samen tegen XTC Van onze politieke redactie Den Haag - Er komen speciale Frans-Nederlandse teams om de productie en verkoop van synthetische drugs te bestrijden. President Chirac en premier Kok hebben dat gisteren afgesproken op de tweede dag van Chiracs staatsbezoek. Chirac noemde de samenwerking in de strijd tegen drugs als stc 'bijzonder goed' en 'voorbeeldig'. Hoewel er 'verschil van inzicht' blijft bestaan, werken beide landen 'op positieve wijze' samen, zei hij gisteravond als eregast op het jaarlijkse diner van de Vrienden van Nieuwspoort. Frankrijk en Nederland sloegen in 1995 de handen ineen inzake politie, justitie en douane. De samenwerking kreeg een deuk toen een Franse senator Nederland begin 1996 een 'narcostaat' noemde en Frankrijk de grenscontrole met België weer inroerde om het drugstoerisme de pas af te snijden. Pas sinds 1997 zit er vaart in de samenwerking. De toenmalige ministers van Justitie Sorgdrager en Toubon maakten in dat jaar een einde aan de pesterijtjes over en weer met een symbolische zoen voor de camera's. Met steun van Chirac en Kok werd vanaf dat moment voluit de strijd tegen xtc ingezet. Gisteren zei Chirac overigens dat hij de tijd nog niet rijp acht voor volledige toepassing van het Verdrag van Schengen, dat open grenzen voorschrijft. Premier Kok zat er niet mee. Hij deed de handhaving van de grenscontrole gisteren af als 'symboolpolitiek': \"In de praktijk stelt het weinig voor.\" Het staatsbezoek stond in het teken van verzoening en begrip. Volgens Chirac leidden 'natuurwetten, vrijheid van denken en economische noodzaak' in Nederland tot 'wat wij Fransen vaak voor overdreven laisser-faire houden', maar wat Nederland beschouwt als 'een rechtmatig gedogen'. \"Ikben blij hier te zijn, en ditis absoluut geen diplomatiek zinnetje\", zei Chirac. Op pagina g| Chirac vraagt begrip voor Franse deugden Chirac vraagt begrip voor Franse deugden Vervolg van pagina i Staatsbezoek Maar de liefde kan niet van één kant komen. Chirac vroeg eveneens begrip voor enkele Franse deugden, die vaak versleten worden voor verkalkte tradities. \"U vindt dat het alles 'staatsbemoeienis' is wat klok slaat in Frankrijk. Maar dat is steeds minder het geval. U vindt dat Frankrijk protectionistische trekken vertoont. Maar wij staan in de wereld op de vierde plaats qua in- en uitvoer. Frankrijk wekt vaak verbazing doordat het zo aan zijn taal en cultuur gehecht is. Het tegendeel is waar: wij zitten midden in de mondialisering, maar opening is niet synoniem met eenvormigheid.\" De jonge SP-senator Van Vugt (20) werd gisteren gearresteerd wegens verstoring van de openbare orde: hij protesteerde zonder vergunning in het aangezicht van Chirac tegen het Franse nucleaire beleid met een fluitje en een T-shirt dat herinnerde aan de Franse kernproeven. Andere demonstranten slaagden er in te ontkomen.'"
EXAMPLE_CATEGORY_1 = "Is not about food"
EXAMPLE_TEXT_2 = "Kaas van de markt Eten Wie 's zaterdags zijn kaas koopt op een Hollandse markt, sluit aan in een oude traditie. Wat is er nodig? Een kraam, wat kazen en een mes, een handelaar en een klant - net zoals eeuwen geleden. Onze voorouders spendeerden graag geld aan een brok kaas zoals we uit i7e-eeuwse huishoudboekjes en schuldbekentenissen (!) weten. Ook de export van kaas is stokoud. Reeds voor het jaar 1000 moet er handel zijn geweest, maar het eerste 'harde' bewijs is een document uit 1118, waaruit blijkt dat handelaren uit Stavoren op de Rijn tol betaalden, met boter en kaas, Die kaas kwam veelal uit Noord-Holland en werd over de Zuiderzee naar Friesland gebracht. Een andere route liep via Kampen en Deventer (toen een transitostad van betekenis), vanwaar men de kaas verder over de IJssel verscheepte. Een artistiek bewijs van kaashandel is te zien in de dom van Munster: op een wandschildering prijken boeren uit Friese gewesten die boter en kaas als geschenk aan de Duitse bisschop brengen. Het bereik van de Friese handelaren was opvallend groot: boter, kaas, vet, vee en paarden exporteerden ze naar Haarlem en Alkmaar, maar ook naar Vlaanderen, Denemarken en zelfs naar de Baltische staten en Frankrijk. In de 15e eeuw was de kaashandel - naar huidige maatstaven gerekend - van immense omvang: talloze schippers uit Edam en Enkhuizen voeren schepen vol kaas naar verdere be-stemmingen. Zo stak in de herfst van 1439 schipper Vrederik Hillebrantz uit Edam van wal. Hij voer naar Kampen met 15 vaten boter, 3000 pond kaas en 2500 'deyne kasekens'. Vandaag: een selectie Hollandse 'kasekens' uit het vuistje, met boerenbrood, een zoute haring van de kar en een glas bier. Voor ons een middag- of avondmaal, voor een Gouden Eeuwer een 'banketje' of ontbijt! Alma Huisken"
EXAMPLE_CATEGORY_2 = "Is about food"

CLASSIFICATION_SYSTEM_PROMPT = f"""
You are a text classification system. Your task is to classify the user's text into one of the following two categories: {CLASSIFICATION_CATEGORIES}.
You MUST ONLY output the category name. And nothing else. Do not include any other text, explanation, or punctuation.
Here is a one-shot example to guide your classification:
Example Text 1: {EXAMPLE_TEXT_1}
Category 1: {EXAMPLE_CATEGORY_1}
Here is a one-shot example to guide your classification:
Example Text 2: {EXAMPLE_TEXT_2}
Category 2: {EXAMPLE_CATEGORY_2}
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


def process_single_item(search_result, client, file_lock, out_file_path, processed_ids):
    """Process a single classification task."""
    # Check if already processed
    if search_result.identifier in processed_ids:
        return None

    start_time = time()
    text = truncate_text(normalize_unicode(strip_xml_tags(search_result.ocr_xml)), 2000)

    classification = classify_text_with_ollama(client, text)

    result = {
        "id": search_result.identifier,
        "original_text": text,
        "predicted_category": classification,
    }

    end_time = time()

    # Thread-safe file writing
    with file_lock:
        with open(out_file_path, "a", encoding="utf-8") as out_file:
            out_file.write(f"ID: {search_result.identifier}\n")
            out_file.write(f"Predicted Category: {classification}\n")
            out_file.write(f"Original Text: {text}\n")
            out_file.write("--------------------------------------------------\n")
        processed_ids.add(search_result.identifier)

    return {
        "result": result,
        "time": end_time - start_time,
        "classification": classification,
    }


def classify_about_food(
    texts: list[PlainTextSearchResult],
) -> list[OllamaClassificationResult]:
    classification_data_path = os.environ.get("CLASSIFICATION_DATA_PATH")
    if not classification_data_path:
        raise ValueError(
            "Please set the CLASSIFICATION_DATA_PATH environment variable."
        )

    out_file_path = os.environ.get("OLLAMA_OUT_FILE_PATH")
    if not out_file_path:
        raise ValueError("Please set the OLLAMA_OUT_FILE_PATH environment variable.")

    num_workers = os.environ.get("NUM_WORKERS_ABOUT_FOOD_CLASSIFIER")
    if not num_workers:
        raise ValueError(
            "Please set the NUM_WORKERS_ABOUT_FOOD_CLASSIFIER environment variable."
        )
    num_workers = int(num_workers)

    texts_to_classify = import_search_results_ndjson(classification_data_path)
    shuffle(texts_to_classify)

    # Create output file if it doesn't exist and load already processed IDs
    if not os.path.exists(out_file_path):
        open(out_file_path, "w").close()

    processed_ids = set()
    with open(out_file_path, "r") as f:
        for line in f:
            if line.startswith("ID: "):
                processed_ids.add(line.replace("ID: ", "").strip())

    # Filter out already processed items
    texts_to_process = [
        sr for sr in texts_to_classify if sr.identifier not in processed_ids
    ]

    print(f"Total texts: {len(texts_to_classify)}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining to process: {len(texts_to_process)}")
    print(
        f"Starting classification with {num_workers} parallel workers using {MODEL_NAME}..."
    )

    results = []
    file_lock = Lock()
    global_start_time = time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_item,
                search_result,
                ollama.Client(),  # Each thread gets its own client
                file_lock,
                out_file_path,
                processed_ids,
            ): search_result
            for search_result in texts_to_process
        }

        for future in as_completed(futures):
            result_data = future.result()
            if result_data is not None:
                results.append(result_data["result"])
                completed_count += 1

                total_to_process = len(texts_to_process)
                elapsed_time = time() - global_start_time
                avg_time_per_item = (
                    elapsed_time / completed_count if completed_count > 0 else 0
                )
                remaining_items = total_to_process - completed_count
                estimated_remaining_time = avg_time_per_item * remaining_items

                print(
                    f"✓ Processed {completed_count}/{total_to_process} | "
                    f"Result: {result_data['classification']} | "
                    f"Time: {result_data['time']:.2f}s | "
                    f"Est. remaining: {estimated_remaining_time/60:.1f}min"
                )

    global_end_time = time()
    total_time = global_end_time - global_start_time

    print(f"\n{'='*60}")
    print(
        f"⏱️  Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print(f"📊 Items processed: {completed_count}")
    print(f"⚡ Average time per item: {total_time/completed_count:.2f} seconds")
    print(f"🚀 Throughput: {completed_count*3600/total_time:.1f} items/hour")
    print(f"{'='*60}")

    output_filename = "output/about_food_classification_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Automation complete! Results saved to {output_filename}")
