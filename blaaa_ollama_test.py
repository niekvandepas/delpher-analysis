from random import shuffle
from time import time
import ollama
import json
import os
from dotenv import load_dotenv

from constants import PROJECT_DIR
from data_import import import_search_results_ndjson, normalize_unicode, strip_xml_tags

MODEL_NAME = 'llama3:8b'
CLASSIFICATION_CATEGORIES = "[Is about food], [Is not about food]"
EXAMPLE_TEXT_1 = '\'Drugsteams met Fransen Frankrijk en Nederland strijden samen tegen XTC Van onze politieke redactie Den Haag - Er komen speciale Frans-Nederlandse teams om de productie en verkoop van synthetische drugs te bestrijden. President Chirac en premier Kok hebben dat gisteren afgesproken op de tweede dag van Chiracs staatsbezoek. Chirac noemde de samenwerking in de strijd tegen drugs als stc \'bijzonder goed\' en \'voorbeeldig\'. Hoewel er \'verschil van inzicht\' blijft bestaan, werken beide landen \'op positieve wijze\' samen, zei hij gisteravond als eregast op het jaarlijkse diner van de Vrienden van Nieuwspoort. Frankrijk en Nederland sloegen in 1995 de handen ineen inzake politie, justitie en douane. De samenwerking kreeg een deuk toen een Franse senator Nederland begin 1996 een \'narcostaat\' noemde en Frankrijk de grenscontrole met België weer inroerde om het drugstoerisme de pas af te snijden. Pas sinds 1997 zit er vaart in de samenwerking. De toenmalige ministers van Justitie Sorgdrager en Toubon maakten in dat jaar een einde aan de pesterijtjes over en weer met een symbolische zoen voor de camera\'s. Met steun van Chirac en Kok werd vanaf dat moment voluit de strijd tegen xtc ingezet. Gisteren zei Chirac overigens dat hij de tijd nog niet rijp acht voor volledige toepassing van het Verdrag van Schengen, dat open grenzen voorschrijft. Premier Kok zat er niet mee. Hij deed de handhaving van de grenscontrole gisteren af als \'symboolpolitiek\': "In de praktijk stelt het weinig voor." Het staatsbezoek stond in het teken van verzoening en begrip. Volgens Chirac leidden \'natuurwetten, vrijheid van denken en economische noodzaak\' in Nederland tot \'wat wij Fransen vaak voor overdreven laisser-faire houden\', maar wat Nederland beschouwt als \'een rechtmatig gedogen\'. "Ikben blij hier te zijn, en ditis absoluut geen diplomatiek zinnetje", zei Chirac. Op pagina g| Chirac vraagt begrip voor Franse deugden Chirac vraagt begrip voor Franse deugden Vervolg van pagina i Staatsbezoek Maar de liefde kan niet van één kant komen. Chirac vroeg eveneens begrip voor enkele Franse deugden, die vaak versleten worden voor verkalkte tradities. "U vindt dat het alles \'staatsbemoeienis\' is wat klok slaat in Frankrijk. Maar dat is steeds minder het geval. U vindt dat Frankrijk protectionistische trekken vertoont. Maar wij staan in de wereld op de vierde plaats qua in- en uitvoer. Frankrijk wekt vaak verbazing doordat het zo aan zijn taal en cultuur gehecht is. Het tegendeel is waar: wij zitten midden in de mondialisering, maar opening is niet synoniem met eenvormigheid." De jonge SP-senator Van Vugt (20) werd gisteren gearresteerd wegens verstoring van de openbare orde: hij protesteerde zonder vergunning in het aangezicht van Chirac tegen het Franse nucleaire beleid met een fluitje en een T-shirt dat herinnerde aan de Franse kernproeven. Andere demonstranten slaagden er in te ontkomen.\''
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

    # Construct the messages list for the /api/chat endpoint
    messages = [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Test Text: {text}\nCategory:"}
    ]

    try:
        # Call the chat endpoint
        response = client.chat(
            model=MODEL_NAME,
            messages=messages,
            options={"temperature": 0.0} # Lower temperature for deterministic classification
        )
        # Extract and clean up the model's response
        # We use .strip() to remove any leading/trailing whitespace
        return response['message']['content'].strip()

    except Exception as e:
        # Simple error handling for connection issues, etc.
        return f"ERROR: {e}"


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncates text to a maximum number of characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

if __name__ == "__main__":
    dotenv_path = os.path.join(PROJECT_DIR, '.env')
    load_dotenv(dotenv_path)

    classification_data_path = os.environ.get("CLASSIFICATION_DATA_PATH")

    if not classification_data_path:
        raise ValueError("Please set the CLASSIFICATION_DATA_PATH environment variable.")

    texts_to_classify = import_search_results_ndjson(classification_data_path)

    # Shuffle the to make it easier to measure performance on a random sample
    shuffle(texts_to_classify)

    client = ollama.Client()

    results = []

    print(f"Starting classification of {len(texts_to_classify)} texts with {MODEL_NAME}...")

    global_start_time = time()

    out_file_path = os.environ.get("OLLAMA_OUT_FILE_PATH")

    if not out_file_path:
        raise ValueError("Please set the OLLAMA_OUT_FILE_PATH environment variable.")

    if not os.path.exists(out_file_path):
        open(out_file_path, 'w').close()
    with open(out_file_path, "r") as f:
        out_file_contents = f.read()

    for i, search_result in enumerate(texts_to_classify):
        if search_result.identifier is not None and search_result.identifier in out_file_contents:
            print(f"Skipping {i+1}/{len(texts_to_classify)} (ID: {search_result.identifier}) - already processed.")
            continue
        start_time = time()

        text = truncate_text(
            normalize_unicode(
                strip_xml_tags(search_result.ocr_xml)
            ),
            2000
        )
        classification = classify_text_with_ollama(client, text)
        results.append({
            "id": search_result.identifier,
            "original_text": text,
            "predicted_category": classification
        })

        end_time = time()

        with open(out_file_path, 'a', encoding='utf-8') as out_file:
            out_file.write(f"ID: {search_result.identifier}\n")
            out_file.write(f"Predicted Category: {classification}\n")
            out_file.write(f"Original Text: {text}\n")
            out_file.write("--------------------------------------------------\n")
        print(f"Processed {i+1}/{len(texts_to_classify)}. Result: {classification}. Took {end_time - start_time:.2f} seconds.")

    global_end_time = time()
    total_time = global_end_time - global_start_time
    print(f"\n⏱️  Total time taken: {total_time:.2f} seconds")

    # Save all results to a JSON file
    output_filename = "food_classification_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Automation complete! Results saved to {output_filename}")
