#!/usr/bin/env python3

import json
from pathlib import Path
from transformers import pipeline  # type: ignore

classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1
)

candidate_labels = ["sustainability", "health", "economics"]

def classify_articles(english_texts: list[str]) -> list[dict]:
    results = []

    for i, text in enumerate(english_texts):
      # https://huggingface.co/typeform/distilbert-base-uncased-mnli/blame/b91e7a74c63c287d22a105a9f050cd26d648879f/config.json
      max_tokens = 512

      truncated_text = " ".join(text.split()[:max_tokens])
      classification = classifier(truncated_text, candidate_labels)
      label = classification["labels"][0]
      score = classification["scores"][0]

      results.append({
          "translated_text": text,
          "label": label,
          "score": score
      })

      print(f"Processed article #{i}: {label} ({score:.2f})", end="\r")

    # Ensure last line stays visible
    print(f"Processed {len(english_texts)} articles", end="\n")
    return results
