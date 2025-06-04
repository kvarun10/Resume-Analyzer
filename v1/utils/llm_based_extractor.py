# extractor/llm_api_extractor.py

import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_huggingface_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # will raise an error for bad requests
    return response.json()

def extract_entities_from_api(text):
    """
    Uses Hugging Face's hosted inference API to extract named entities.
    Returns a dictionary with grouped entities.
    """
    results = query_huggingface_api({"inputs": text})

    extracted = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}

    for entity in results:
        entity_type = entity.get("entity_group", "")
        word = entity.get("word", "")
        if entity_type in extracted:
            extracted[entity_type].append(word)

    return extracted


#GEMINI
# import os
# import requests
# from transformers import pipeline # Still useful if you want a local fallback or other pipelines

# class LLMExtractor:
#     def __init__(self, use_api=False, hf_token=None):
#         self.use_api = use_api
#         self.model_id = "dslim/bert-base-NER" # You can change this to other NER models

#         if self.use_api:
#             self.api_token = hf_token if hf_token else os.getenv("HF_API_TOKEN")
#             if not self.api_token:
#                 raise ValueError("Hugging Face API token not provided or HF_API_TOKEN not set.")
#             self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
#             self.headers = {"Authorization": f"Bearer {self.api_token}"}
#             print(f"LLMExtractor initialized to use Inference API for model: {self.model_id}")
#         else:
#             # This runs the model locally after downloading it from Hugging Face Hub
#             self.ner_pipeline_local = pipeline("ner", model=self.model_id, grouped_entities=True)
#             print(f"LLMExtractor initialized to use local pipeline for model: {self.model_id}")

#     def _query_api(self, payload):
#         response = requests.post(self.api_url, headers=self.headers, json=payload)
#         response.raise_for_status()  # Raises an exception for HTTP errors
#         return response.json()

#     def extract_entities(self, text):
#         """
#         Extract named entities from the text.
#         Uses local pipeline or Inference API based on initialization.
#         """
#         if not text.strip(): # Handle empty or whitespace-only input
#             return {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}

#         if self.use_api:
#             try:
#                 api_result = self._query_api({"inputs": text, "options": {"wait_for_model": True}})
#                 # The API output format might be slightly different, adjust parsing as needed
#                 # For dslim/bert-base-NER, it's usually a list of entity objects
#                 extracted = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}
#                 if isinstance(api_result, list):
#                     for entity in api_result:
#                         entity_type = entity.get("entity_group")
#                         word = entity.get("word")
#                         # API might sometimes add # signs or split words, might need cleaning
#                         if word:
#                             word = word.replace("##", "") 
#                         if entity_type in extracted and word:
#                             extracted[entity_type].append(word)
#                 return extracted
#             except requests.exceptions.RequestException as e:
#                 print(f"API request failed: {e}")
#                 # Optionally, fall back to local model or return empty
#                 return {"PERSON": [], "ORG": [], "LOC": [], "MISC": []} # Or raise error
#         else:
#             # Use the local pipeline
#             entities = self.ner_pipeline_local(text)
#             extracted = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}
#             for entity in entities:
#                 entity_type = entity.get("entity_group")
#                 word = entity.get("word")
#                 if entity_type in extracted and word:
#                     extracted[entity_type].append(word)
#             return extracted

#NOT FINAL....HAVE TO USE HUGGING FACE

# from transformers import pipeline

# class LLMExtractor:
#     def __init__(self):
#         self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

#     def extract_entities(self, text):
#         """
#         Extract named entities from the resume using a BERT-based NER model.
#         Returns a dictionary categorizing entities like names, organizations, etc.
#         """
#         entities = self.ner_pipeline(text)
#         extracted = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}

#         for entity in entities:
#             entity_type = entity["entity_group"]
#             word = entity["word"]
#             if entity_type in extracted:
#                 extracted[entity_type].append(word)

#         return extracted
