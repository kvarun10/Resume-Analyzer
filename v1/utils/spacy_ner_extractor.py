# utils/spacy_ner_extractor.py

import spacy

class SpacyNERExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {"PERSON": [], "ORG": [], "GPE": [], "DATE": []}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        return entities
