# main.py

import json
from parsing.pdf_parser import extract_text
from utils.text_preprocessor import preprocess_text
from utils.rule_based_extractor import RuleBasedExtractor
from utils.spacy_ner_extractor import SpacyNERExtractor
from utils.llm_based_extractor import extract_entities_from_api

# Step 1: Read PDF
pdf_path = r"D:\resume-analyzer\VarunKaushal-Resume.pdf"
resume_text = extract_text(pdf_path)

# Step 2: Preprocess Text
tokens = preprocess_text(resume_text)

# Step 3: Rule-Based Extraction
rule_extractor = RuleBasedExtractor(resume_text, tokens)
rule_based_data = {
    "name": rule_extractor.extract_name(),
    "email": rule_extractor.extract_email(),
    "phone": rule_extractor.extract_phone(),
    "skills": rule_extractor.extract_skills(),
    "education": rule_extractor.extract_education()
}

# Step 4: spaCy NER Extraction
ner_extractor = SpacyNERExtractor()
ner_entities = ner_extractor.extract_entities(resume_text)

# Step 5: LLM-Based Extraction
llm_entities = extract_entities_from_api(resume_text)

# Combine all outputs
combined_output = rule_based_data.copy()
combined_output["spacy_entities"] = ner_entities
combined_output["llm_entities"] = llm_entities

# Save to JSON file
with open("extracted_data.json", "w", encoding="utf-8") as f:
    json.dump(combined_output, f, indent=4, ensure_ascii=False)

