# extractor.py

import re
import requests
import json
import spacy
from preprocessing.preprocessor import preprocess_text

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Classical NLP NER (spaCy)
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    data = {"NAME": None, "EMAILS": [], "PHONES": [], "DEGREES": [], "ORG": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not data["NAME"]:
            data["NAME"] = ent.text
        elif ent.label_ == "EMAIL":
            data["EMAILS"].append(ent.text)
        elif ent.label_ == "PHONE":
            data["PHONES"].append(ent.text)
        elif ent.label_ in ["ORG"]:
            data["ORG"].append(ent.text)
        elif ent.label_ in ["EDUCATION", "DEGREE"]:
            data["DEGREES"].append(ent.text)
    return data

# -----------------------------
# Hugging Face NER API-Based Extraction (BERT)
# -----------------------------
HF_API_TOKEN = "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"
HF_MODEL = "dslim/bert-base-NER"

def extract_entities_from_api(text):
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json={"inputs": text})
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        print("❌ Hugging Face API NER Error:", e)
        return {}

    grouped = {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}
    for entity in results:
        label = entity.get("entity_group", "")
        word = entity.get("word", "")
        if label in grouped:
            grouped[label].append(word)

    return grouped

# -----------------------------
# LLM-Based Resume Section Extraction (Mistral)
# -----------------------------
# HF_LLM_MODEL = " "

# def extract_sections_via_llm(text):
#     url = f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}"
#     headers = {
#         "Authorization": f"Bearer {HF_API_TOKEN}",
#         "Content-Type": "application/json"
#     }

#     prompt = f"""
# You are an AI assistant helping extract structured data from resumes.
# Given the following resume text, extract:

# 1. Full name
# 2. Email
# 3. Phone number
# 4. Education section
# 5. Experience section
# 6. Projects section
# 7. Skills

# Return your output as valid JSON using these keys: name, email, phone, education, experience, projects, skills.

# Resume:
# {text}
# """

#     try:
#         response = requests.post(url, headers=headers, json={"inputs": prompt})
#         response.raise_for_status()
#         result = response.json()
#         generated_text = result[0].get("generated_text", "")
#         return json.loads(generated_text)
#     except Exception as e:
#         print("❌ Failed to extract structured LLM output:", e)
#         return {}

# -----------------------------
# Skills Extraction (Keyword)
# -----------------------------
TECH_SKILLS = ["python", "java", "c++", "sql", "machine learning", "deep learning", "data analysis", "nlp",
               "html", "css", "javascript", "react", "node.js", "django"]

def extract_skills(tokens):
    found_skills = set()
    for skill in TECH_SKILLS:
        if skill.lower() in " ".join(tokens):
            found_skills.add(skill)
    return list(found_skills)

# -----------------------------
# Rule-based Section Extraction (Optional Backup)
# -----------------------------
def extract_experience_section(text):
    pattern = r"(?i)(experience|work history|employment history)(.*?)(education|skills|projects|certifications|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2).strip() if match else ""

def extract_education_section(text):
    pattern = r"(?i)(education)(.*?)(experience|skills|projects|certifications|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2).strip() if match else ""

def classify_sections(text):
    sections = {}
    headers = ["education", "experience", "projects", "skills", "certifications"]
    for header in headers:
        pattern = rf"(?i){header}(.*?)(?=(education|experience|projects|skills|certifications|$))"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[header.capitalize()] = match.group(1).strip()
    return sections

# -----------------------------
# Main Test (Optional)
# -----------------------------
if __name__ == "__main__":
    with open("sample_resume-output.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    tokens = preprocess_text(resume_text)
    skills = extract_skills(tokens)
    spaCy_entities = extract_entities(resume_text)
    hf_entities = extract_entities_from_api(resume_text)
    # structured_llm = extract_sections_via_llm(resume_text)
    experience = extract_experience_section(resume_text)
    education = extract_education_section(resume_text)
    sections = classify_sections(resume_text)

    result = {
        "Skills": skills,
        "spaCy_NER": spaCy_entities,
        "HF_NER_BERT_API": hf_entities,
        # "Structured_LLM": structured_llm,
        "Experience (Regex)": experience,
        "Education (Regex)": education,
        "Sections (Regex)": sections
    }

    from pprint import pprint
    pprint(result)
