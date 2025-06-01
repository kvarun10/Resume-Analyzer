import re

# Skill Extraction
TECH_SKILLS = ["python", "java", "c++", "sql", "machine learning", "deep learning", "data analysis", "nlp",
               "html", "css", "javascript", "react", "node.js", "django"]

def extract_skills(tokens):
    found_skills = set()
    for skill in TECH_SKILLS:
        if skill.lower() in " ".join(tokens):
            found_skills.add(skill)
    return list(found_skills)


# ✅ NER using spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

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


# Extract Experience Section

def extract_experience_section(text):
    pattern = r"(?i)(experience|work history|employment history)(.*?)(education|skills|projects|certifications|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2).strip() if match else ""


# Extract Education Section

def extract_education_section(text):
    pattern = r"(?i)(education)(.*?)(experience|skills|projects|certifications|$)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(2).strip() if match else ""


# Classify Sections

def classify_sections(text):
    sections = {}
    headers = ["education", "experience", "projects", "skills", "certifications"]
    for header in headers:
        pattern = rf"(?i){header}(.*?)(?=(education|experience|projects|skills|certifications|$))"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[header.capitalize()] = match.group(1).strip()
    return sections


# 🚀 Usage Example
if __name__ == "__main__":
    with open("sample_resume-output.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    tokens = preprocess_text(resume_text)
    skills = extract_skills(tokens)
    entities = extract_entities(resume_text)
    experience = extract_experience_section(resume_text)
    education = extract_education_section(resume_text)
    sections = classify_sections(resume_text)

    result = {
        "Entities": entities,
        "Skills": skills,
        "Experience": experience,
        "Education": education,
        "Sections": sections
    }

    from pprint import pprint
    pprint(result)
