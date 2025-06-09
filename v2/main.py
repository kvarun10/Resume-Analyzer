from parsing.pdf_parser import extract_text
from preprocessing.preprocessor import preprocess_text
from parsing.extractor import (
    extract_entities,
    extract_entities_from_api,
    extract_experience_section,
    extract_education_section,
    extract_skills,
    classify_sections,
    # extract_sections_via_llm
)
from scoring.scorer import rule_based_score, llm_based_score
import json

def main():
    # 1. Extract text from PDF
    resume_path = "sample_resume1.pdf"
    raw_text = extract_text(resume_path)

    # 2. Preprocess text (cleaning, lowercasing, removing stopwords, etc.)
    cleaned_text = preprocess_text(raw_text)

    # 3. Extract Skills (NLP rule-based)
    skills = extract_skills(cleaned_text)

    # 4. Named Entity Recognition (spaCy)
    entities_spacy = extract_entities(raw_text)

    # 5. Named Entity Recognition (Hugging Face BERT API)
    entities_hf = extract_entities_from_api(raw_text)

    # 6. Experience section (regex)
    experience = extract_experience_section(raw_text)

    # 7. Education section (regex)
    education = extract_education_section(raw_text)

    # 8. Section Classification (regex)
    sections = classify_sections(raw_text)

    # 9. Structured extraction using LLM (optional placeholder for now)
    structured_llm = {
        "projects": "Project 1 details... Project 2 details..." * 2,
        "education": "B.Tech in Computer Science from XYZ..." * 2
    }

    # 10. Combine all data
    resume_data = {
        "skills": skills,
        "spaCy_entities": entities_spacy,
        "HF_entities_BERT_API": entities_hf,
        "experience_regex": experience,
        "education_regex": education,
        "sections_regex": sections,
        "structured_llm": structured_llm
    }

    # 11. Scoring
    score, feedback = rule_based_score(resume_data)
    llm_feedback = llm_based_score(raw_text)

    resume_data["score"] = score
    resume_data["rule_feedback"] = feedback
    resume_data["llm_feedback"] = llm_feedback

    # 12. Save JSON
    with open("structured_resume.json", "w", encoding="utf-8") as f:
        json.dump(resume_data, f, ensure_ascii=False, indent=4)

    print("✅ Resume analysis complete. Data saved to structured_resume.json")

if __name__ == "__main__":
    main()
