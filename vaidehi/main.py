from parsing.pdf_parser import extract_text
from preprocessing.preprocessor import clean_resume_text
from parsing.extractor import (
    extract_skills,
    extract_entities,
    extract_experience_section,  # Changed from extract_experience
    extract_education_section,   # Changed from extract_education
    classify_sections,
)
import json

def main():
    # 1. Extract text from PDF
    resume_path = "sample_resume.pdf"
    raw_text = extract_text(resume_path)

    # 2. Preprocess text (cleaning, lowercasing, removing stopwords, etc.)
    cleaned_text = clean_resume_text(raw_text)

    # 3. Extract Skills (Day 4)
    skills = extract_skills(cleaned_text)

    # 4. Named Entity Recognition (Day 5) - Use raw text for spaCy
    entities = extract_entities(raw_text)

    # 5. Extract Experience section (Day 6)
    experience = extract_experience_section(raw_text)  # Updated function name

    # 6. Extract Education section (Day 7)
    education = extract_education_section(raw_text)    # Updated function name

    # 7. Section Classification (Day 8)
    sections = classify_sections(raw_text)

    # 8. Combine all extracted data
    resume_data = {
        "entities": entities,
        "skills": skills,
        "experience": experience,
        "education": education,
        "sections": sections
    }

    # 9. Output to structured JSON
    with open("structured_resume.json", "w", encoding="utf-8") as f:
        json.dump(resume_data, f, ensure_ascii=False, indent=4)

    print("✅ Resume analysis complete. Data saved to structured_resume.json")

if __name__ == "__main__":
    main()
