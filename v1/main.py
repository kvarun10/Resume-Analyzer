# main.py

from parsing.pdf_parser import extract_text
from utils.text_preprocessor import preprocess_text
from utils.rule_based_extractor import RuleBasedExtractor
from utils.llm_based_extractor import extract_entities_from_api


# Step 1: Read PDF
pdf_path = r"D:\resume-analyzer\VarunKaushal-Resume.pdf"
resume_text = extract_text(pdf_path)

# Step 2: Preprocess Text
tokens = preprocess_text(resume_text)

# Step 3: Rule-Based Extraction
print("\n--- Using Rule-Based Extractor ---")
extractor = RuleBasedExtractor(resume_text, tokens)
name = extractor.extract_name()
email = extractor.extract_email()
phone = extractor.extract_phone()
skills = extractor.extract_skills()
education = extractor.extract_education()

# Output
print("Name:", name)
print("Email:", email)
print("Phone:", phone)
print("Skills:", skills)
print("Education:", education)

print("\n--- Using LLM-Based Extractor ---")
llm_entities = extract_entities_from_api(resume_text)

print("LLM Entities:")
for key, values in llm_entities.items():
    print(f"{key}: {values}")

#FOR MERGING THEM BOTH
# def merge_extracted_data(rule_based_data, llm_data):
#     merged = {
#         "name": rule_based_data.get("name"),
#         "email": rule_based_data.get("email"),
#         "phone": rule_based_data.get("phone"),
#         "skills": list(set(rule_based_data.get("skills", []) + llm_data.get("MISC", []))),
#         "education": rule_based_data.get("education", []),
#         "organizations": list(set(llm_data.get("ORG", []))),
#         "locations": list(set(llm_data.get("LOC", []))),
#         "people": list(set(llm_data.get("PERSON", []))),
#     }
#     return merged
