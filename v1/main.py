# main.py (version 1 - text-based feedback)
import json
from parsing.pdf_parser import extract_text
from utils.text_preprocessor import preprocess_text
from utils.rule_based_extractor import RuleBasedExtractor
from utils.spacy_ner_extractor import SpacyNERExtractor
from utils.llm_based_extractor import extract_entities_from_api
from scoring.scorer import ResumeScorerAPI
from scoring.rater import ResumeRaterAPI
from scoring.feedback import get_feedback_from_resume_text,get_strength_from_resume_text,get_weakness_from_resume_text

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

# Step 6: Relevance Score using Hugging Face API
job_description = """
Looking for a developer with ML experience,
strong in algorithms and data structures, and preferably some frontend and backend.
 """
scorer = ResumeScorerAPI()
relevance_score = scorer.score_similarity(resume_text, job_description)

# Step 7: Resume Rating
rater = ResumeRaterAPI()
resume_rating = rater.rate_resume(resume_text)

# Step 8: LLM Feedback
strengths= get_strength_from_resume_text(resume_text)
weakness= get_weakness_from_resume_text(resume_text)
feedback = get_feedback_from_resume_text(resume_text)
llm_feedback = {
    "strengths": strengths,
    "weaknesses": weakness,
    "suggestions": feedback
}

# Combine All Outputs
combined_output = rule_based_data.copy()
combined_output["spacy_entities"] = ner_entities
combined_output["llm_entities"] = llm_entities
combined_output["relevance_score_out_of_100"] = relevance_score
combined_output["resume_rating_out_of_10"] = resume_rating
combined_output["llm_feedback"] = llm_feedback

# Save Final JSON
with open("extracted_data.json", "w", encoding="utf-8") as f:
    json.dump(combined_output, f, indent=4, ensure_ascii=False)
