import streamlit as st
import json
import os
from parsing.pdf_parser import extract_text
from utils.text_preprocessor import preprocess_text
from utils.rule_based_extractor import RuleBasedExtractor
from utils.spacy_ner_extractor import SpacyNERExtractor
from utils.llm_based_extractor import extract_entities_from_api
from scoring.scorer import ResumeScorerAPI
from scoring.rater import ResumeRaterAPI
from scoring.feedback import (
    get_feedback_from_resume_text,
    get_strength_from_resume_text,
    get_weakness_from_resume_text,
)


def format_education_entries(education_list):
    corrected = []
    for item in education_list:
        item = item.replace("b.tech", "B.Tech").replace("m.tech", "M.Tech").replace("b.sc", "B.Sc").replace("m.sc", "M.Sc")
        item = " ".join([word.capitalize() for word in item.split()])
        corrected.append(item)
    return corrected


def main():
    st.set_page_config("Resume Analyzer", layout="wide")
    st.title("📄 Resume Analyzer")
    st.markdown("Upload a resume and provide a job description to receive a detailed analysis.")

    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
    job_description = st.text_area("Enter the job description", height=150)

    if uploaded_file and job_description:
        with st.spinner("Analyzing your resume..."):

            # Step 1: Extract text from uploaded PDF
            with open("temp_resume.pdf", "wb") as f:
                f.write(uploaded_file.read())
            resume_text = extract_text("temp_resume.pdf")

            # Step 2: Preprocess Text
            tokens = preprocess_text(resume_text)

            # Step 3: Rule-Based Extraction
            rule_extractor = RuleBasedExtractor(resume_text, tokens)
            rule_based_data = {
                "name": rule_extractor.extract_name(),
                "email": rule_extractor.extract_email(),
                "phone": rule_extractor.extract_phone(),
                "skills": rule_extractor.extract_skills(),
                "education": rule_extractor.extract_education(),
            }

            # Step 4: spaCy NER Extraction (optional - ignored for display)
            ner_extractor = SpacyNERExtractor()
            _ = ner_extractor.extract_entities(resume_text)

            # Step 5: LLM-Based Extraction (optional - ignored for display)
            _ = extract_entities_from_api(resume_text)

            # Step 6: Relevance Score
            scorer = ResumeScorerAPI()
            relevance_score = scorer.score_similarity(resume_text, job_description)

            # Step 7: Resume Rating
            rater = ResumeRaterAPI()
            resume_rating = rater.rate_resume(resume_text)

            # Step 8: LLM Feedback
            strengths = get_strength_from_resume_text(resume_text)
            weaknesses = get_weakness_from_resume_text(resume_text)
            suggestions = get_feedback_from_resume_text(resume_text)

            # Clean up temporary file
            os.remove("temp_resume.pdf")

        # Display Results
        st.markdown("## 📊 Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resume Rating", f"{resume_rating}/10")
        with col2:
            st.metric("Relevance Score", f"{relevance_score}/100")

        st.markdown("---")
        st.subheader("🧾 Basic Information")
        st.markdown(f"**Name:** {rule_based_data['name'].title() if rule_based_data['name'] else 'Not Found'}")
        st.markdown(f"**Email:** {rule_based_data['email']}")
        st.markdown(f"**Phone:** {rule_based_data['phone']}")

        st.markdown("---")
        st.subheader("🎓 Education")
        formatted_edu = format_education_entries(rule_based_data["education"])
        if formatted_edu:
            for edu in formatted_edu:
                st.markdown(f"- {edu}")
        else:
            st.write("No education details found.")

        st.markdown("---")
        st.subheader("🛠️ Skills")
        if rule_based_data["skills"]:
            for skill in rule_based_data["skills"]:
                st.markdown(f"- {skill.title()}")
        else:
            st.write("No skills found.")

        st.markdown("---")
        st.subheader("💡 LLM Feedback")

        with st.expander("✅ Strengths"):
            st.write(strengths or "No strengths identified.")

        with st.expander("⚠️ Weaknesses"):
            st.write(weaknesses or "No weaknesses identified.")

        with st.expander("🛠 Suggestions for Improvement"):
            st.write(suggestions or "No suggestions provided.")


if __name__ == "__main__":
    main()
