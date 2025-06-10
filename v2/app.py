import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import tempfile
from io import StringIO

# Import your existing modules
from parsing.pdf_parser import extract_text
from preprocessing.preprocessor import preprocess_text
from parsing.extractor import (
    extract_entities,
    extract_entities_from_api,
    extract_experience_section,
    extract_education_section,
    extract_skills,
    classify_sections,
)
from scoring.scorer import rule_based_score, llm_based_score

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .feedback-positive {
        color: #28a745;
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
    .feedback-negative {
        color: #dc3545;
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_score_gauge(score):
    """Create a gauge chart for the resume score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Resume Score"},
        delta = {'reference': 7},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 5], 'color': "lightgray"},
                {'range': [5, 8], 'color': "yellow"},
                {'range': [8, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_skills_chart(skills):
    """Create a bar chart for skills"""
    if not skills:
        return None
    
    skills_df = pd.DataFrame({
        'Skill': skills,
        'Count': [1] * len(skills)
    })
    
    fig = px.bar(
        skills_df, 
        x='Skill', 
        y='Count',
        title='Detected Skills',
        color='Skill'
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig

def create_entities_chart(entities):
    """Create a pie chart for entity types"""
    entity_counts = {}
    for entity_type, values in entities.items():
        if isinstance(values, list):
            entity_counts[entity_type] = len(values)
        elif values:
            entity_counts[entity_type] = 1
        else:
            entity_counts[entity_type] = 0
    
    # Filter out empty entities
    entity_counts = {k: v for k, v in entity_counts.items() if v > 0}
    
    if not entity_counts:
        return None
    
    fig = px.pie(
        values=list(entity_counts.values()),
        names=list(entity_counts.keys()),
        title='Detected Entities Distribution'
    )
    return fig

def analyze_resume(uploaded_file):
    """Analyze the uploaded resume file"""
    with st.spinner("🔍 Analyzing your resume..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # 1. Extract text from PDF
            raw_text = extract_text(tmp_file_path)
            
            # 2. Preprocess text
            cleaned_text = preprocess_text(raw_text)
            
            # 3. Extract Skills
            skills = extract_skills(cleaned_text)
            
            # 4. Named Entity Recognition (spaCy)
            entities_spacy = extract_entities(raw_text)
            
            # 5. Named Entity Recognition (Hugging Face BERT API)
            with st.spinner("🤖 Processing with AI models..."):
                entities_hf = extract_entities_from_api(raw_text)
            
            # 6. Extract sections
            experience = extract_experience_section(raw_text)
            education = extract_education_section(raw_text)
            sections = classify_sections(raw_text)
            
            # 7. Mock structured LLM data (replace with actual implementation)
            structured_llm = {
                "projects": "Project details extracted...",
                "education": "Education details extracted..."
            }
            
            # 8. Combine all data
            resume_data = {
                "skills": skills,
                "spaCy_entities": entities_spacy,
                "HF_entities_BERT_API": entities_hf,
                "experience_regex": experience,
                "education_regex": education,
                "sections_regex": sections,
                "structured_llm": structured_llm,
                "raw_text": raw_text
            }
            
            # 9. Scoring
            score, feedback = rule_based_score(resume_data)
            
            # LLM feedback (optional, can be slow)
            if st.session_state.get('use_llm_feedback', False):
                with st.spinner("🧠 Getting AI feedback..."):
                    llm_feedback = llm_based_score(raw_text)
            else:
                llm_feedback = "LLM feedback disabled to improve performance."
            
            resume_data["score"] = score
            resume_data["rule_feedback"] = feedback
            resume_data["llm_feedback"] = llm_feedback
            
            return resume_data
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Upload your resume and get instant analysis with AI-powered insights!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        use_llm_feedback = st.checkbox("Enable LLM Feedback", help="May take longer to process")
        st.session_state['use_llm_feedback'] = use_llm_feedback
        
        st.header("📊 Analysis Methods")
        st.info("""
        **NLP Techniques Used:**
        - spaCy NER
        - BERT-based NER
        - Regex Pattern Matching
        - Rule-based Scoring
        - Skills Extraction
        """)
        
        if st.button("📋 View Sample Analysis"):
            st.session_state['show_sample'] = True
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF resume file",
        type="pdf",
        help="Upload a PDF resume for analysis"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Analyze button
        if st.button("🚀 Analyze Resume", type="primary"):
            try:
                # Analyze the resume
                results = analyze_resume(uploaded_file)
                
                # Store results in session state
                st.session_state['analysis_results'] = results
                st.session_state['file_name'] = uploaded_file.name
                
            except Exception as e:
                st.error(f"❌ Error analyzing resume: {str(e)}")
                st.error("Please check your file format and try again.")
    
    # Display results if they exist
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        file_name = st.session_state.get('file_name', 'Unknown')
        
        st.markdown(f'<h2 class="section-header">📈 Analysis Results for: {file_name}</h2>', unsafe_allow_html=True)
        
        # Score section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(create_score_gauge(results['score']), use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Quick Stats")
            
            # Metrics
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Score", f"{results['score']}/10")
                st.metric("Skills Found", len(results.get('skills', [])))
            
            with col2_2:
                name = results.get('spaCy_entities', {}).get('NAME', 'Not detected')
                st.metric("Name", "✅" if name != 'Not detected' else "❌")
                emails = len(results.get('spaCy_entities', {}).get('EMAILS', []))
                st.metric("Emails", emails)
            
            with col2_3:
                phones = len(results.get('spaCy_entities', {}).get('PHONES', []))
                st.metric("Phone Numbers", phones)
                orgs = len(results.get('spaCy_entities', {}).get('ORG', []))
                st.metric("Organizations", orgs)
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Skills & Entities", 
            "📊 Sections", 
            "💡 Feedback", 
            "🔍 Raw Data", 
            "📄 Original Text"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🛠️ Detected Skills")
                if results.get('skills'):
                    for skill in results['skills']:
                        st.markdown(f"- {skill}")
                    
                    # Skills chart
                    skills_chart = create_skills_chart(results['skills'])
                    if skills_chart:
                        st.plotly_chart(skills_chart, use_container_width=True)
                else:
                    st.warning("No skills detected. Consider adding more technical skills to your resume.")
            
            with col2:
                st.markdown("#### 👤 Named Entities (spaCy)")
                spacy_entities = results.get('spaCy_entities', {})
                
                if spacy_entities.get('NAME'):
                    st.success(f"**Name:** {spacy_entities['NAME']}")
                
                if spacy_entities.get('EMAILS'):
                    st.info(f"**Emails:** {', '.join(spacy_entities['EMAILS'])}")
                
                if spacy_entities.get('PHONES'):
                    st.info(f"**Phones:** {', '.join(spacy_entities['PHONES'])}")
                
                if spacy_entities.get('ORG'):
                    st.info(f"**Organizations:** {', '.join(spacy_entities['ORG'])}")
                
                # Entities chart
                entities_chart = create_entities_chart(spacy_entities)
                if entities_chart:
                    st.plotly_chart(entities_chart, use_container_width=True)
        
        with tab2:
            st.markdown("#### 📑 Resume Sections")
            
            sections = results.get('sections_regex', {})
            if sections:
                for section_name, content in sections.items():
                    with st.expander(f"{section_name.title()} Section"):
                        if content.strip():
                            st.text_area("", content, height=100, disabled=True)
                        else:
                            st.warning(f"No {section_name.lower()} content detected")
            else:
                st.warning("No clear sections detected. Consider using standard section headers like 'Experience', 'Education', 'Skills'.")
        
        with tab3:
            st.markdown("#### 💡 Improvement Suggestions")
            
            # Rule-based feedback
            st.markdown("##### 🤖 Automated Analysis")
            if results.get('rule_feedback'):
                for feedback in results['rule_feedback']:
                    st.markdown(f'<div class="feedback-negative">⚠️ {feedback}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="feedback-positive">✅ All basic requirements met!</div>', unsafe_allow_html=True)
            
            # LLM feedback
            if results.get('llm_feedback') and results['llm_feedback'] != "LLM feedback disabled to improve performance.":
                st.markdown("##### 🧠 AI-Powered Insights")
                st.markdown(results['llm_feedback'])
        
        with tab4:
            st.markdown("#### 🔍 Detailed Extraction Data")
            
            # Display raw extraction data
            st.json(results)
            
            # Download button for JSON
            json_str = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Analysis as JSON",
                data=json_str,
                file_name=f"resume_analysis_{file_name.replace('.pdf', '')}.json",
                mime="application/json"
            )
        
        with tab5:
            st.markdown("#### 📄 Extracted Text")
            if results.get('raw_text'):
                st.text_area("Original Resume Text", results['raw_text'], height=400, disabled=True)
            else:
                st.warning("No text could be extracted from the PDF.")
    
    # Sample analysis section
    if st.session_state.get('show_sample', False):
        st.markdown('<h2 class="section-header">📋 Sample Analysis</h2>', unsafe_allow_html=True)
        
        sample_data = {
            "score": 7,
            "skills": ["Python", "Machine Learning", "SQL", "JavaScript"],
            "spaCy_entities": {
                "NAME": "John Doe",
                "EMAILS": ["john.doe@email.com"],
                "PHONES": ["+1-234-567-8900"],
                "ORG": ["Google", "Microsoft"]
            },
            "rule_feedback": ["Consider adding more project details", "Add certifications section"]
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_score_gauge(sample_data['score']), use_container_width=True)
        
        with col2:
            st.markdown("**Sample Skills:**")
            for skill in sample_data['skills']:
                st.markdown(f"- {skill}")
        
        if st.button("❌ Hide Sample"):
            st.session_state['show_sample'] = False
            st.rerun()

if __name__ == "__main__":
    main()

# import streamlit as st
# import json

# # Load JSON file (you can also upload it if needed)
# with open("structured_resume.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# st.set_page_config(page_title="📄 Resume Review", layout="centered")
# st.title("📄 Resume Analyzer")
# st.markdown("Get an overview of your resume's score, strengths, and suggestions.")

# # Score Display
# st.subheader("🎯 Resume Score")
# st.metric(label="Overall Score", value=f"{data['score']} / 10")

# # Feedback Sections
# col1, col2 = st.columns(2)

# with col1:
#     with st.expander("🛠️ Rule-Based Feedback"):
#         for fb in data.get("rule_feedback", []):
#             st.markdown(f"✅ {fb}" if "missing" not in fb.lower() else f"⚠️ {fb}")

# with col2:
#     with st.expander("🧠 LLM Feedback"):
#         llm_text = data.get("llm_feedback", "")
#         if "Rating:" in llm_text:
#             rating_part = llm_text.split("Rating:")[1]
#             st.markdown("**" + rating_part.strip() + "**")
#         else:
#             st.markdown(llm_text)

# # Skills
# with st.expander("🧪 Skills"):
#     skills = data.get("skills", [])
#     if skills:
#         st.markdown(", ".join(f"`{s}`" for s in skills))
#     else:
#         st.warning("No skills found.")

# # Education Section
# with st.expander("🎓 Education"):
#     edu = data.get("education_regex", "")
#     st.markdown(f"```\n{edu.strip()}\n```")

# # Projects Section
# with st.expander("💼 Projects"):
#     projects = data.get("sections_regex", {}).get("Projects", "")
#     st.markdown(f"```\n{projects.strip()}\n```")

# # Optional: JSON Viewer
# with st.expander("📄 View Raw JSON"):
#     st.json(data)

# # Optional: Download Button
# st.download_button(
#     label="📥 Download Structured Resume JSON",
#     data=json.dumps(data, indent=2),
#     file_name="structured_resume.json",
#     mime="application/json"
# )
