import requests
import json

HF_API_TOKEN = "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"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def rule_based_score(data):
    score = 0
    feedback = []

    if data.get("spaCy_entities", {}).get("NAME"):
        score += 1
    else:
        feedback.append("Name not detected.")

    if data.get("spaCy_entities", {}).get("EMAILS"):
        score += 1
    else:
        feedback.append("Email not found.")

    if data.get("spaCy_entities", {}).get("PHONES"):
        score += 1
    else:
        feedback.append("Phone number missing.")

    if data.get("education_regex"):
        score += 1
    else:
        feedback.append("Education section seems missing.")

    if data.get("experience_regex"):
        score += 1
    else:
        feedback.append("Experience section seems weak or missing.")

    if len(data.get("skills", [])) >= 4:
        score += 2
    else:
        feedback.append("Consider adding more skills.")

    if len(data.get("structured_llm", {}).get("projects", "")) > 10:
        score += 1
    else:
        feedback.append("Add more project detail.")

    if len(data.get("structured_llm", {}).get("education", "")) > 10:
        score += 1

    final_score = min(score, 10)
    return final_score, feedback

def llm_based_score(text):
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You're an expert resume reviewer.
1. Rate this resume out of 10.
2. Briefly explain the reasoning.
3. Suggest 2–3 improvements.

Resume:
{text}
"""

    response = requests.post(url, headers=headers, json={"inputs": prompt})

    try:
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"]
    except Exception as e:
        print("LLM scoring failed:", e)
        return "LLM scoring unavailable."



# """
# Comprehensive Resume Scorer using NLP-extracted data
# Provides detailed scoring across multiple categories
# """

# import re
# from textblob import TextBlob
# from collections import Counter
# import math

# class ResumeScorer:
#     def __init__(self):
#         """Initialize scoring criteria and weights"""
        
#         # Scoring weights (total should be 100)
#         self.weights = {
#             'contact_info': 15,      # Contact information completeness
#             'technical_skills': 25,  # Technical skills diversity and relevance
#             'experience': 20,        # Work experience quality
#             'education': 15,         # Educational background
#             'projects': 15,          # Project portfolio
#             'presentation': 10       # Resume presentation and quality
#         }
        
#         # Skill categories and their values
#         self.skill_categories = {
#             'programming': ['Python', 'Java', 'JavaScript', 'C++', 'C#', 'Go', 'Rust', 'Swift', 'TypeScript'],
#             'web_dev': ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'HTML', 'CSS'],
#             'database': ['MySQL', 'PostgreSQL', 'MongoDB', 'SQLite', 'Oracle', 'Redis'],
#             'cloud': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes'],
#             'data_science': ['Machine Learning', 'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Scikit-learn'],
#             'tools': ['Git', 'Jenkins', 'Terraform', 'Ansible', 'Maven', 'Gradle']
#         }
        
#         # Experience keywords for quality assessment
#         self.experience_keywords = {
#             'leadership': ['lead', 'manage', 'supervise', 'coordinate', 'direct', 'head'],
#             'development': ['develop', 'build', 'create', 'implement', 'design', 'architect'],
#             'impact': ['improve', 'optimize', 'increase', 'reduce', 'enhance', 'streamline'],
#             'collaboration': ['collaborate', 'team', 'cross-functional', 'stakeholder', 'communicate']
#         }
    
#     def score_contact_info(self, data):
#         """Score contact information completeness"""
#         score = 0
#         max_score = self.weights['contact_info']
#         criteria = []
        
#         # Email (required) - 40% of contact score
#         if data.get('emails'):
#             score += max_score * 0.4
#             criteria.append("✅ Email provided")
#         else:
#             criteria.append("❌ Email missing")
        
#         # Phone (important) - 30% of contact score
#         if data.get('phones'):
#             score += max_score * 0.3
#             criteria.append("✅ Phone number provided")
#         else:
#             criteria.append("❌ Phone number missing")
        
#         # LinkedIn (professional) - 20% of contact score
#         if data.get('linkedin'):
#             score += max_score * 0.2
#             criteria.append("✅ LinkedIn profile provided")
#         else:
#             criteria.append("❌ LinkedIn profile missing")
        
#         # GitHub/Portfolio (technical) - 10% of contact score
#         if data.get('github') or data.get('websites'):
#             score += max_score * 0.1
#             criteria.append("✅ GitHub/Portfolio provided")
#         else:
#             criteria.append("❌ GitHub/Portfolio missing")
        
#         return {
#             'score': round(score, 1),
#             'max_score': max_score,
#             'criteria': criteria
#         }
    
#     def score_technical_skills(self, data):
#         """Score technical skills diversity and relevance"""
#         skills = data.get('skills', [])
#         score = 0
#         max_score = self.weights['technical_skills']
#         criteria = []
        
#         if not skills:
#             criteria.append("❌ No technical skills detected")
#             return {
#                 'score': 0,
#                 'max_score': max_score,
#                 'criteria': criteria
#             }
        
#         # Count skills in each category
#         skill_categories_found = {}
#         total_skills = len(skills)
        
#         for category, skill_list in self.skill_categories.items():
#             found_skills = [skill for skill in skills if skill in skill_list]
#             skill_categories_found[category] = found_skills
        
#         # Diversity bonus (having skills across multiple categories)
#         categories_with_skills = sum(1 for skills_in_cat in skill_categories_found.values() if skills_in_cat)
#         diversity_score = min(categories_with_skills * 3, 15)  # Max 15 points for diversity
        
#         # Quantity score (number of skills)
#         if total_skills >= 10:
#             quantity_score = 10
#             criteria.append(f"✅ Excellent skill count ({total_skills} skills)")
#         elif total_skills >= 6:
#             quantity_score = 7
#             criteria.append(f"✅ Good skill count ({total_skills} skills)")
#         elif total_skills >= 3:
#             quantity_score = 5
#             criteria.append(f"⚠️ Moderate skill count ({total_skills} skills)")
#         else:
#             quantity_score = 2
#             criteria.append(f"❌ Limited skill count ({total_skills} skills)")
        
#         # Category-specific scoring
#         for category, found_skills in skill_
