#LLM BASED EXTRACTION!!
#THIS IS OPTIONAL.FINALIZE LATER



# import re
# import requests
# import os
# import time

# # Load your API key from an environment variable
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# # Try multiple working models as fallbacks
# MODELS = [
#     "microsoft/DialoGPT-medium",
#     "google/flan-t5-base", 
#     "facebook/blenderbot-400M-distill",
#     "mistralai/Mistral-7B-Instruct-v0.1"
# ]

# HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# SKILL_KEYWORDS = {
#     'Python', 'Java', 'JavaScript', 'C++', 'C', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin',
#     'Machine Learning', 'Deep Learning', 'Data Science', 'AI', 'NLP',
#     'HTML', 'CSS', 'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'MySQL',
#     'React', 'Vue', 'Angular', 'Node.js', 'Express', 'Django', 'Flask', 'Spring',
#     'Pandas', 'NumPy', 'TensorFlow', 'PyTorch', 'Scikit-learn',
#     'Git', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP',
#     'Linux', 'Unix', 'Windows', 'MacOS',
#     'Communication', 'Teamwork', 'Leadership', 'Problem Solving', 'Critical Thinking',
#     'Project Management', 'Agile', 'Scrum', 'DevOps', 'CI/CD'
# }

# def extract_skills_rule_based(text):
#     """Extract skills using rule-based keyword matching"""
#     extracted = []
#     for skill in SKILL_KEYWORDS:
#         pattern = r'\b' + re.escape(skill) + r'\b'
#         if re.search(pattern, text, re.IGNORECASE):
#             extracted.append(skill)
#     return list(set(extracted))

# def query_huggingface(prompt: str, model_name: str):
#     """Query Hugging Face API for a specific model"""
#     api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
#     payload = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": 150,
#             "temperature": 0.3,
#             "return_full_text": False,
#             "do_sample": True
#         },
#     }
    
#     try:
#         response = requests.post(api_url, headers=HEADERS, json=payload, timeout=30)
        
#         if response.status_code == 200:
#             result = response.json()
#             if isinstance(result, list) and len(result) > 0:
#                 return result[0].get("generated_text", "")
#             elif isinstance(result, dict):
#                 return result.get("generated_text", "")
#             return ""
            
#         elif response.status_code == 503:
#             print(f"Model {model_name} is loading, waiting...")
#             time.sleep(10)  # Wait for model to load
#             # Try once more
#             response = requests.post(api_url, headers=HEADERS, json=payload, timeout=30)
#             if response.status_code == 200:
#                 result = response.json()
#                 if isinstance(result, list) and len(result) > 0:
#                     return result[0].get("generated_text", "")
#             return ""
            
#         else:
#             print(f"Error with {model_name}: {response.status_code}")
#             return ""
            
#     except Exception as e:
#         print(f"Request failed for {model_name}: {e}")
#         return ""

# def extract_skills_llm(text):
#     """Extract skills using LLM with multiple model fallbacks"""
    
#     # Limit text length to avoid token limits
#     text = text[:1000] if len(text) > 1000 else text
    
#     # Create a focused prompt
#     prompt = f"""Please extract all technical and soft skills from this resume text:

# {text}

# List the skills in this format:
# Skills: Python, JavaScript, Machine Learning, Communication, Leadership

# Skills:"""

#     # Try each model until one works
#     for model_name in MODELS:
#         print(f"Trying model: {model_name}")
        
#         output = query_huggingface(prompt, model_name)
        
#         if output and output.strip():
#             print(f"Got response from {model_name}")
#             print(f"Raw output: {output[:200]}...")
            
#             # Parse the skills from output
#             skills = parse_skills_from_output(output)
#             if skills:
#                 return skills
#         else:
#             print(f"No response from {model_name}")
    
#     print("All LLM models failed, returning empty list")
#     return []

# def parse_skills_from_output(output):
#     """Parse skills from LLM output"""
#     skills = []
    
#     # Clean the output
#     output = output.strip()
    
#     # Try different parsing strategies
#     if "Skills:" in output:
#         # Extract everything after "Skills:"
#         skills_text = output.split("Skills:")[-1].strip()
#         skills = [s.strip().strip('.,;') for s in skills_text.split(',')]
#     elif "skills:" in output.lower():
#         # Case insensitive
#         skills_text = output.lower().split("skills:")[-1].strip()
#         skills = [s.strip().strip('.,;') for s in skills_text.split(',')]
#     else:
#         # Try to extract comma-separated items from the whole output
#         potential_skills = [s.strip().strip('.,;') for s in output.split(',')]
#         # Filter for reasonable skill names
#         skills = [s for s in potential_skills if 2 <= len(s) <= 30 and not s.lower().startswith('the')]
    
#     # Clean and validate skills
#     cleaned_skills = []
#     for skill in skills:
#         skill = skill.strip().title()
#         # Remove common non-skill words
#         if (skill and 
#             len(skill) > 1 and 
#             len(skill) < 50 and
#             not skill.lower() in ['and', 'or', 'the', 'a', 'an', 'with', 'using', 'for']):
#             cleaned_skills.append(skill)
    
#     return cleaned_skills[:15]  # Limit to 15 skills

# # Alternative simpler LLM extraction if the main one fails
# def extract_skills_simple_llm(text):
#     """Simpler LLM approach using a basic text generation model"""
    
#     # Use a very simple prompt
#     prompt = f"Skills in this resume: {text[:500]}\n\nSkills found:"
    
#     model_name = "microsoft/DialoGPT-medium"
#     output = query_huggingface(prompt, model_name)
    
#     if output:
#         # Extract potential skills
#         words = re.findall(r'\b[A-Za-z][A-Za-z+#.]*\b', output)
#         # Filter against known skills
#         found_skills = []
#         for word in words:
#             for skill in SKILL_KEYWORDS:
#                 if word.lower() == skill.lower():
#                     found_skills.append(skill)
#         return list(set(found_skills))
    
#     return []





# import re
# import requests
# import os

# # Load your API key from an environment variable or replace with your token (not recommended)
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# print(HF_API_TOKEN)

# API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
# HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# SKILL_KEYWORDS = {
#     'Python', 'Java', 'JavaScript', 'C++', 'C', 'Machine Learning', 'Deep Learning',
#     'HTML', 'CSS', 'SQL', 'React', 'Node.js', 'Pandas', 'NumPy', 'Git', 'Linux',
#     'Communication', 'Teamwork', 'Leadership', 'Problem Solving', 'Critical Thinking'
# }

# def extract_skills_rule_based(text):
#     extracted = []
#     for skill in SKILL_KEYWORDS:
#         pattern = r'\b' + re.escape(skill) + r'\b'
#         if re.search(pattern, text, re.IGNORECASE):
#             extracted.append(skill)
#     return list(set(extracted))

# def query_huggingface(prompt: str):
#     payload = {
#         "inputs": prompt,
#         "parameters": {"max_new_tokens": 100, "temperature": 0.7},
#     }
#     response = requests.post(API_URL, headers=HEADERS, json=payload)
#     if response.status_code == 200:
#         generated = response.json()[0]["generated_text"]
#         return generated
#     else:
#         print("Error from Hugging Face API:", response.status_code, response.text)
#         return ""

# def extract_skills_llm(text):
#     prompt = (
#         "Extract all the technical and soft skills mentioned in the resume text below:\n\n"
#         f"{text}\n\n"
#         "Return the skills as a comma-separated list."
#     )
#     output = query_huggingface(prompt)
#     skills_part = output.split("Return the skills as a comma-separated list.")[-1]
#     skills = [s.strip() for s in skills_part.split(",") if s.strip()]
#     return skills
