import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL")

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

    try:
        response = requests.post(url, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"]
    except Exception as e:
        print("LLM scoring failed:", e)
        return "LLM scoring unavailable."
