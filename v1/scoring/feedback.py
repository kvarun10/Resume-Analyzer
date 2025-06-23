# feedback.py (markdown-based feedback using Novita + LLaMA-3)
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

client = InferenceClient(provider="novita", api_key=HF_TOKEN)

def get_strength_from_resume_text(resume_text: str):
    prompt = f"""
You are a professional resume reviewer.

Please analyze the following resume and provide feedback in markdown format with the following sections:

## Strengths
- Bullet points

The feedback should be concise and within 5 bullet points. Also dont give the answer any heading like "Strengths", just start with the bullet points.
Resume:
\"\"\"
{resume_text}
\"\"\"
"""

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Exception while generating feedback: {e}")
        return None

def get_weakness_from_resume_text(resume_text: str):
    prompt = f"""
You are a professional resume reviewer.

Please analyze the following resume and provide feedback in markdown format with the following sections:

## Weaknesses
- Bullet points

The feedback should be concise and within 5 bullet points. Also dont give the answer any heading like "Weaknesses", just start with the bullet points.
Resume:
\"\"\"
{resume_text}
\"\"\"
"""

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Exception while generating feedback: {e}")
        return None



def get_feedback_from_resume_text(resume_text: str):
    prompt = f"""
You are a professional resume reviewer.

Please analyze the following resume and provide feedback in markdown format with the following sections:

## Suggestions for Improvement
- Bullet points

The analysis should be specific and the suggestions for improvement should also pinpoint the corresponding area of weakness in the resume.

The feedback should be concise and within 5 bullet points. Also dont give the answer any heading like "Suggestions", just start with the bullet points.
Resume:
\"\"\"
{resume_text}
\"\"\"
"""

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Exception while generating feedback: {e}")
        return None
