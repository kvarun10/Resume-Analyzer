# scoring/rater.py
import os
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load your Hugging Face token from .env
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class ResumeRaterAPI:
    def __init__(self, model="google/gemma-2b-it"):
        self.client = InferenceClient(
            provider="together",
            api_key=HF_TOKEN,
        )
        self.model = model

    def rate_resume(self, resume_text: str):
        prompt = f"""
You are an expert resume reviewer.

Rate the following resume on a scale from 0 to 10 based on:
- Clarity and formatting
- Technical and soft skills
- Relevance to modern job market
- Overall presentation

Only output a single number between 0 and 10 with up to one decimal place. Do not provide any explanation.

Resume:
\"\"\"
{resume_text}
\"\"\"
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            output = completion.choices[0].message.content.strip()
            print("🔎 Raw model output:", output)

            # Extract a number from the output
            match = re.search(r"\b\d{1,2}(\.\d)?\b", output)
            if match:
                return float(match.group())

            print("❌ Rating format not recognized.")
            return None

        except Exception as e:
            print(f"❌ Error in resume rating: {e}")
            return None
