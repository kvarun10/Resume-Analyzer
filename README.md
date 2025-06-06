# Resume-Analyzer

✅ Day 1 – PDF Text Extraction

Tool: PyMuPDF (fitz)

You implemented extract_text(pdf_path) to load and extract raw text from PDF resumes.

Output: Raw resume content as a string.

✅ Day 2 – Text Preprocessing

Tool: NLTK

You cleaned and tokenized text for consistent processing.

Removed punctuation, converted to lowercase, removed stopwords.

Output: List of clean tokens for downstream tasks.

✅ Day 3 – Rule-Based Information Extraction

Custom class: RuleBasedExtractor

Used regex and keyword matching to extract:

Name (heuristic: first line, capitalized)

Email (regex)

Phone number (regex, Indian format)

Skills (from predefined skills database)

Education (based on education keywords)

Output: Dictionary of structured extracted fields

✅ Day 4 – spaCy NER Extraction

Tool: spaCy

You loaded spaCy’s pretrained NER pipeline to extract:

PERSON, ORG, LOC, DATE, etc.

Used to validate rule-based results or find additional entities.

Output: spaCy entity dictionary stored under spacy_entities in JSON

✅ Day 5 – LLM-Based NER Extraction (Optional, On Hold)

Tool: Hugging Face Transformers (e.g., BERT, Mistral)

You set up extract_entities_from_api() to call the Hugging Face inference API

Extracted named entities like PERSON, ORG, LOC, MISC via model prediction

You added the extracted LLM-based entities to JSON under llm_entities but paused integrating it into the main pipeline for now.

🧠 Remaining Plan Overview (Days 6–9)

🟡 Day 6 – Resume Scoring using Transformers (Semantic Similarity)

Goal: Compare resume to job description for relevance

Tool: SentenceTransformers (e.g., all-MiniLM, BERT)

You’ll compute cosine similarity between resume content and job description

Optionally: Normalize this similarity score to a scale of 0–10 or 0–100

Why it matters: This shows how well a resume matches a specific job, very useful for recruiters or auto-screening.

🟡 Day 7 – Resume Improvement Suggestions (LLM-Powered)

Goal: Prompt an LLM to review the resume and suggest improvements.

Tool: OpenAI API / Hugging Face LLMs (GPT-3, GPT-2, Mistral)

Sample prompt: "Suggest improvements to the following resume..."

Why it matters: Helps users enhance their resume content with actionable suggestions.

🟡 Day 8 – Job Matching using Transformers

Goal: Match resume to job description using semantic relevance

Tool: Similar to Day 6, but enhanced by:

Skill matching

Named Entity cross-reference (e.g., ORG, TITLE)

Output: Top-matching jobs or detailed job-resume fit explanation

Why it matters: Allows your analyzer to act as a career recommendation engine.

🟢 Day 9 – Final Polish + UI (Optional)

Tools: Streamlit, Gradio, or CLI

Upload resume → view extracted info, score, suggestions

Bonus: Export feedback/report as PDF
