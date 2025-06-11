# scoring/scorer.py
import requests
import numpy as np
import os
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file at the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
print(f"Hugging Face Token Loaded: {'Yes' if HF_TOKEN else 'No'}")


class ResumeScorerAPI:
    """
    Scores resume similarity using the correct Hugging Face Inference Providers API
    with TF-IDF fallback for reliability.
    """
    def __init__(self, model_id="thenlper/gte-large"):
        self.model_id = model_id
        # Use the correct Inference Providers endpoint
        self.api_url = "https://api-inference.huggingface.co/models/" + model_id
        self.headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Test models in order of preference (feature extraction models)
        self.test_models = [
            "intfloat/multilingual-e5-large-instruct"
        ]
        
        self.working_model = None
        self._find_working_model()

    def _find_working_model(self):
        """Find a working feature extraction model."""
        if not HF_TOKEN:
            print("No HF token available, using TF-IDF")
            return
        
        for model in self.test_models:
            try:
                print(f"Testing model: {model}")
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                
                response = requests.post(
                    api_url,
                    headers=self.headers,
                    json={
                        "inputs": "Test sentence for model verification",
                        "options": {"wait_for_model": True}
                    },
                    timeout=20
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Check if we got a valid embedding response
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], (list, float, int)):
                            self.working_model = model
                            self.api_url = api_url
                            print(f"✅ Successfully connected to: {model}")
                            return
                
                print(f"❌ Model {model} response: {response.status_code}")
                
            except Exception as e:
                print(f"❌ Model {model} failed: {e}")
                continue
        
        print("No working HF models found, will use TF-IDF similarity")

    def get_embedding(self, text: str, max_retries=2):
        """
        Get embedding using the correct HF Inference Providers API.
        """
        if not self.working_model or not text.strip():
            return None
        
        # Clean and limit text
        clean_text = re.sub(r'\s+', ' ', text.strip())
        words = clean_text.split()
        if len(words) > 300:  # Conservative limit
            clean_text = ' '.join(words[:300])
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "inputs": clean_text,
                        "options": {"wait_for_model": True}
                    },
                    timeout=25
                )
                
                if response.status_code == 503:
                    print(f"Model loading, waiting... (attempt {attempt + 1})")
                    time.sleep(3)
                    continue
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats for embeddings
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            # 2D array - flatten or take first/mean
                            embeddings = np.array(result)
                            if embeddings.ndim == 2:
                                return np.mean(embeddings, axis=0)  # Mean pooling
                            return embeddings.flatten()
                        elif isinstance(result[0], (int, float)):
                            # 1D array of numbers
                            return np.array(result)
                    
                    print(f"Unexpected response format: {type(result)}")
                    
                else:
                    print(f"API Error {response.status_code}: {response.text[:200]}")
                    
            except Exception as e:
                print(f"Embedding request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return None

    def score_similarity(self, resume_text, job_description):
        """
        Calculate similarity using the best available method.
        """
        # Try HF API first
        if self.working_model:
            print("Attempting HF API similarity...")
            hf_score = self._hf_similarity(resume_text, job_description)
            if hf_score is not None and hf_score > 0:
                print(f"✅ HF API similarity: {hf_score}%")
                return hf_score
        
        # Fallback to TF-IDF
        print("Using TF-IDF similarity...")
        return self._tfidf_similarity(resume_text, job_description)

    def _hf_similarity(self, resume_text, job_description):
        """
        Calculate similarity using HF embeddings.
        """
        resume_emb = self.get_embedding(resume_text)
        job_emb = self.get_embedding(job_description)
        
        if resume_emb is None or job_emb is None:
            print("Could not generate HF embeddings")
            return None
        
        # Ensure same dimensions
        resume_emb = np.array(resume_emb).flatten()
        job_emb = np.array(job_emb).flatten()
        
        if len(resume_emb) != len(job_emb):
            print(f"Embedding dimension mismatch: {len(resume_emb)} vs {len(job_emb)}")
            return None
        
        # Calculate cosine similarity
        similarity = np.dot(resume_emb, job_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(job_emb))
        return round(max(0, float(similarity)) * 100, 2)

    def _tfidf_similarity(self, resume_text, job_description):
        """
        Calculate similarity using TF-IDF and cosine similarity.
        This is our reliable fallback method.
        """
        try:
            # Preprocessing
            def preprocess_text(text):
                # Remove extra whitespace and convert to lowercase
                text = re.sub(r'\s+', ' ', text.lower().strip())
                # Remove special characters but keep letters, numbers, and spaces
                text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
                return text
            
            resume_clean = preprocess_text(resume_text)
            job_clean = preprocess_text(job_description)
            
            if not resume_clean.strip() or not job_clean.strip():
                return 0.0
            
            # Create TF-IDF vectors with optimized parameters
            vectorizer = TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for better matching
                min_df=1,
                max_df=0.95,
                sublinear_tf=True  # Use sublinear tf scaling
            )
            
            # Fit and transform both texts
            tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            similarity_score = similarity_matrix[0, 1]
            
            return round(max(0, similarity_score) * 100, 2)
            
        except Exception as e:
            print(f"TF-IDF similarity calculation failed: {e}")
            return 0.0
        

# import requests
# import numpy as np
# import os
# import re
# import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from dotenv import load_dotenv

# # Load environment variables from .env file at the project root
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
# HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
# print(f"Hugging Face Token Loaded: {'Yes' if HF_TOKEN else 'No'}")


# class ResumeScorerAPI:
#     """
#     Scores resume similarity using the correct Hugging Face Inference Providers API
#     with TF-IDF fallback for reliability.
#     """
#     def __init__(self, model_id="thenlper/gte-large"):
#         self.model_id = model_id
#         # Use the correct Inference Providers endpoint
#         self.api_url = "https://api-inference.huggingface.co/models/" + model_id
#         self.headers = {
#             "Authorization": f"Bearer {HF_TOKEN}",
#             "Content-Type": "application/json"
#         }
        
#         # Test models in order of preference (feature extraction models)
#         self.test_models = [
#             "thenlper/gte-large",
#             "intfloat/multilingual-e5-large-instruct", 
#             "sentence-transformers/all-mpnet-base-v2",
#             "microsoft/mpnet-base"
#         ]
        
#         self.working_model = None
#         self._find_working_model()

#     def _find_working_model(self):
#         """Find a working feature extraction model."""
#         if not HF_TOKEN:
#             print("No HF token available, using TF-IDF")
#             return
        
#         for model in self.test_models:
#             try:
#                 print(f"Testing model: {model}")
#                 api_url = f"https://api-inference.huggingface.co/models/{model}"
                
#                 response = requests.post(
#                     api_url,
#                     headers=self.headers,
#                     json={
#                         "inputs": "Test sentence for model verification",
#                         "options": {"wait_for_model": True}
#                     },
#                     timeout=20
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     # Check if we got a valid embedding response
#                     if isinstance(result, list) and len(result) > 0:
#                         if isinstance(result[0], (list, float, int)):
#                             self.working_model = model
#                             self.api_url = api_url
#                             print(f"✅ Successfully connected to: {model}")
#                             return
                
#                 print(f"❌ Model {model} response: {response.status_code}")
                
#             except Exception as e:
#                 print(f"❌ Model {model} failed: {e}")
#                 continue
        
#         print("No working HF models found, will use TF-IDF similarity")

#     def get_embedding(self, text: str, max_retries=2):
#         """
#         Get embedding using the correct HF Inference Providers API.
#         """
#         if not self.working_model or not text.strip():
#             return None
        
#         # Clean and limit text
#         clean_text = re.sub(r'\s+', ' ', text.strip())
#         words = clean_text.split()
#         if len(words) > 300:  # Conservative limit
#             clean_text = ' '.join(words[:300])
        
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     self.api_url,
#                     headers=self.headers,
#                     json={
#                         "inputs": clean_text,
#                         "options": {"wait_for_model": True}
#                     },
#                     timeout=25
#                 )
                
#                 if response.status_code == 503:
#                     print(f"Model loading, waiting... (attempt {attempt + 1})")
#                     time.sleep(3)
#                     continue
                
#                 if response.status_code == 200:
#                     result = response.json()
                    
#                     # Handle different response formats for embeddings
#                     if isinstance(result, list) and len(result) > 0:
#                         if isinstance(result[0], list):
#                             # 2D array - flatten or take first/mean
#                             embeddings = np.array(result)
#                             if embeddings.ndim == 2:
#                                 return np.mean(embeddings, axis=0)  # Mean pooling
#                             return embeddings.flatten()
#                         elif isinstance(result[0], (int, float)):
#                             # 1D array of numbers
#                             return np.array(result)
                    
#                     print(f"Unexpected response format: {type(result)}")
                    
#                 else:
#                     print(f"API Error {response.status_code}: {response.text[:200]}")
                    
#             except Exception as e:
#                 print(f"Embedding request failed (attempt {attempt + 1}): {e}")
#                 if attempt < max_retries - 1:
#                     time.sleep(2)
        
#         return None

#     def score_similarity(self, resume_text, job_description):
#         """
#         Calculate similarity using the best available method.
#         """
#         # Try HF API first
#         if self.working_model:
#             print("Attempting HF API similarity...")
#             hf_score = self._hf_similarity(resume_text, job_description)
#             if hf_score is not None and hf_score > 0:
#                 print(f"✅ HF API similarity: {hf_score}%")
#                 return hf_score
        
#         # Fallback to TF-IDF
#         print("Using TF-IDF similarity...")
#         return self._tfidf_similarity(resume_text, job_description)

#     def _hf_similarity(self, resume_text, job_description):
#         """
#         Calculate similarity using HF embeddings.
#         """
#         resume_emb = self.get_embedding(resume_text)
#         job_emb = self.get_embedding(job_description)
        
#         if resume_emb is None or job_emb is None:
#             print("Could not generate HF embeddings")
#             return None
        
#         # Ensure same dimensions
#         resume_emb = np.array(resume_emb).flatten()
#         job_emb = np.array(job_emb).flatten()
        
#         if len(resume_emb) != len(job_emb):
#             print(f"Embedding dimension mismatch: {len(resume_emb)} vs {len(job_emb)}")
#             return None
        
#         # Calculate cosine similarity
#         similarity = np.dot(resume_emb, job_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(job_emb))
#         return round(max(0, float(similarity)) * 100, 2)

#     def _tfidf_similarity(self, resume_text, job_description):
#         """
#         Calculate similarity using TF-IDF and cosine similarity.
#         This is our reliable fallback method.
#         """
#         try:
#             # Preprocessing
#             def preprocess_text(text):
#                 # Remove extra whitespace and convert to lowercase
#                 text = re.sub(r'\s+', ' ', text.lower().strip())
#                 # Remove special characters but keep letters, numbers, and spaces
#                 text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
#                 return text
            
#             resume_clean = preprocess_text(resume_text)
#             job_clean = preprocess_text(job_description)
            
#             if not resume_clean.strip() or not job_clean.strip():
#                 return 0.0
            
#             # Create TF-IDF vectors with optimized parameters
#             vectorizer = TfidfVectorizer(
#                 max_features=2000,
#                 stop_words='english',
#                 ngram_range=(1, 3),  # Include trigrams for better matching
#                 min_df=1,
#                 max_df=0.95,
#                 sublinear_tf=True  # Use sublinear tf scaling
#             )
            
#             # Fit and transform both texts
#             tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
            
#             # Calculate cosine similarity
#             similarity_matrix = cosine_similarity(tfidf_matrix)
#             similarity_score = similarity_matrix[0, 1]
            
#             return round(max(0, similarity_score) * 100, 2)
            
#         except Exception as e:
#             print(f"TF-IDF similarity calculation failed: {e}")
#             return 0.0