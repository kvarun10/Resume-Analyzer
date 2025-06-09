


# """
# NLP-based Resume Information Extractor
# Uses spaCy, NLTK, and other NLP techniques for intelligent text analysis
# """

# import re
# import spacy
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.chunk import ne_chunk
# from nltk.tag import pos_tag
# from textblob import TextBlob
# from collections import Counter
# import logging

# # Download required NLTK data
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('averaged_perceptron_tagger', quiet=True)
#     nltk.download('maxent_ne_chunker', quiet=True)
#     nltk.download('words', quiet=True)
#     nltk.download('stopwords', quiet=True)
# except:
#     pass

# class NLPResumeExtractor:
#     def __init__(self):
#         """Initialize NLP models and resources"""
#         try:
#             # Load spaCy model
#             self.nlp = spacy.load("en_core_web_sm")
#             print("✅ spaCy model loaded successfully")
#         except OSError:
#             print("❌ spaCy model not found. Please install with:")
#             print("python -m spacy download en_core_web_sm")
#             self.nlp = None
        
#         # Initialize NLTK stopwords
#         try:
#             self.stop_words = set(stopwords.words('english'))
#         except:
#             self.stop_words = set()
        
#         # Technical skills database (expanded)
#         self.tech_skills = {
#             'programming_languages': [
#                 'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C',
#                 'Go', 'Rust', 'Swift', 'Kotlin', 'Scala', 'Ruby', 'PHP',
#                 'R', 'MATLAB', 'Perl', 'Haskell', 'Clojure', 'Dart'
#             ],
#             'web_technologies': [
#                 'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js',
#                 'Express.js', 'Django', 'Flask', 'FastAPI', 'Spring Boot',
#                 'ASP.NET', 'jQuery', 'Bootstrap', 'Tailwind CSS', 'SASS', 'LESS'
#             ],
#             'databases': [
#                 'MySQL', 'PostgreSQL', 'MongoDB', 'SQLite', 'Oracle',
#                 'SQL Server', 'Redis', 'Cassandra', 'DynamoDB', 'Elasticsearch'
#             ],
#             'cloud_platforms': [
#                 'AWS', 'Azure', 'Google Cloud', 'GCP', 'Heroku', 'DigitalOcean',
#                 'Vercel', 'Netlify', 'Firebase'
#             ],
#             'tools_frameworks': [
#                 'Git', 'Docker', 'Kubernetes', 'Jenkins', 'Terraform',
#                 'Ansible', 'Maven', 'Gradle', 'Webpack', 'Babel'
#             ],
#             'data_science': [
#                 'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch',
#                 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn',
#                 'Jupyter', 'Apache Spark', 'Hadoop', 'Tableau', 'Power BI'
#             ]
#         }
        
#         # Flatten skills for easier searching
#         self.all_skills = []
#         for category in self.tech_skills.values():
#             self.all_skills.extend(category)
    
#     def extract_contact_info(self, text):
#         """Extract contact information using regex and NLP"""
#         contact_info = {
#             'emails': [],
#             'phones': [],
#             'linkedin': None,
#             'github': None,
#             'websites': []
#         }
        
#         # Email extraction
#         email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#         contact_info['emails'] = list(set(re.findall(email_pattern, text)))
        
#         # Phone extraction (multiple formats)
#         phone_patterns = [
#             r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
#             r'(\+\d{1,3}[-.\s]?)?\d{10}',
#             r'(\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
#         ]
#         phones = []
#         for pattern in phone_patterns:
#             phones.extend(re.findall(pattern, text))
#         contact_info['phones'] = list(set([phone[0] + phone[1] if isinstance(phone, tuple) else phone for phone in phones]))
        
#         # Social media and websites
#         linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/pub/)[A-Za-z0-9-]+'
#         github_pattern = r'(?:github\.com/)[A-Za-z0-9-]+'
#         website_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'
        
#         linkedin_matches = re.findall(linkedin_pattern, text)
#         contact_info['linkedin'] = linkedin_matches[0] if linkedin_matches else None
        
#         github_matches = re.findall(github_pattern, text)
#         contact_info['github'] = github_matches[0] if github_matches else None
        
#         contact_info['websites'] = list(set(re.findall(website_pattern, text)))
        
#         return contact_info
    
#     def extract_name(self, text):
#         """Extract name using NLP named entity recognition"""
#         if not self.nlp:
#             # Fallback: assume first line contains name
#             lines = text.strip().split('\n')
#             return lines[0].strip() if lines else "Unknown"
        
#         doc = self.nlp(text[:500])  # Check first 500 characters
        
#         # Look for PERSON entities
#         for ent in doc.ents:
#             if ent.label_ == "PERSON":
#                 return ent.text.strip()
        
#         # Fallback to NLTK
#         try:
#             sentences = sent_tokenize(text[:300])
#             if sentences:
#                 tokens = word_tokenize(sentences[0])
#                 pos_tags = pos_tag(tokens)
#                 chunks = ne_chunk(pos_tags)
                
#                 for chunk in chunks:
#                     if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
#                         return ' '.join([token for token, pos in chunk])
#         except:
#             pass
        
#         # Final fallback
#         lines = text.strip().split('\n')
#         return lines[0].strip() if lines else "Unknown"
    
#     def extract_education(self, text):
#         """Extract education information using NLP"""
#         education = []
        
#         # Education section patterns
#         edu_section_pattern = r'(?i)(?:education|academic|qualification|degree).*?(?=(?:experience|work|employment|skill|project|certification)|$)'
#         edu_match = re.search(edu_section_pattern, text, re.DOTALL)
        
#         if edu_match:
#             edu_text = edu_match.group()
#         else:
#             edu_text = text
        
#         # Degree patterns
#         degree_patterns = [
#             r'(?i)(bachelor|master|phd|doctorate|b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?a|m\.?a|b\.?com|m\.?com|mba|bba)[\s\w]*',
#             r'(?i)(undergraduate|graduate|postgraduate)[\s\w]*'
#         ]
        
#         # Institution patterns
#         institution_patterns = [
#             r'(?i)(?:university|college|institute|iit|nit|iiit)[\s\w]*',
#             r'(?i)(?:motilal nehru|indian institute|national institute)[\s\w]*'
#         ]
        
#         # Extract degrees
#         for pattern in degree_patterns:
#             degrees = re.findall(pattern, edu_text)
#             education.extend([deg.strip() for deg in degrees if len(deg.strip()) > 2])
        
#         # Extract institutions
#         for pattern in institution_patterns:
#             institutions = re.findall(pattern, edu_text)
#             education.extend([inst.strip() for inst in institutions if len(inst.strip()) > 5])
        
#         # Use spaCy for organization detection
#         if self.nlp:
#             doc = self.nlp(edu_text)
#             for ent in doc.ents:
#                 if ent.label_ == "ORG" and any(keyword in ent.text.lower() for keyword in ['university', 'college', 'institute']):
#                     education.append(ent.text.strip())
        
#         return list(set(education))
    
#     def extract_experience(self, text):
#         """Extract work experience using NLP"""
#         experience = []
        
#         # Experience section pattern
#         exp_section_pattern = r'(?i)(?:experience|employment|work|career|professional).*?(?=(?:education|skill|project|certification)|$)'
#         exp_match = re.search(exp_section_pattern, text, re.DOTALL)
        
#         if exp_match:
#             exp_text = exp_match.group()
#         else:
#             exp_text = text
        
#         # Job title patterns
#         job_patterns = [
#             r'(?i)(software engineer|developer|programmer|analyst|consultant|manager|intern|associate|senior|junior|lead|architect|designer|scientist|researcher)[\w\s]*',
#             r'(?i)(data scientist|machine learning|ai engineer|devops|full stack|backend|frontend|web developer)[\w\s]*'
#         ]
        
#         # Company patterns
#         company_patterns = [
#             r'(?i)(?:at|@)\s+([A-Z][A-Za-z\s&.,-]+(?:inc|ltd|llc|corp|company|technologies|systems|solutions)?)',
#             r'(?i)(google|microsoft|amazon|apple|facebook|meta|netflix|uber|airbnb|spotify|adobe|oracle|ibm|intel|nvidia|tesla)[\w\s]*'
#         ]
        
#         # Extract job titles
#         for pattern in job_patterns:
#             jobs = re.findall(pattern, exp_text)
#             experience.extend([job.strip() for job in jobs if len(job.strip()) > 3])
        
#         # Extract companies
#         for pattern in company_patterns:
#             companies = re.findall(pattern, exp_text)
#             if isinstance(companies[0], tuple) if companies else False:
#                 companies = [comp[0] for comp in companies]
#             experience.extend([comp.strip() for comp in companies if len(comp.strip()) > 2])
        
#         # Use spaCy for organization detection
#         if self.nlp:
#             doc = self.nlp(exp_text)
#             for ent in doc.ents:
#                 if ent.label_ == "ORG":
#                     experience.append(ent.text.strip())
        
#         return list(set(experience))
    
#     def extract_skills(self, text):
#         """Extract technical skills using NLP and pattern matching"""
#         skills_found = []
        
#         # Convert text to lowercase for case-insensitive matching
#         text_lower = text.lower()
        
#         # Direct skill matching
#         for skill in self.all_skills:
#             if skill.lower() in text_lower:
#                 skills_found.append(skill)
        
#         # Skills section pattern
#         skills_section_pattern = r'(?i)(?:skills?|technologies?|technical|proficient|expertise|competencies).*?(?=(?:experience|education|project|certification)|$)'
#         skills_match = re.search(skills_section_pattern, text, re.DOTALL)
        
#         if skills_match:
#             skills_text = skills_match.group()
            
#             # Extract from skills section
#             for skill in self.all_skills:
#                 if skill.lower() in skills_text.lower():
#                     skills_found.append(skill)
        
#         # Use NLP for additional skill extraction
#         if self.nlp:
#             doc = self.nlp(text)
            
#             # Look for technology-related entities and noun phrases
#             for token in doc:
#                 if token.pos_ in ['NOUN', 'PROPN'] and token.text in self.all_skills:
#                     skills_found.append(token.text)
            
#             # Extract noun phrases that might be skills
#             for chunk in doc.noun_chunks:
#                 chunk_text = chunk.text.strip()
#                 if any(skill.lower() in chunk_text.lower() for skill in self.all_skills):
#                     for skill in self.all_skills:
#                         if skill.lower() in chunk_text.lower():
#                             skills_found.append(skill)
        
#         # Remove duplicates and return
#         return list(set(skills_found))
    
#     def extract_projects(self, text):
#         """Extract project information using NLP"""
#         projects = []
        
#         # Project section pattern
#         project_section_pattern = r'(?i)(?:projects?|portfolio|work samples?).*?(?=(?:experience|education|skill|certification)|$)'
#         project_match = re.search(project_section_pattern, text, re.DOTALL)
        
#         if project_match:
#             project_text = project_match.group()
#         else:
#             project_text = text
        
#         # Project title patterns
#         project_patterns = [
#             r'(?i)(?:project|built|developed|created|designed)[\s:]*([A-Z][A-Za-z\s-]+?)(?:\n|$|\.)',
#             r'(?i)([A-Z][A-Za-z\s-]+?)(?:\s*-\s*|:\s*)(?:web|mobile|desktop|machine learning|ai|data)[\w\s]*'
#         ]
        
#         for pattern in project_patterns:
#             matches = re.findall(pattern, project_text)
#             projects.extend([match.strip() for match in matches if len(match.strip()) > 5])
        
#         # Use spaCy for noun phrase extraction
#         if self.nlp:
#             doc = self.nlp(project_text[:1000])  # Limit to avoid processing too much text
            
#             for sent in doc.sents:
#                 if any(keyword in sent.text.lower() for keyword in ['project', 'built', 'developed', 'created']):
#                     for chunk in sent.noun_chunks:
#                         if len(chunk.text.strip()) > 5 and chunk.text.strip() not in projects:
#                             projects.append(chunk.text.strip())
        
#         return list(set(projects))[:10]  # Limit to top 10 projects
    
#     def extract_certifications(self, text):
#         """Extract certifications and achievements"""
#         certifications = []
        
#         # Certification patterns
#         cert_patterns = [
#             r'(?i)(certified|certification|certificate)[\w\s]*',
#             r'(?i)(aws|azure|google cloud|oracle|cisco|microsoft|adobe)[\s\w]*(?:certified|certification)',
#             r'(?i)(comptia|cissp|cisa|cism|pmp|agile|scrum)[\w\s]*'
#         ]
        
#         for pattern in cert_patterns:
#             certs = re.findall(pattern, text)
#             certifications.extend([cert.strip() for cert in certs if len(cert.strip()) > 3])
        
#         return list(set(certifications))
    
#     def analyze_text_quality(self, text):
#         """Analyze text quality using TextBlob"""
#         try:
#             blob = TextBlob(text)
            
#             return {
#                 'word_count': len(blob.words),
#                 'sentence_count': len(blob.sentences),
#                 'avg_sentence_length': len(blob.words) / len(blob.sentences) if blob.sentences else 0,
#                 'polarity': blob.sentiment.polarity,  # -1 to 1
#                 'subjectivity': blob.sentiment.subjectivity  # 0 to 1
#             }
#         except:
#             return {
#                 'word_count': len(text.split()),
#                 'sentence_count': len(text.split('.')),
#                 'avg_sentence_length': 0,
#                 'polarity': 0,
#                 'subjectivity': 0
#             }
    
#     def extract_all_information(self, text):
#         """Extract all information from resume text using NLP"""
#         print("🔍 Extracting contact information...")
#         contact_info = self.extract_contact_info(text)
        
#         print("👤 Extracting name...")
#         name = self.extract_name(text)
        
#         print("🎓 Extracting education...")
#         education = self.extract_education(text)
        
#         print("💼 Extracting experience...")
#         experience = self.extract_experience(text)
        
#         print("🛠️ Extracting skills...")
#         skills = self.extract_skills(text)
        
#         print("🎯 Extracting projects...")
#         projects = self.extract_projects(text)
        
#         print("🏆 Extracting certifications...")
#         certifications = self.extract_certifications(text)
        
#         print("📊 Analyzing text quality...")
#         text_analysis = self.analyze_text_quality(text)
        
#         # Combine all extracted information
#         extracted_data = {
#             'name': name,
#             'emails': contact_info['emails'],
#             'phones': contact_info['phones'],
#             'linkedin': contact_info['linkedin'],
#             'github': contact_info['github'],
#             'websites': contact_info['websites'],
#             'education': education,
#             'experience': experience,
#             'skills': skills,
#             'projects': projects,
#             'certifications': certifications,
#             'text_analysis': text_analysis
#         }
        
#         print(f"✅ Extraction complete! Found {len(skills)} skills, {len(projects)} projects, {len(experience)} experience items")
        
#         return extracted_data