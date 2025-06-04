# extractor/rule_based_extractor.py
#MODULAR
import re

SKILLS_DB = [
    'Python', 'Java', 'JavaScript', 'C++', 'C', 'Machine Learning', 'Deep Learning',
    'HTML', 'CSS', 'SQL', 'React', 'Node.js', 'Pandas', 'NumPy', 'Git', 'Linux',
    'Communication', 'Teamwork', 'Leadership', 'Problem Solving', 'Critical Thinking'
]

EDUCATION_KEYWORDS = ['bachelor', 'master', 'b.tech', 'm.tech', 'bsc', 'msc', 'phd']

class RuleBasedExtractor:
    def __init__(self, text, tokens):
        self.text = text
        self.tokens = tokens

    def extract_email(self):
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        matches = re.findall(email_pattern, self.text)
        return matches[0] if matches else None

    def extract_phone(self):
        phone_pattern = r'(\+91[-\s]?\d{10})|(\d{10})'
        matches = re.findall(phone_pattern, self.text)
        flat_numbers = [num[0] if num[0] else num[1] for num in matches]
        return flat_numbers[0] if flat_numbers else None

    def extract_name(self):
        lines = self.text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if re.search(r"@|\d", line): 
                continue
            if len(line.split()) == 2:
                first, last = line.split()
                if first[0].isupper() and last[0].isupper():
                    return f"{first} {last}"
        return None

    def extract_skills(self):
        token_set = set([token.lower() for token in self.tokens])
        matched = [skill for skill in SKILLS_DB if skill.lower() in token_set]
        return matched

    def extract_education(self):
        lines = self.text.lower().split('\n')
        education_entries = [line for line in lines if any(word in line for word in EDUCATION_KEYWORDS)]
        return education_entries

#DRAFT 1:(not modular)
# import re

# #extract email
# def extract_email(text):
#     email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
#     matches = re.findall(email_pattern, text)
#     return matches[0] if matches else None

# #extract phone no.
# def extract_phone(text):
#     phone_pattern = r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}"
#     matches = re.findall(phone_pattern, text)
#     flat_numbers = [''.join(match) for match in matches]
#     return flat_numbers[0] if flat_numbers else None

# #extract name
# def extract_name(tokens):
#     for i in range(len(tokens) - 1):
#         if tokens[i][0].isupper() and tokens[i+1][0].isupper():
#             return f"{tokens[i]} {tokens[i+1]}"
#     return None

# #extract skills
# SKILLS_DB = ['Python', 'Java', 'JavaScript', 'C++', 'C', 'Machine Learning', 'Deep Learning',
#      'HTML', 'CSS', 'SQL', 'React', 'Node.js', 'Pandas', 'NumPy', 'Git', 'Linux',
#      'Communication', 'Teamwork', 'Leadership', 'Problem Solving', 'Critical Thinking']

# def extract_skills(tokens):
#     token_set = set([token.lower() for token in tokens])
#     matched = [skill for skill in SKILLS_DB if skill.lower() in token_set]
#     return matched

# #extract education
# EDUCATION_KEYWORDS = ['bachelor', 'master', 'b.tech', 'm.tech', 'bsc', 'msc', 'phd']

# def extract_education(text):
#     lines = text.lower().split('\n')
#     education_entries = [line for line in lines if any(word in line for word in EDUCATION_KEYWORDS)]
#     return education_entries
