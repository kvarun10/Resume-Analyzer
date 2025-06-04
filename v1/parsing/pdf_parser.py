# parsing/pdf_parser.py

import fitz  # PyMuPDF

def extract_text(filepath):
    """
    Extracts all text from a given PDF file using PyMuPDF.
    
    Args:
        filepath (str): Path to the PDF resume file.
    
    Returns:
        str: Extracted text content.
    """
    try:
        with fitz.open(filepath) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
