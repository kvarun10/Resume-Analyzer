# parsing/pdf_parser.py

import fitz  # PyMuPDF

def extract_text(filepath):
    print(f"🔍 Opening PDF: {filepath}")
    try:
        with fitz.open(filepath) as doc:
            text = ""
            print(f"📄 PDF has {len(doc)} pages")
            for page in doc:
                page_text = page.get_text()
                text += page_text
            if not text:
                print("⚠️ No text extracted! The PDF might be image-based or empty.")
        return text
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""
