import fitz # PyMuPDF
import sys

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        with open(pdf_path + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Successfully extracted {pdf_path}")
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")

pdfs = [
    "s00778-025-00936-6.pdf",
    "A_Survey_on_Advancing_the_DBMS_Query_Optimizer_Car.pdf"
]

for pdf in pdfs:
    extract_text(pdf)
