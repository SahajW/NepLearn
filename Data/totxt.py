import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def pdf_to_text(pdf_path, output_txt, dpi=300):
   

    pages = convert_from_path(pdf_path, dpi=dpi)
    all_text = []

    
    for i, page in enumerate(pages, start=1):
        print(f"OCR on page {i}/{len(pages)}...")

        # Convert to grayscale to improve accuracy
        gray = page.convert("L")

        text = pytesseract.image_to_string(gray)
        all_text.append(text)

    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    print(f"\nDone! Extracted text saved to: {output_txt}")



pdf_to_text("LetUsC.pdf", "LetusC2.txt")
