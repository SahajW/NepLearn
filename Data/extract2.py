import pytesseract
from pdf2image import convert_from_path
import re
import json
import os


PDF_PATH = "C:\\Users\\ACER\\Desktop\\Python\\Project\\Data\\ExtractQ.pdf"
OUTPUT_JSON = "text_questions.json"

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# CLEAN TEXT
def clean_text(text):
    text = (text.replace("“", '"')
                 .replace("”", '"')
                 .replace("×", "x")
                 .replace("‘", "'")
                 .replace("’", "'"))

    # remove camscanner watermark
    text = re.sub(r'scanned\s+with\s+camscanner.*', '', text, flags=re.IGNORECASE)
    return text

# OCR
pages = convert_from_path(PDF_PATH)
page_texts = []

for pg in pages:
    w, h = pg.size
    pg = pg.crop((0, 0, w, int(h * 0.92)))   # remove watermark area
    text = pytesseract.image_to_string(pg, config="--psm 6")
    page_texts.append(text)

text = clean_text("\n".join(page_texts))


# MAIN QUESTION REGEX
# Matches: 1. , 2. , 10. , 3.1. 
question_matches = list(re.finditer(
    r'^\s*(\d+(?:\.\d+)*)\.\s+',
    text,
    flags=re.MULTILINE
))

question_blocks = []
for i, m in enumerate(question_matches):
    start = m.start()
    end = question_matches[i + 1].start() if i + 1 < len(question_matches) else len(text)
    question_blocks.append({
        "qnum": m.group(1),
        "block": text[start:end].strip()
    })



# SUB-QUESTION REGEX
# Supports (a), (b), (i), (ii), (1), (2)
SUBQ_REGEX = re.compile(
    r'\(\s*([a-z]|[ivxlcdm]+|\d+)\s*\)\s*(.*?)'
    r'(?=(\(\s*[a-z]|'
    r'\(\s*[ivxlcdm]+|'
    r'\(\s*\d+|$))',
    re.IGNORECASE | re.DOTALL
)

# PARSE QUESTIONS + SUB QUESTIONS
final_questions = []

for qb in question_blocks:
    qnum = qb["qnum"]
    block = qb["block"]

    # Remove "1. " from start
    content = re.sub(r'^\s*\d+(?:\.\d+)*\.\s*', '', block).strip()

    subs = list(SUBQ_REGEX.finditer(content))

    # Extract main question 
    if subs:
        main_question = content[:subs[0].start()].strip().rstrip(':')
    else:
        main_question = content.strip()

    # If NO sub-questions, just store single question
    if not subs:
        final_questions.append({
            "question_number": qnum,
            "question_text": main_question,
            "sub_question_number": None,
            "sub_question_text": None
        })
        continue

    # If sub-questions exist
    for sm in subs:
        sub_num = sm.group(1)
        sub_text = sm.group(2).strip()

        final_questions.append({
            "question_number": qnum,
            "question_text": main_question,
            "sub_question_number": sub_num,
            "sub_question_text": sub_text
        })

# SAVE JSON
if os.path.exists(OUTPUT_JSON):
    try:
        old_data = json.load(open(OUTPUT_JSON, "r", encoding="utf-8"))
    except:
        old_data = []
else:
    old_data = []

old_data.extend(final_questions)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(old_data, f, indent=4, ensure_ascii=False)

print(f"Extracted {len(final_questions)} questions successfully.")
