import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re
import json
import os


def extract_single_question_marks(raw):
    if not raw:
        return None

    s = raw.replace("×", "x").replace("*", "x").replace(" ", "")

    m = re.search(r'\d+Qx(\d+)', s, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r'\d+x(\d+)', s)
    if m:
        return m.group(1)

    m = re.search(r'(\d+)Qx(\d+)=?(\d+)?', s, flags=re.IGNORECASE)
    if m:
        return m.group(2)

    nums = re.findall(r'\d+', s)
    if len(nums) == 1:
        return nums[0]

    return None


# OCR setup and PDF reading
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

pages = convert_from_path("COMP.pdf")

# read all pages and join OCR results
page_texts = []
for i, pg in enumerate(pages):
    img_path = f"page_{i+1}.png"
    pg.save(img_path, "PNG")
    page_texts.append(pytesseract.image_to_string(Image.open(img_path)))

raw_text = "\n".join(page_texts)


# basic text cleanup
text = (raw_text.replace("“", '"')
                 .replace("”", '"')
                 .replace("×", "x")
                 .replace("‘", "'")
                 .replace("’", "'"))


# extract course and semester
course_match = re.search(r'Course\s*:\s*([A-Za-z0-9 ]+)', text)
course = course_match.group(1).strip() if course_match else None

sem_match = re.search(r'Semester\s*:\s*([A-Za-z0-9]+)', text)
semester = sem_match.group(1).strip() if sem_match else None


# find section headings
sections = []
for m in re.finditer(r'SECTION\s+"?([A-Z])"?', text, flags=re.IGNORECASE):
    sections.append({"section": m.group(1), "pos": m.start()})


# find marks blocks
marks_list = []
for m in re.finditer(r'\[([^\]]*?)\s*marks\]', text, flags=re.IGNORECASE):
    marks_list.append({"marks": m.group(1).strip(), "pos": m.start()})


# find question numbers
questions = []
for m in re.finditer(r'\b(\d{1,2})\.\s', text):
    questions.append({"qnum": m.group(1), "pos": m.start()})

# add end marker
questions.append({"qnum": None, "pos": len(text)})


# extract question text
question_blocks = []
for i in range(len(questions) - 1):
    start = questions[i]["pos"]
    end = questions[i + 1]["pos"]

    num = questions[i]["qnum"]
    block = text[start:end].strip()

    clean = re.sub(r'^[^A-Za-z0-9]\d+\s\.\s*', '', block).strip()
    if num is None:
        continue

    question_blocks.append({"qnum": num, "text": clean, "pos": start})


# combine extracted data
final_questions = []

for q in question_blocks:
    pos = q["pos"]

    sec = None
    for s in reversed(sections):
        if s["pos"] <= pos:
            sec = s["section"]
            break

    marks = None
    for m in reversed(marks_list):
        if m["pos"] <= pos:
            marks = extract_single_question_marks(m["marks"])
            break

    final_questions.append({
        "course": course,
        "semester": semester,
        "question_number": q["qnum"],
        "question_text": q["text"],
        "section": sec,
        "marks": marks,
        "year": "2013",
        "exam_type": "End Semester"
    })


# for q in final_questions:
#     print(q)
#     print("")

filename = "multi_year_questions.json"

# if file exists, load previous data; otherwise start empty
if os.path.exists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        try:
            old_data = json.load(f)
        except json.JSONDecodeError:
            old_data = []
else:
    old_data = []

# extend old data with newly extracted questions
old_data.extend(final_questions)

# write everything back to the file
with open(filename, "w", encoding="utf-8") as f:
    json.dump(old_data, f, indent=4, ensure_ascii=False)