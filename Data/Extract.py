import pytesseract
from pdf2image import convert_from_path
import re
import fitz
import json

pages = convert_from_path("COMP.pdf")

text = ""
for i, page in enumerate(pages):
    print(f"Processing page {i+1}...")
    text += pytesseract.image_to_string(page)



# Save text to file
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Extraction completed. Check output.txt")

# -------- CONFIG --------
PDF_FILE = "COMP.pdf"
OUTPUT_JSON = "multi_year_questions.json"
COURSE_SUBJECT_MAP = {
    "COMP 103": "C Programming",
    "COMP 104": "Data Structures",
    "MATH 101": "Calculus"
}

# -------- STEP 1: Convert PDF to Text --------
def pdf_to_text(pdf_file):
    text = ""
    try:
        doc = fitz.open(pdf_file)
        for page in doc:
            text += page.get_text()
        if text.strip():
            return text
    except Exception as e:
        print("PyMuPDF failed:", e)
    
    print("Using OCR for PDF...")
    pages = convert_from_path(pdf_file)
    for i, page in enumerate(pages):
        print(f"Processing page {i+1}/{len(pages)}...")
        text += pytesseract.image_to_string(page)
    return text

# -------- STEP 2: Split PDF into Exam Blocks --------
def split_exam_blocks(text):
    # This regex assumes each exam starts with "End Semester Examination" + year
    blocks = re.split(r'(End Semester Examination\s*\n\d{4})', text)
    exams = []
    i = 0
    while i < len(blocks) - 1:
        header = blocks[i] + blocks[i+1]  # merge header with content
        content = blocks[i+2] if i+2 < len(blocks) else ""
        exams.append(header + "\n" + content)
        i += 3
    return exams

# -------- STEP 3: Extract Metadata & Questions from Each Exam --------
def process_exam_block(block):
    # Metadata
    year_match = re.search(r'(\d{4})', block)
    year = int(year_match.group(1)) if year_match else None

    course_match = re.search(r'Course\s*:\s*([A-Z]+\s*\d+)', block)
    course_code = course_match.group(1).strip() if course_match else None

    semester_match = re.search(r'Semester\s*:\s*([A-Z0-9]+)', block)
    semester = semester_match.group(1).strip() if semester_match else None

    subject = COURSE_SUBJECT_MAP.get(course_code, course_code)

    # Sections & Questions
    questions_list = []
    section_matches = re.split(r'SECTION\s*[“"]?([A-Z])', block, flags=re.I)
    
    # section_matches[0] is before first section
    for i in range(1, len(section_matches), 2):
        section_letter = section_matches[i].strip()
        section_text = section_matches[i+1].strip()

        # Optional: extract marks
        marks_match = re.search(r'Marks\s*:\s*(\d+)', section_text)
        section_marks = int(marks_match.group(1)) if marks_match else None

        # Extract questions
        q_matches = re.findall(r'(\d+)\.\s*(.+?)(?=\n\d+\.|\Z)', section_text, re.S)
        for q_num, q_text in q_matches:
            questions_list.append({
                "exam_year": year,
                "course_code": course_code,
                "subject": subject,
                "semester": semester,
                "section": section_letter,
                "question_number": int(q_num),
                "marks": section_marks,
                "text": q_text.strip(),
                "difficulty": "medium",  # default
                "question_id": f"{year}_{course_code}_{section_letter}_{q_num}"
            })
    return questions_list

# -------- MAIN --------
def main():
    full_text = pdf_to_text(PDF_FILE)
    exam_blocks = split_exam_blocks(full_text)
    
    all_questions = []
    for block in exam_blocks:
        questions = process_exam_block(block)
        all_questions.extend(questions)

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(all_questions)} questions across {len(exam_blocks)} exams.")
    print(f"JSON saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
