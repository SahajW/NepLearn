import re
import json

with open("exam_questions.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

subject = "C Programming"
source = "End Semester Examination"

questions = []
question_counter = 1

# Split by year headings (2013, 2014, ..., 2024)
year_blocks = re.split(r"\n\s*(20\d{2})\s*\n", raw_text)

for i in range(1, len(year_blocks), 2):
    year = int(year_blocks[i])
    block_text = year_blocks[i + 1]

    # Extract Year / Semester (flexible)
    sem_match = re.search(
        r"Year:\s*([IV]+)\s*Semester:\s*([IV]+)",
        block_text,
        flags=re.I
    )
    year_semester = f"{sem_match.group(1)}/{sem_match.group(2)}" if sem_match else None

    # Extract sections B and C safely
    sections = re.split(r"\n\s*SECTION\s+[\"“]?([BC])[\"”]?\s*\n", block_text)

    for j in range(1, len(sections), 2):
        section = sections[j]
        section_text = sections[j + 1]

        marks = 4 if section == "B" else 8

        # Split questions ONLY on main numbers (1., 2., 3.)
        q_blocks = re.split(r"\n\s*(?=\d+\.\s)", section_text)

        for q in q_blocks:
            q = q.strip()
            if not re.match(r"\d+\.\s", q):
                continue

            # Remove leading question number but keep text intact
            q_text = re.sub(r"^\d+\.\s*", "", q, count=1).strip()

            questions.append({
                "id": f"KU_Q{question_counter}",
                "year": year,
                "year_semester": year_semester,
                "section": section,
                "marks": marks,
                "raw_text": q_text,
                "cleaned_text": None,
                "embedding": None,
                "cluster_important": None,
                "cluster_conceptual": None,
                "features": {
                    "frequency": None,
                    "last_seen_year": None,
                    "similarity_score": None,
                    "transition_probability": None
                }
            })

            question_counter += 1

output = {
    "subject": subject,
    "source": source,
    "questions": questions
}

with open("KUexam_questions.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"Processed {len(questions)} questions successfully.")
