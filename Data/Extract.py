import re
import json

INPUT_TXT = "LetusC_questions.txt"
OUTPUT_JSON = "LetusC_questions.json"

SUBJECT = "C Programming"
SOURCE = "Textbook"


def extract_questions(text):
    pattern = re.compile(
        r'(\d+)\)\s*(.*?)(?=\n\d+\)|\Z)',
        re.DOTALL
    )

    matches = pattern.findall(text)

    questions = []
    q_counter = 1

    for _, q_text in matches:
        q_text = q_text.strip()

        questions.append({
            "id": f"LUC_Q{q_counter}",
            "year": None,
            "year_semester":"I/I",
            "section": None,
            "marks": None,

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

        q_counter += 1

    return questions


# ---- SCRIPT EXECUTION STARTS HERE ----

with open(INPUT_TXT, "r", encoding="utf-8") as f:
    raw_text = f.read()

questions = extract_questions(raw_text)

output_data = {
    "subject": SUBJECT,
    "source": SOURCE,
    "questions": questions
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f" Extracted {len(questions)} questions")
print(f" Output saved to: {OUTPUT_JSON}")
