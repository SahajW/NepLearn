import json

with open("KUexam_questions.json", "r", encoding="utf-8") as f:
    exam_data = json.load(f)

with open("e_book.json", "r", encoding="utf-8") as f:
    text_data = json.load(f)


def normalize_exam_question(q, subject):
    return {
        "id": q["id"],
        "source": "exam",
        "subject": subject,

        "raw_text": q["raw_text"],
        "cleaned_text": q["cleaned_text"],
        "embedding": q["embedding"],

        "topic_id": None,
        "concept_id": None,

        "exam_meta": {
            "year": q["year"],
            "year_semester": q["year_semester"],
            "section": q["section"],
            "marks": q["marks"]
        },

        "derived_features": {
            "frequency": None,
            "last_seen_year": q["year"],
            "gap_since_last_seen": None,
            "recency_decay": None,
            "topic_importance": None,
            "concept_importance": None,
            "section_prob": {
                "A": None,
                "B": None,
                "C": None
            }
        }
    }


def normalize_textbook_question(q, subject):
    return {
        "id": q["id"],
        "source": "textbook",
        "subject": subject,

        "raw_text": q["raw_text"],
        "cleaned_text": q["cleaned_text"],
        "embedding": q["embedding"],

        "topic_id": None,
        "concept_id": None,

        "exam_meta": None,

        "derived_features": {
            "frequency": None,
            "last_seen_year": None,
            "gap_since_last_seen": None,
            "recency_decay": None,
            "topic_importance": None,
            "concept_importance": None,
            "section_prob": {
                "A": None,
                "B": None,
                "C": None
            }
        }
    }


master_questions = []

subject = exam_data["subject"]       #duitai ma same nai xa

for q in exam_data["questions"]:
    master_questions.append(normalize_exam_question(q, subject))

for q in text_data["questions"]:
    master_questions.append(normalize_textbook_question(q, subject))


with open("mixed_questions.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "subject": subject,
            "questions": master_questions
        },
        f,
        indent=2
    )
