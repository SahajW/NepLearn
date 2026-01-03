#Program to split and clean raw question text

import json
import re

def split_into_subquestions(text):
    """
    Smartly split a question text into subquestions.
    Handles:
    1. a. / b. / c.
    2. i) / ii) / iii)
    3. Newline-based menus
    4. Multiple sentences with questions
    """
    # Normalize newlines and spaces
    text = text.replace("\n", " ").replace("\r", " ").strip()
    
    # Patterns for subquestions
    sub_patterns = [
        r"(?<=\s)[a-z]\.\s",           # a. b. c.
        r"(?<=\s)[ivxlcdm]+\)\s",      # i) ii) iii)
        r"(?<=:)\s+"                    # after colon, e.g., menu options
    ]
    
    # Combined regex
    combined_pattern = "|".join(sub_patterns)
    
    # Find all splits
    splits = re.split(combined_pattern, text)
    splits = [s.strip() for s in splits if s.strip()]
    
    # Heuristic: If only one split and contains multiple questions separated by '.', '?', keep as one
    if len(splits) == 1:
        # Attempt sentence-based splitting for multiple questions
        sentences = re.split(r"(?<=[.?!])\s+(?=[A-Z`])", splits[0])
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    return splits

def clean_text(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Remove exam artifacts
    exam_artifacts = [
        r"ku\b", r"university\b", r"end semester examination", 
        r"question \d+", r"q\d+", r"page \d+", r"^\s*[-–—]\s*"  # headers/footers
    ]
    for pattern in exam_artifacts:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 3. Remove question numbering and marks
    text = re.sub(r"\(?\d+\s*marks?\)?", "", text)
    text = re.sub(r"\[[0-9]+\s*m\]", "", text)
    text = re.sub(r"\([a-zA-Z]\)", "", text)

    # 4. Remove code blocks
    text = re.sub(r"#include[^\n]*", "", text)
    text = re.sub(r"int\s+main\s*\([^)]*\)\s*{[^}]*}", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    
    # Replace full code with semantic summary placeholder
    text = re.sub(r"\b(write a program|define a structure|implement a function)\b", r"write a program ", text)

    # 5. Normalize symbols
    text = text.replace("×", "x").replace("³", "3")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # 6. Remove unnecessary punctuation
    text = re.sub(r"[!;:\[\]{}<>]", "", text)

    # 7. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Load input JSON
with open("KUexam_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Detect if top-level dict has "questions" key
if isinstance(data, dict) and "questions" in data:
    questions = data["questions"]
else:
    questions = data

cleaned_questions = []


# Process each question
for q in questions:
    raw_text = q.get("raw_text") or q.get("text") or ""
    subquestions_raw = split_into_subquestions(raw_text)
    
    # Clean each subquestion
    for sub in subquestions_raw:
        cleaned = clean_text(sub)
        cleaned_questions.append({
            "id": q.get("id"),
            "year": q.get("year"),
            "year_semester": q.get("year_semester"),
            "section": q.get("section"),
            "marks": q.get("marks"),
            "raw_text": raw_text,
            "raw_subquestion": sub,
            "cleaned_text": cleaned,
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


# Save cleaned JSON
with open("cleaned_KUexam_questions.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_questions, f, ensure_ascii=False, indent=4)

print("Cleaning done! Cleaned JSON saved to: cleaned_KUexam_questions.json")
