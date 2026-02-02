import re

def get_year(text):
    match = re.search(r'\b(20\d{2})\b', text)
    if match:
        return int(match.group(1))
    return None

def check_input(text):
    """Check if text is asking to generate questions"""
    text = text.lower()
    
    # Look for generation-related words
    generation_words = ["generate", "create", "make", "prepare", "build","give"]
    question_words = ["question", "paper", "exam", "test", "quiz"]
    
    has_generation = any(word in text for word in generation_words)
    has_question = any(word in text for word in question_words)
    
    return has_generation and has_question


def check_subject(text):
    """Extract subject from text (returns subject name or None)"""
    text = text.lower()
    
    # Common subjects - add more as needed
    subjects = {
        "c": ["c programming", "c-programming", " c ", "language c"],
        "python": ["python"],
        "java": ["java"],
        "data structures": ["data structures", "ds", "dsa"],
        "database": ["database", "dbms", "sql"],
    }
    
    for subject, keywords in subjects.items():
        if any(keyword in text for keyword in keywords):
            return subject
    
    return None
