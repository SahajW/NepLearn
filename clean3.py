#program to removes punctuations like . and ?

import json
import string
import re

# Load JSON file
with open("cleaned_KUexam.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Function to remove punctuation and normalize spaces
def clean_text(text):
    if not text:
        return text
    # Remove punctuation (keep backticks for code if needed)
    keep = "`"  # remove or add other symbols if you want to preserve them
    text = text.translate(str.maketrans("", "", ''.join(c for c in string.punctuation if c not in keep)))
    # Convert multiple spaces/newlines/tabs into single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces
    text = text.strip()
    return text

# Update cleaned_text for all entries
for item in data:
    if "cleaned_text" in item and item["cleaned_text"]:
        item["cleaned_text"] = clean_text(item["cleaned_text"])

# Save the updated JSON
with open("KUexam_questions.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Punctuation removed and spaces normalized in cleaned_text for all entries.")
