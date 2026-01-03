#Program to keep track of id and sub id

import json
from collections import Counter


INPUT="cleaned_KUexam_questions.json"
OUTPUT="cleaned_KUexam.json"

# Load your JSON file
with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count how many subquestions exist per original id
id_counter = Counter(item["id"] for item in data)

# Keep track of sub-index for each original id
sub_index_tracker = {}

for item in data:
    original_id = item["id"]
    item["id_original"] = original_id  # store original ID
    
    # If multiple subquestions exist for this id
    if id_counter[original_id] > 1:
        if original_id not in sub_index_tracker:
            sub_index_tracker[original_id] = 1
        else:
            sub_index_tracker[original_id] += 1
        item["id"] = f"{original_id}_{sub_index_tracker[original_id]}"
    else:
        # Single subquestion: id = original id
        item["id"] = original_id

# Save updated JSON
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("JSON updated successfully with conditional IDs!")
