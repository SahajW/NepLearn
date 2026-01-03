import json
from sentence_transformers import SentenceTransformer

INPUT="KUexam_questions.json"

#loading json file
with open(INPUT,"r",encoding="utf-8") as f:
    data=json.load(f)
questions=data["questions"]
#lists of cleaned questions text
texts=[item["cleaned_text"] for item in questions]

#loading MiniLM model
model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#generating embeddings(array of vectors)
'''MiniLM tokenizes the text → breaks it into tokens the model understands.It passes the tokens through the neural network.
The model produces a dense vector ( 384  dimensions for the MiniLM variant we are using).
This vector captures the semantic meaning of the question.
Similar questions get vectors that are close together in the vector space.
All vectors are collected into a list or array, same order as your texts.'''

embeddings=model.encode(texts,show_progress_bar=True) #progress bar ta fancy dekhinxa

#adding embeddings back to json
for i,item in enumerate(questions):     #i for index,item for element 
    item["embedding"]=embeddings[i].tolist()   #to convert numpy array to list

#overwritting and updating json file
with open(INPUT,"w",encoding="utf-8") as f:
    json.dump(data,f,indent=4)

print("embeddings generated successfully")
