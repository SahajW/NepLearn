from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

device = "cuda" if torch.cuda.is_available() else "cpu"
model_dict = {}

def load_model():
    global model_dict
    if not model_dict:
        print(f"Using device: {device}")
        print(f"Loading {MODEL_NAME}...")
        
        model_dict["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        model_dict["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print("Model loaded successfully!")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_llm(query: Query):
    load_model()
    
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    
    # TinyLlama chat format
    prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{query.question}</s>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the answer
    answer = answer.replace(prompt, "").strip()
    
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "LLM API with TinyLlama", "status": "ready"}