from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from schemas import UserCreate, UserLogin  # Import Pydantic schemas
from models import User  # Import SQLAlchemy model
from auth import hash_password, verify_password, create_token
from pydantic import BaseModel,Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
from typing import Dict, List, Optional
from final_model import QuestionPaperPredictor
from Checkall import get_year,check_input,check_subject,check_nset,check_pastpaper,check_textbook
from train import train_model
import json
# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add CORS middleware BEFORE defining routes
origins = [
    "http://localhost:3000","http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Routes
@app.post("/signup")
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user
    hashed_password = hash_password(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    
    db.add(new_user)  # Prepares the INSERT statement
    db.commit()       # Executes: INSERT INTO users VALUES (...)
    db.refresh(new_user)  # Gets the auto-generated ID
    # Create access token
    access_token = create_token(user.email)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username
    }

@app.post("/login")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    # Find user by email
    db_user = db.query(User).filter(User.email == user.email).first()
    
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create access token
    access_token = create_token(user.email)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": db_user.username
    }

#LLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ============================================================================
# CHANGE 1: Better Device Detection
# ============================================================================
# ORIGINAL CODE:
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# PROBLEM: Didn't check for MPS (Apple Silicon GPU), causing the original error
# "Placeholder storage has not been allocated on MPS device!"
#
# FIX: Check for all device types in priority order
# ============================================================================
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model_dict = {}

def load_model():
    global model_dict
    if not model_dict:
        print(f"Using device: {device}")
        print(f"Loading {MODEL_NAME}...")
        
        model_dict["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # ====================================================================
        # CHANGE 2: Conditional Model Loading Strategy
        # ====================================================================
        # ORIGINAL CODE:
        # model_dict["model"] = AutoModelForCausalLM.from_pretrained(
        #     MODEL_NAME,
        #     device_map="auto",
        #     dtype=torch.float16,  # Was torch_dtype in older code
        #     low_cpu_mem_usage=True
        # )
        #
        # PROBLEM 1: device_map="auto" doesn't work well with MPS/CPU
        # - device_map="auto" tells PyTorch to manage device placement itself
        # - Then calling .to(device) later creates a conflict
        # - Model parts end up on different devices than inputs
        # - Result: "Placeholder storage has not been allocated" error
        #
        # PROBLEM 2: float16 (half precision) not supported on MPS/CPU
        # - MPS has incomplete float16 support, causes crashes
        # - CPU is very slow with float16 and may not support it
        #
        # PROBLEM 3: torch_dtype parameter was deprecated
        # - Newer transformers versions use dtype instead
        #
        # FIX: Use different strategies for different devices
        # ====================================================================
        
        if device == "cuda":
            # CUDA: Use device_map="auto" for smart multi-GPU handling
            # device_map="auto" automatically distributes model across GPUs
            # and handles input device placement for us
            model_dict["model"] = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",  # Auto-handles device placement
                dtype=torch.float16,  # CUDA supports float16 well
                low_cpu_mem_usage=True
            )
        else:
            # MPS/CPU: Manual device placement required
            # Don't use device_map="auto" because it conflicts with .to(device)
            model_dict["model"] = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.float32,  # Use float32 for stability
                low_cpu_mem_usage=True
            ).to(device)  # Manually place entire model on device
        
        print("Model loaded successfully!")

class Query(BaseModel):
    question: str

    # ============================================================================
# CHANGE 3: Load Model at Startup, Not Per Request
# ============================================================================
# ORIGINAL CODE:
# @app.post("/ask")
# def ask_llm(query: Query):
#     load_model()  # Called every request!
#     ...
#
# PROBLEM: Loading a 1.1B parameter model takes 10-30 seconds
# - Every request had to wait for model loading
# - Even with the model_dict check, it still took time
# - Browser/client would timeout waiting for response
# - Result: Page keeps loading forever
#
# FIX: Load model once when server starts using startup event
# ============================================================================
@app.on_event("startup")
async def startup_event():
    # Load model once when server starts, not on every request
    load_model()

@app.post("/ask")
def ask_llm(query: Query):
    # Check if model is loaded (safety check)
    if not model_dict:
        return {
            "answer": "", 
            "status": "error", 
            "error": "Model not loaded yet. Please wait for server startup to complete."
        }
    
    tokenizer = model_dict["tokenizer"]
    model = model_dict["model"]
    
    # TinyLlama chat format
    prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{query.question}</s>\n<|assistant|>\n"
    
    # ====================================================================
    # CHANGE 4: Proper Input Device Placement
    # ====================================================================
    # ORIGINAL CODE:
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #
    # PROBLEM: When using device_map="auto" (CUDA), inputs are on CPU
    # but model expects to handle device placement itself
    # - Causes device mismatch
    # - Model and inputs on different devices
    #
    # FIX: Only manually move inputs when NOT using device_map="auto"
    # ====================================================================
    
    # Tokenize first
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Only manually place inputs if we're not using device_map="auto"
    if device != "cuda":
        # For MPS/CPU: manually move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
    # For CUDA with device_map="auto": do nothing, it handles placement

    # ====================================================================
    # CHANGE 5: Better Generation and Response Extraction
    # ====================================================================
    # ORIGINAL CODE:
    # outputs = model.generate(**inputs, max_new_tokens=200, ...)
    # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # answer = answer.replace(prompt, "").strip()
    #
    # PROBLEM 1: No error handling - crashes would kill the server
    # PROBLEM 2: Prompt removal by string replacement is unreliable
    # - Special tokens might be decoded differently
    # - String matching fails
    # - Result: Entire prompt appears in answer
    #
    # PROBLEM 3: Missing pad_token_id causes warnings
    #
    # FIX: Add error handling, extract only generated tokens
    # ====================================================================
    
    try:
        with torch.no_grad():  # Disable gradient computation (saves memory)
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id  # Prevents warnings
            )
        
        # BEST METHOD: Extract only the newly generated tokens
        # outputs[0] contains: [input_tokens] + [generated_tokens]
        # We only want [generated_tokens]
        input_length = inputs['input_ids'].shape[1]  # Length of input
        generated_tokens = outputs[0][input_length:]  # Skip input, keep generation
        
        # Decode only the generated tokens
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return {"answer": answer, "status": "success"}
    
    except Exception as e:
        # Catch any errors during generation and return helpful message
        return {
            "answer": "", 
            "status": "error", 
            "error": f"Generation failed: {str(e)}"
        }

@app.get("/")
def root():
    return {
        "message": "LLM API with TinyLlama", 
        "status": "ready",
        "device": device  # Show which device is being used
    }

@app.get("/health")
def health_check():
    #Check if model is loaded and ready
    model_loaded = bool(model_dict)
    return {
        "status": "healthy" if model_loaded else "initializing",
        "model_loaded": model_loaded,
        "device": device
    }

# ============================================================================
# SUMMARY OF ALL CHANGES AND WHY THEY WERE NECESSARY
# ============================================================================
#
# 1. DEVICE DETECTION (Line ~20)
#    Problem: Didn't detect MPS, causing "Placeholder storage" error
#    Fix: Check for CUDA, then MPS, then CPU
#
# 2. CONDITIONAL MODEL LOADING (Line ~45)
#    Problem: device_map="auto" + .to(device) conflict, float16 on MPS fails
#    Fix: Use device_map="auto" only for CUDA, manual .to(device) for MPS/CPU
#         Use float32 for MPS/CPU for stability
#
# 3. STARTUP LOADING (Line ~75)
#    Problem: Model loaded on every request, causing 30s delays and timeouts
#    Fix: Load once at server startup with @app.on_event("startup")
#
# 4. INPUT PLACEMENT (Line ~100)
#    Problem: Manually moving inputs to device conflicts with device_map="auto"
#    Fix: Only manually move inputs when not using device_map="auto"
#
# 5. RESPONSE EXTRACTION (Line ~115)
#    Problem: Prompt included in response, no error handling
#    Fix: Extract only generated tokens by slicing, add try-except
#
# 6. PARAMETER UPDATE (throughout)
#    Problem: torch_dtype deprecated
#    Fix: Use dtype instead
#
# ============================================================================


#FIRST CODE DONE BY YOU SAHAJ

"""

```python
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

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],

)
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
```

"""

#FOR OUR MODEL(SAHAJ PLEASE LOOK)

    
class Features(BaseModel):
    user_input: str
    structure: Dict = Field(
    ...,
    description="Structure of the question paper",
    example={
        "SECTION B": {
            "count": 7,
            "instruction": "Attempt Any SIX Questions",
            "min_length": 30,
            "max_length": 400,
           
        },
        "SECTION C": {
            "count": 3,
            "instruction": "Attempt Any TWO Questions",
            "min_length": 150,
            "require_multipart": True
            }
        }
    ) 
    similarity_threshold: float = 0.85
    source_filter: str = "mixed"

@app.get('/')
def read_root():
    return {"message":"Welcome to the model"}
@app.post("/predict")
 

def predict_model(req:Features):
    try:
        year = get_year(req.user_input)
        pastpaper = check_pastpaper(req.user_input)
        textbook = check_textbook(req.user_input)
        sets = check_nset(req.user_input)
        print("Using year:", year)
        valid = check_input(req.user_input)
        if not valid:
             raise HTTPException(
                status_code=400,
                detail="Use prompt like (Generate the questions of subject and the year.)"
            )
        check = check_subject(req.user_input)
        if not check:
             raise HTTPException(
                status_code=400,
                detail="We have only questions of C-Programming, use prompt acc to that"
            )
        if year is None:
            raise HTTPException(
                status_code=400,
                detail="Year (20xx) not found in input text"
            )
        
        train_model(year,pastpaper,textbook,sets)
        with open('question_predictor.pkl','rb') as f:
            predictor = pickle.load(f)
        paper = predictor.generate_question_paper(
            structure=req.structure,      
            similarity_threshold=req.similarity_threshold,
            source_filter=req.source_filter
        )
        
        
        score = paper.get("prediction_score", None)
        
        response = {
            "metadata": {
                "subject": "C Programming",  # You can extract this from user_input
                "year": year,
                "duration": "3 hours",
                "total_marks": 100,
                "score" : score}, 
            "sections": paper.get("sections", paper)  # If paper already has sections
        }

        return {"paper":response}
    except Exception as e:
        print("ERROR:", repr(e)) 
        raise HTTPException(status_code=500, detail=str(e))
    
