from passlib.context import CryptContext #For password hasing
from jose import jwt  #JWT = signed string.
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import hashlib

# Load environment variables from .env file
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.hash(password_hash)

def verify_password(password, hashed):
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.verify(password_hash, hashed)

def create_token(email: str):
    payload = {
        "sub": email,
        "exp": datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

""" Token contains:
email
expiry time
Used by frontend for authentication. """