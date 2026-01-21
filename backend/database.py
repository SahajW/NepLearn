from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=True) #Connects to postgre
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) #Creates datababase sessions
Base = declarative_base() #base class for models
