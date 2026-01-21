from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "postgresql://postgres:Prabin@localhost:5432/nep_learn_db"

engine = create_engine(DATABASE_URL, echo=True) #Connects to postgre
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) #Creates datababase sessions
Base = declarative_base() #base class for models
