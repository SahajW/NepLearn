from sqlalchemy import Column,Integer,String 
from database import Base #from database.py importing base class

class User(Base):
    __tablename__= "users" #Table name for database
    id = Column(Integer,primary_key=True,index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String,unique=True, index=True)
    hashed_password = Column(String)
