from pymongo import MongoClient
from pymongo.collection import Collection
from functools import cache
from pydantic import BaseModel
from typing import Optional
from config import (
    OPENAI_API_KEY,
    MONGO_URI,
    MILVUS_HOST,
    JWT_AUDIENCE,
    JWKS_URL,
    GCP_BUCKET_NAME,
    GCP_PROJECT_ID,
    MOTORHEAD_API_KEY,
    MOTORHEAD_CLIENT_ID
)

class User(BaseModel):
    _id: str
    name: str
    email: str
    mobile: str
    age: int
    state: str
    contractType: str
    plan: str

mongo_client = MongoClient(MONGO_URI, unicode_decode_error_handler='ignore')


@cache
def get_db_collection(collection_name: str, db_name: str="AHS") -> Collection: 
    return mongo_client[db_name][collection_name]

@cache
def get_user_details_from_mobile(mobile_number: str) -> Optional[User]:
    users_collection = get_db_collection("Users")
    user = users_collection.find_one({"mobile": mobile_number})
    try:
        if user:
            # Convert ObjectId to string for JSON serialization
            if "_id" in user:
                user["_id"] = str(user["_id"])
            # Return user details as JSON string
            return User(**user)
    except Exception as e:
        print("Exception Occured: ", e)
    return None

    
