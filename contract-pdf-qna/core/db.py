from pymongo import MongoClient
from pymongo.collection import Collection
from functools import cache
from dataclasses import dataclass
from typing import Optional


@dataclass
class Users:
    name: str
    email: str
    mobile: str
    age: str
    state: str
    contractType: str
    plan: str

mongo_client = MongoClient(unicode_decode_error_handler='ignore')


@cache
def get_db_collection(collection_name: str, db_name: str="AHS") -> Collection: 
    return mongo_client[db_name][collection_name]

@cache
def get_user_details_from_mobile(mobile_number: str) -> Optional[Users]:
    users_collection = get_db_collection("Users")
    user = users_collection.find_one({"mobile": mobile_number})
    try:
        if user:
            # Convert ObjectId to string for JSON serialization
            if "_id" in user:
                user["_id"] = str(user["_id"])
            # Return user details as JSON string
            return Users(**user)
    except Exception as e:
        print("Exception Occured: ", e)
    return None

    