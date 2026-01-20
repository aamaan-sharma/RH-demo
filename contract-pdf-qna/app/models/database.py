"""MongoDB database operations with dependency injection."""
from typing import Dict, Any, Optional, List
from bson.objectid import ObjectId
from pymongo import MongoClient
from app.config.settings import settings


class Database:
    """Database service for MongoDB operations."""
    
    def __init__(self, mongo_client: MongoClient, db_name: str = "FrontDoorDB"):
        """Initialize database service.
        
        Args:
            mongo_client: MongoDB client instance
            db_name: Database name
        """
        self.client = mongo_client
        self.db = mongo_client[db_name]
        self.db2 = mongo_client[settings.MONGO_DB_NAME] if settings.MONGO_DB_NAME else None
        self.transcripts_collection = self.db2.call_transcripts if self.db2 is not None else None
    
    # Feedback CRUD Operations
    def insert_feedback(self, data: Dict[str, Any], email_id: str) -> ObjectId:
        """Insert feedback document.
        
        Args:
            data: Feedback data dictionary
            email_id: User email ID
            
        Returns:
            Inserted document ID
        """
        feedbacks_collection_user = f"feedbacks_{email_id}"
        feedbacks_collection = self.db[feedbacks_collection_user]
        result = feedbacks_collection.insert_one(data)
        print(f"Document inserted with ID: {result.inserted_id}")
        return result.inserted_id
    
    def read_feedback(self, query: Dict[str, Any], email_id: str) -> List[Dict[str, Any]]:
        """Read feedback documents.
        
        Args:
            query: Search query dictionary
            email_id: User email ID
            
        Returns:
            List of matching documents
        """
        feedbacks_collection_user = f"feedbacks_{email_id}"
        feedbacks_collection = self.db[feedbacks_collection_user]
        search_query = {"entered_query": query} if query else {}
        documents = feedbacks_collection.find(search_query) if search_query else feedbacks_collection.find()
        return list(documents)
    
    def update_feedback(self, query: Dict[str, Any], new_data: Dict[str, Any], email_id: str) -> int:
        """Update feedback document.
        
        Args:
            query: Search query dictionary
            new_data: Update data dictionary
            email_id: User email ID
            
        Returns:
            Number of modified documents
        """
        feedbacks_collection_user = f"feedbacks_{email_id}"
        feedbacks_collection = self.db[feedbacks_collection_user]
        search_query = {"entered_query": query}
        result = feedbacks_collection.update_one(search_query, {"$set": new_data})
        print(f"Modified {result.modified_count} document(s)")
        return result.modified_count
    
    def delete_feedback(self, query: Dict[str, Any], email_id: str) -> int:
        """Delete feedback document.
        
        Args:
            query: Search query dictionary
            email_id: User email ID
            
        Returns:
            Number of deleted documents
        """
        feedbacks_collection_user = f"feedbacks_{email_id}"
        feedbacks_collection = self.db[feedbacks_collection_user]
        search_query = {"entered_query": query}
        result = feedbacks_collection.delete_one(search_query)
        print(f"Deleted {result.deleted_count} document(s)")
        return result.deleted_count
    
    # Questions and Answers CRUD Operations
    def insert_qna(self, data: Dict[str, Any], email_id: str) -> Any:
        """Insert Q&A document.
        
        Args:
            data: Q&A data dictionary
            email_id: User email ID
            
        Returns:
            Insert result
        """
        qna_collection_today = f"chats_{email_id}"
        qna_collection = self.db[qna_collection_today]
        result = qna_collection.insert_one(data)
        print(f"Document inserted with ID: {result.inserted_id}")
        return result
    
    def read_qna(self, email_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Read Q&A document by conversation ID.
        
        Args:
            email_id: User email ID
            conversation_id: Conversation ID
            
        Returns:
            Document dictionary or None
        """
        qna_collection_user = f"chats_{email_id}"
        qna_collection = self.db[qna_collection_user]
        search_query = {"_id": ObjectId(conversation_id)}
        document = qna_collection.find_one(search_query)
        return document
    
    def update_qna(self, query: Dict[str, Any], new_data: Dict[str, Any], email_id: str) -> int:
        """Update Q&A document.
        
        Args:
            query: Search query dictionary
            new_data: Update data dictionary
            email_id: User email ID
            
        Returns:
            Number of modified documents
        """
        qna_collection_today = f"chats_{email_id}"
        qna_collection = self.db[qna_collection_today]
        search_query = {"entered_query": query}
        result = qna_collection.update_one(search_query, {"$set": new_data})
        print(f"Modified {result.modified_count} document(s)")
        return result.modified_count
    
    def delete_qna(self, query: Dict[str, Any], email_id: str) -> int:
        """Delete Q&A document.
        
        Args:
            query: Search query dictionary
            email_id: User email ID
            
        Returns:
            Number of deleted documents
        """
        qna_collection_today = f"chats_{email_id}"
        qna_collection = self.db[qna_collection_today]
        search_query = {"entered_query": query}
        result = qna_collection.delete_one(search_query)
        print(f"Deleted {result.deleted_count} document(s)")
        return result.deleted_count
    
    def update_chat(self, new_data: Dict[str, Any], conversation_id: str, email_id: str) -> int:
        """Update chat in conversation document.
        
        Args:
            new_data: Chat data to append
            conversation_id: Conversation ID
            email_id: User email ID
            
        Returns:
            Number of modified documents
        """
        qna_collection_user = f"chats_{email_id}"
        qna_collection = self.db[qna_collection_user]
        search_query = {"_id": ObjectId(conversation_id)}
        result = qna_collection.update_one(search_query, {"$push": {"chats": new_data}})
        print(f"Modified {result.modified_count} document(s)")
        return result.modified_count
    
    def get_collection(self, collection_name: str):
        """Get a collection from the database.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection object
        """
        return self.db[collection_name]
    
    def fetch_user_by_mobile(self, mobile_number: str) -> Optional[Dict[str, Any]]:
        """Fetch user details from database by mobile number.
        
        Args:
            mobile_number: Mobile number to search for
            
        Returns:
            User document dictionary or None
        """
        try:
            ahs_db = self.client["AHS"]
            users_collection = ahs_db["Users"]
            user = users_collection.find_one({"mobile": mobile_number})
            
            if user and "_id" in user:
                user["_id"] = str(user["_id"])
            
            return user
        except Exception as e:
            print(f"Error fetching user by mobile: {e}")
            return None
