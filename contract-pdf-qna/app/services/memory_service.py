"""Memory service for conversation memory management."""
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from app.models.database import Database


class MemoryService:
    """Service for managing conversation memory."""
    
    def __init__(self, database: Database):
        """Initialize memory service.
        
        Args:
            database: Database service instance
        """
        self.database = database
    
    def load_conversation_memory(
        self,
        email_id: str,
        conversation_id: str,
        memory: InMemoryChatMessageHistory,
        max_pairs: int = 3
    ) -> Tuple[str, str]:
        """Load conversation memory from MongoDB.
        
        Args:
            email_id: User email ID
            conversation_id: Conversation ID
            memory: InMemoryChatMessageHistory instance to populate
            max_pairs: Maximum Q&A pairs to load
            
        Returns:
            Tuple of (question1, answer1) for standalone prompt
        """
        memory.clear()
        question1 = ""
        answer1 = ""
        
        if not conversation_id:
            return question1, answer1
        
        docs = self.database.read_qna(email_id, conversation_id)
        if docs and "chats" in docs and len(docs["chats"]) > 0:
            skip_chat_ids = {"final_answer", "claim_decision", "case_closed"}
            skip_entered = {"Final Answer for transcript"}
            pairs = []
            
            for c in reversed(docs.get("chats") or []):
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("chat_id") or "").strip()
                q = str(c.get("entered_query") or "").strip()
                a = str(c.get("response") or "").strip()
                
                if not q or not a:
                    continue
                if cid in skip_chat_ids or q in skip_entered:
                    continue
                if a == "Loading Response":
                    continue
                
                pairs.append((q, a))
                if len(pairs) >= max_pairs:
                    break
            
            if pairs:
                question1, answer1 = pairs[0]
            
            # Store in memory (oldest -> newest)
            for q, a in reversed(pairs):
                memory.add_message(HumanMessage(content=q))
                memory.add_message(AIMessage(content=a))
        
        return question1, answer1
    
    def create_buffer_memory(
        self,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        input_key: str = "input",
        output_key: str = "output"
    ) -> ConversationBufferMemory:
        """Create ConversationBufferMemory instance.
        
        Args:
            memory_key: Memory key
            return_messages: Whether to return messages
            input_key: Input key
            output_key: Output key
            
        Returns:
            ConversationBufferMemory instance
        """
        return ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages,
            input_key=input_key,
            output_key=output_key,
        )
