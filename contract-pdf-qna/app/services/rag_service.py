"""RAG service for retrieval-augmented generation."""
from typing import List, Dict, Any, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config.settings import settings
from app.utils.milvus_utils import get_milvus_collection_name


class RAGService:
    """Service for RAG operations."""
    
    def __init__(self, llm_factory, embedding: OpenAIEmbeddings):
        """Initialize RAG service.
        
        Args:
            llm_factory: LLMFactory instance
            embedding: OpenAIEmbeddings instance
        """
        self.llm_factory = llm_factory
        self.embedding = embedding
        self._vector_db_cache: Dict[str, Milvus] = {}
    
    def get_vector_db(
        self,
        collection_name: str,
        milvus_host: Optional[str] = None
    ) -> Milvus:
        """Get or create Milvus vector database instance.
        
        Args:
            collection_name: Milvus collection name
            milvus_host: Milvus host (defaults to settings)
            
        Returns:
            Milvus instance
        """
        if collection_name in self._vector_db_cache:
            return self._vector_db_cache[collection_name]
        
        host = milvus_host or settings.MILVUS_HOST
        vector_db = Milvus(
            self.embedding,
            collection_name=collection_name,
            connection_args={"host": host, "port": "19530"},
        )
        self._vector_db_cache[collection_name] = vector_db
        return vector_db
    
    def create_retriever(
        self,
        collection_name: str,
        k: int = None,
        milvus_host: Optional[str] = None
    ) -> Any:
        """Create retriever from Milvus collection.
        
        Args:
            collection_name: Milvus collection name
            k: Number of documents to retrieve (defaults to settings)
            milvus_host: Milvus host
            
        Returns:
            Retriever instance
        """
        vector_db = self.get_vector_db(collection_name, milvus_host)
        k = k or settings.MILVUS_RETRIEVER_K
        return vector_db.as_retriever(search_kwargs={"k": k})
    
    def create_retrieval_qa(
        self,
        retriever: Any,
        llm: ChatOpenAI,
        prompt_template: Optional[str] = None
    ) -> RetrievalQA:
        """Create RetrievalQA chain.
        
        Args:
            retriever: Retriever instance
            llm: ChatOpenAI instance
            prompt_template: Optional custom prompt template string (not PromptTemplate object)
            
        Returns:
            RetrievalQA chain
        """
        if prompt_template:
            # If prompt_template is a string, create PromptTemplate from it
            # Extract standalone_result placeholder if present
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
        else:
            chain_type_kwargs = {}
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            verbose=True,
            chain_type_kwargs=chain_type_kwargs
        )
    
    def get_relevant_documents(self, query: str, retriever: Any) -> str:
        """Get relevant documents as stringified format.
        
        Args:
            query: Search query
            retriever: Retriever instance
            
        Returns:
            Stringified documents
        """
        try:
            docs = retriever.get_relevant_documents(query)
            relevant_document = "Referred Documents: " + str(docs)
            return relevant_document
        except Exception as e:
            print(f"[CHUNKS] relevant_docs: ERROR calling retriever: {e}")
            return "Referred Documents: []"
    
    def create_standalone_llm(self) -> ChatOpenAI:
        """Create LLM for standalone question rephrasing.
        
        Returns:
            ChatOpenAI instance configured for standalone prompts
        """
        return self.llm_factory.create_standalone_llm()
    
    def get_collection_for_context(
        self,
        contract_type: str,
        selected_plan: str,
        selected_state: str
    ) -> Optional[str]:
        """Get Milvus collection name for given context.
        
        Args:
            contract_type: Contract type (RE or DTC)
            selected_plan: Plan name
            selected_state: State name
            
        Returns:
            Collection name or None
        """
        return get_milvus_collection_name(contract_type, selected_plan, selected_state)
    
    def get_relevant_documents_string(self, query: str, retriever: Any) -> str:
        """Get relevant documents as stringified format (legacy format).
        
        Args:
            query: Search query
            retriever: Retriever instance
            
        Returns:
            Stringified documents in legacy format
        """
        try:
            docs = retriever.get_relevant_documents(query)
            relevant_document = "Referred Documents: " + str(docs)
            return relevant_document
        except Exception as e:
            print(f"[CHUNKS] relevant_docs: ERROR calling retriever: {e}")
            return "Referred Documents: []"
