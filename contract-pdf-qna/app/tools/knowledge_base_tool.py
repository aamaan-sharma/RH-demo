"""Knowledge Base tool for LangChain agents."""
from typing import Any
from langchain.agents import Tool
from langchain.chains import RetrievalQA


def create_knowledge_base_tool(qa_chain: RetrievalQA) -> Tool:
    """Create Knowledge Base tool for agent.
    
    Args:
        qa_chain: RetrievalQA chain instance
        
    Returns:
        Tool instance for Knowledge Base
    """
    return Tool(
        name="Knowledge Base",
        func=qa_chain.run,
        description=(
            "Useful for answering questions related to insurance coverage of home appliances, "
            "home fixtures, their repairs/replacement, service requests, about the renewal, "
            "cancellation or refund policies, whether a certain service is covered under the "
            "contract, permit limit, code violation limit, modification limit, limitations and exclusions."
        ),
    )
