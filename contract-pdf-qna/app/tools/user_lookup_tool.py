"""User Lookup tool for LangChain agents."""
from typing import Any, Optional
from langchain.agents import Tool
from app.models.database import Database


def create_user_lookup_tool(database: Database) -> Tool:
    """Create User Lookup tool for agent.
    
    Args:
        database: Database service instance
        
    Returns:
        Tool instance for User Lookup
    """
    def fetch_user_by_mobile(mobile_number: str) -> str:
        """Fetch user details from the database based on mobile number.
        
        Args:
            mobile_number: The mobile number to search for
            
        Returns:
            A string containing user details in JSON format, or an error message
        """
        try:
            user = database.fetch_user_by_mobile(mobile_number)
            
            if user:
                import json
                return json.dumps(user, indent=2, default=str)
            else:
                return f"No user found with mobile number: {mobile_number}"
        except Exception as e:
            return f"Error fetching user details: {str(e)}"
    
    return Tool(
        name="User Lookup",
        func=fetch_user_by_mobile,
        description=(
            "Useful for fetching user details from the database based on mobile number. "
            "Use this tool when you need to retrieve customer information, user profile, "
            "or any user-related data. "
            "Input should be the mobile number as a string. Returns user details in JSON "
            "format if found, or an error message if not found."
        ),
    )
