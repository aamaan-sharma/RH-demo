"""Agent service for LangChain agent orchestration."""
import asyncio
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_community.memory.motorhead_memory import MotorheadMemory
from langchain_openai import ChatOpenAI
from app.config.settings import settings
from app.tools.knowledge_base_tool import create_knowledge_base_tool
from app.tools.user_lookup_tool import create_user_lookup_tool
from app.models.database import Database


# System message for agent
SYS_MSG = """
You are assisting an AHS customer care executive with home insurance related inquiries from AHS customers. 

You are given a tool named Knowledge Base, always use this tool to answer the questions. 

You also have access to a tool named User Lookup that can fetch user details from the database based on mobile number. Use this tool when you need to retrieve customer information or user profile data. 

The inquiry asked might be subject to some exclusions and limitations which need to be checked for first before answering the rest of the inquiry. 
You have to break down these complex inquiries into multiple subqueries and then use the knowledge base tool multiple times to return the overall answer from the subqueries for the customer's inquiry. 
Make sure to answer to all the subqueries before you return the final answer.

Following are some examples of complex inquiries and how they can be broken down into sub queries.

Example 1:
"My dryer is not drying the clothes properly. It could be because of lint blockage. Will you come to fix it?".
1. Is the dryer covered by the plan? If yes, Is repair for link blockage in the dryer covered by the plan?

Example 2: 
"I got my refrigerator fixed last week. But there is another issue with it now. What if that problem was caused by the last repair?"
1. Is the refrigerator covered in the plan?
2. If yes, Can another issue with the refrigerator be fixed in a week's time from the last repair?

Example 3: 
"I purchased a plan from AHS just 5 days ago, and now I want to repair the microwave because it is creating too much noise. Can I get this repair done?" 
1. Is the microwave covered by the plan? If yes, is the repair for noise from the microwave covered?
2. Can I file a service request within 5 days of getting the plan?

Example 4:
"I use my personal washing machine for my daycare business too at my home. The drain pump doesn't seem to be working. Is it covered?"
1. Is washing machine and it's drain pump covered by the plan?
2. Is the breakdown of washing machine due to commercial use covered?

Example 5: 
"My water heater is leaking for some reason. I need to get it fixed. That water leak seeped into the air conditioning system, so that is not working too. So I need to get that fixed too."
1. Is the water heater covered by the plan?
2. Is the air conditioning system covered by the plan?
3. Is secondary damage to the air conditioning system due to the water heater covered?

Some questions might be simpler and so might not need breaking down. Find response to those questions as it is. Following are examples of such inquiries.

Example 6:
"My microwave is not working. Is it covered?"

Example 7:
"My toilet seat is broken. Will you repair it?"

Do not answer any questions for which information is not provided by the knowledge base tool. 

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question." For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.
"""


class AgentService:
    """Service for managing LangChain agents."""
    
    def __init__(self, llm_factory, database: Database):
        """Initialize agent service.
        
        Args:
            llm_factory: LLMFactory instance
            database: Database service instance
        """
        self.llm_factory = llm_factory
        self.database = database
    
    def create_agent(
        self,
        qa_chain: RetrievalQA,
        llm: ChatOpenAI,
        memory: Optional[Any] = None,
        use_motorhead: bool = True,
        handler: Optional[Any] = None
    ) -> Any:
        """Create and initialize agent with tools.
        
        Args:
            qa_chain: RetrievalQA chain for Knowledge Base tool
            llm: ChatOpenAI instance
            memory: Optional memory instance (creates MotorheadMemory if None and use_motorhead=True)
            use_motorhead: Whether to use MotorheadMemory for long-term memory
            handler: Optional callback handler
            
        Returns:
            Initialized agent
        """
        # Create tools
        knowledge_base_tool = create_knowledge_base_tool(qa_chain)
        user_lookup_tool = create_user_lookup_tool(self.database)
        tools = [knowledge_base_tool, user_lookup_tool]
        
        # Create memory if not provided
        if memory is None and use_motorhead:
            memory = self._create_motorhead_memory()
        
        # Initialize agent
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=True,
            memory=memory,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        
        # Set system message
        new_prompt = agent.agent.create_prompt(system_message=SYS_MSG, tools=tools)
        agent.agent.llm_chain.prompt = new_prompt
        
        return agent
    
    def run_agent(
        self,
        agent: Any,
        query: str,
        handler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Run agent with query.
        
        Args:
            agent: Initialized agent
            query: User query
            handler: Optional callback handler
            
        Returns:
            Agent response dictionary
        """
        callbacks = [handler] if handler else []
        response = agent({"input": query}, callbacks=callbacks)
        return response
    
    def create_agent_with_handler(
        self,
        qa_chain: RetrievalQA,
        llm: ChatOpenAI,
        handler: Optional[Any] = None,
        memory: Optional[Any] = None,
        use_motorhead: bool = True
    ) -> Any:
        """Create agent with callback handler support.
        
        Args:
            qa_chain: RetrievalQA chain for Knowledge Base tool
            llm: ChatOpenAI instance
            handler: Optional callback handler
            memory: Optional memory instance
            use_motorhead: Whether to use MotorheadMemory
            
        Returns:
            Initialized agent
        """
        agent = self.create_agent(qa_chain, llm, memory, use_motorhead, handler)
        return agent
    
    def input_prompt(self, entered_query: str, qa_chain: RetrievalQA, llm: ChatOpenAI, handler: Optional[Any] = None) -> Dict[str, Any]:
        """Run agent with query using input_prompt pattern (Infer mode).
        
        Args:
            entered_query: User query
            qa_chain: RetrievalQA chain
            llm: ChatOpenAI instance
            handler: Optional callback handler
            
        Returns:
            Agent response dictionary with 'output' and 'intermediate_steps'
        """
        # Create tools
        knowledge_base_tool = create_knowledge_base_tool(qa_chain)
        user_lookup_tool = create_user_lookup_tool(self.database)
        tools = [knowledge_base_tool, user_lookup_tool]
        
        # Create memory
        memory = self._create_motorhead_memory()
        
        # Initialize agent
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=True,
            memory=memory,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        
        # Set system message
        new_prompt = agent.agent.create_prompt(system_message=SYS_MSG, tools=tools)
        agent.agent.llm_chain.prompt = new_prompt
        
        # Run agent
        callbacks = [handler] if handler else []
        response = agent({"input": entered_query}, callbacks=callbacks)
        return response
    
    def _create_motorhead_memory(self) -> MotorheadMemory:
        """Create MotorheadMemory instance for long-term memory.
        
        Returns:
            MotorheadMemory instance
        """
        import time
        current_time = time.time()
        
        memory = MotorheadMemory(
            api_key=settings.MOTORHEAD_API_KEY,
            client_id=settings.MOTORHEAD_CLIENT_ID,
            session_id=str(current_time),
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output",
        )
        
        # Initialize memory asynchronously
        async def memory_initialize():
            await memory.init()
        
        asyncio.run(memory_initialize())
        
        return memory
