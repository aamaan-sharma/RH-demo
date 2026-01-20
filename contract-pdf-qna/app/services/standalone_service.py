"""Service for standalone question rephrasing."""
from typing import Optional, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from app.services.llm_factory import get_llm_factory


class StandaloneService:
    """Service for creating standalone questions from conversation context."""
    
    def __init__(self, llm_factory=None):
        """Initialize standalone service.
        
        Args:
            llm_factory: Optional LLMFactory instance
        """
        self.llm_factory = llm_factory or get_llm_factory()
    
    def create_standalone_question(
        self,
        current_question: str,
        previous_question: str = "",
        previous_answer: str = "",
        mode: str = "Search",
        handler: Optional[Any] = None
    ) -> str:
        """Create a standalone question from conversation context.
        
        Args:
            current_question: Current user question
            previous_question: Previous question in conversation
            previous_answer: Answer to previous question
            mode: Mode ("Search" or "Infer")
            handler: Optional callback handler for token tracking
            
        Returns:
            Standalone question string
        """
        if mode == "Search":
            prompt_template = """
Act as an expert in question rephrasing and create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
If the current question is not related to previous question and answer, then return the current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based that create the standalone question.
standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
previous question: {previous_question}
answer of previous question: {previous_answer}
current question: {current_question}

examples:
1)  previous question:''
    answer of previous question: ''
    current question: is the Fridge covered?
    standalone question: is the Fridge covered?

2)  previous question: is the Air Conditioner system covered?
    answer of previous question: yes, the air conditioner system is covered under the contract.
    current question: is the compressor covered?
    standalone question: is the compressor of the air conditioner system covered?

In some of the cases, we will not need rephrasing, for example:

3)  previous question: is the kitchen faucet covered?
    answer of previous question: yes, the kitchen faucet is covered under the contract.
    current question: is the garbage disposal covered?
    standalone question: is the garbage disposal covered?

4)  previous question: is the washer covered
    answer of previous question: yes, washer is covered under the contract.
    current question: there is damage to air conditioning unit because of leak but it is secondary, is it covered?
    standalone question: there is damage to air conditioning unit because of leak but it is secondary, is it covered?
"""
        else:  # Infer mode
            prompt_template = """
Identify if the current question is related to previous question and answer and Create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
If the current question is not related to previous question and answer, then return the current question as standalone question. If the previous question and answer is not available, then return current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based that create the standalone question.
standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
Always only return the output.
previous question: {previous_question}
answer of previous question: {previous_answer}
current question: {current_question}

examples:
If there is no previous question or previous answer, then do not create the standalone question at all.
1)  previous question:''
    answer of previous question: ''
    current question: is the Fridge covered?
    standalone question: is the Fridge covered?
    
If there is secondary damage to the appliance being talked, create a standalone question in following way.
2)  previous question: my oven caught fire, is the oven covered?
    answer of the previous question:Yes, your oven is covered by the plan. The plan covers all parts and components of installed ranges, ovens, and cooktops, including burner, display, self-clean, igniter, element, control panel and board, oven heating element, and temperature sensor. However, there are certain limitations and exclusions that apply, so it's important to review the specific terms and conditions of your plan for more details.
    current question: this fire has damaged the exhaust fan located above it, is it covered?
    standalone question: is the secondary damaged caused by the fire in the oven to the exhaust fan covered? 

In some of the cases, current question wont need rephrasing, for example:

3)  previous question: is the washer covered
    answer of previous question: yes, washer is covered under the contract.
    current question: there is damage to air conditioning unit because of leak but it is secondary damage, is it covered?
    standalone question: there is damage to air conditioning unit because of leak but it is secondary damage, is it covered?
"""
        
        standalone_prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = self.llm_factory.create_standalone_llm()
        standalone_chain = standalone_prompt | llm | StrOutputParser()
        
        # Invoke with template variables
        invoke_params = {
            "previous_question": previous_question,
            "previous_answer": previous_answer,
            "current_question": current_question
        }
        
        if handler:
            result = standalone_chain.invoke(invoke_params, config={"callbacks": [handler]})
        else:
            result = standalone_chain.invoke(invoke_params)
        
        return result.strip()
