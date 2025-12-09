import os, openai
from langchain.embeddings.openai import OpenAIEmbeddings



openai.api_key = os.environ["OPENAI_API_KEY"]

model_name = 'text-embedding-ada-002'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
llm2 = ChatOpenAI(temperature=0.0, model_name="gpt-4-1106-preview")
question1 = ""
answer1 = ""
while True:
    entered_query = input("enter the question\n ")
    main_prompt = """
        Identify if the current question is related to previous question and answer and Create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
        If the current question is not related to previous question and answer, then return the current question as standalone question. If the previous question and answer is not available, then return current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based on that create the standalone question.
        standalone question should always contain the appliance name, unless it is a service related question. 
        Always only return the output.
        previous question: """ + question1 + """
        answer of previous question: """ + answer1 + """
        current question: """ + entered_query + """

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
    standalone_prompt = ChatPromptTemplate.from_template(
        main_prompt
    )
    # start = int(time())
    standalone_chain = LLMChain(llm=llm2, prompt=standalone_prompt, verbose=True)

    standalone_result = standalone_chain.run({"input": entered_query})
    question1 = entered_query
    answer1 = standalone_result
    print(standalone_result)