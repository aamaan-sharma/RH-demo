# Set the OpenAI API Keys, embedding model,
import os
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from pymongo import MongoClient
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_community.memory.motorhead_memory import MotorheadMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent, AgentType
from time import time
from bson.objectid import ObjectId
import uuid
from flask_cors import CORS
from oauth2client import client
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
# Safe import of monitoring_module - handle missing dependencies gracefully
try:
    from monitoring_module import q_monitor, tracer, llm_trace_to_jaeger
except ImportError as e:
    print(f"Warning: Could not import monitoring_module: {e}")
    print("Monitoring features will be disabled. The app will continue to run.")
    # Create dummy functions to prevent errors
    def q_monitor(*args, **kwargs):
        pass
    class DummyTracer:
        def start_span(self, *args, **kwargs):
            return DummySpan()
    class DummySpan:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getattr__(self, name):
            return self
    tracer = DummyTracer()
    def llm_trace_to_jaeger(*args, **kwargs):
        pass

from token_module import token_calculator, CallbackHandler
import threading

# Using new LangChain memory API - InMemoryChatMessageHistory
# Note: This is only used to store previous Q&A for standalone prompt, not used in chains
memory1 = InMemoryChatMessageHistory()
handler = CallbackHandler()
load_dotenv()
app = Flask(__name__)

JWT_AUDIENCE = os.getenv("JWT_AUDIENCE")
JWKS_URL = os.getenv("JWKS_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MONGO_URI = (
    os.getenv("MONGO_URI")
)

CORS(app, resources={r"/*": {"origins": "*"}})

mongo_client = MongoClient(MONGO_URI, unicode_decode_error_handler='ignore')
db = mongo_client["FrontDoorDB"]

model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)


class AgentAction:
    def __init__(self, tool: str, tool_input: str, log: str = None):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log


# Creating a prompt template
prompt_template = """

You are assisting a customer care executive. Your role is to review the contract’s contextual information given in the context below.

{context}

Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

You'll be asked about repairs, coverage policy and service questions about home appliances, home fixtures, home care, repairs/replacement and cleaning, and also about the renewal, cancellation or refund policies in the contract, whether a certain service is covered under the contract and similar context.

The contract context given will have information about contractual details, terms and conditions, renewals, cancellation, refund and service request policies, the coverage limits, limitation and exclusion policies. You will need to use and infer from all the information available in context to analyze and then respond with the final answer.

If the question is about a square feet limit, make sure to compare the numerical values properly. 
For example,
Question: "will my 800 square feet guest house be covered?"
The answer to this question will be No, as the square feet limit for guest houses is 750 and 800 is greater than 750.

If the context you have received says that some breakdown is not covered due to "Misuse or Accidental Acts", then say it is not covered.
For example,
Question: "A rat chewed the wires of my ceiling fan. Is the repair covered?"
The answer to this question will be No, as contract clearly mentions that damage due to pests like rat will not be covered.

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question.". For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

Question: {question} Why?
Answer: """

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )
# chain_type_kwargs = {"prompt": PROMPT}

# conversational retrieval QA
PROMPT = PromptTemplate.from_template(prompt_template)

sys_msg = """

You are assisting an AHS customer care executive with home insurance related inquiries from AHS customers. 

You are given a tool named Knowledge Base, always use this tool to answer the questions. 

The inquiry asked might be subject to some exclusions and limitations which need to be checked for first before answering the rest of the inquiry. 
You have to break down these complex inquiries into multiple subqueries and then use the knowledge base tool multiple times to return the overall answer from the subqueries for the customer's inquiry. 
Make sure to answer to all the subqueries before you return the final answer.

Following are some examples of complex inquiries and how they can be broken down into sub queries.

Example 1:
“My dryer is not drying the clothes properly. It could be because of lint blockage. Will you come to fix it?”.
1. Is the dryer covered by the plan? If yes, Is repair for link blockage in the dryer covered by the plan?

Example 2: 
“I got my refrigerator fixed last week. But there is another issue with it now. What if that problem was caused by the last repair?”
1. Is the refrigerator covered in the plan?
2. If yes, Can another issue with the refrigerator be fixed in a week’s time from the last repair?

Example 3: 
“I purchased a plan from AHS just 5 days ago, and now I want to repair the microwave because it is creating too much noise. Can I get this repair done?” 
1. Is the microwave covered by the plan? If yes, is the repair for noise from the microwave covered?
2. Can I file a service request within 5 days of getting the plan?

Example 4:
“I use my personal washing machine for my daycare business too at my home. The drain pump doesn't seem to be working. Is it covered?”
1. Is washing machine and it's drain pump covered by the plan?
2. Is the breakdown of washing machine due to commercial use covered?

Example 5: 
“My water heater is leaking for some reason. I need to get it fixed. That water leak seeped into the air conditioning system, so that is not working too. So I need to get that fixed too.“
1. Is the water heater covered by the plan?
2. Is the air conditioning system covered by the plan?
3. Is secondary damage to the air conditioning system due to the water heater covered?

Some questions might be simpler and so might not need breaking down. Find response to those questions as it is. Following are examples of such inquiries.

Example 6:
“My microwave is not working. Is it covered?”

Example 7:
“My toilet seat is broken. Will you repair it?”

Do not answer any questions for which information is not provided by the knowledge base tool. 

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question." For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

"""


def input_prompt(entered_query, qa, llm):
    # Retriever chain as Tool for agent
    knowledge_base_tool = Tool(
        name="Knowledge Base",
        func=qa.run,
        description=(
            "Useful for answering questions related to insurance coverage of home appliances, home fixtures, their repairs/replacement, service requests, about the renewal, cancellation or refund policies, whether a certain service is covered under the contract, permit limit, code violation limit, modification limit, limitations and exclusions."
        ),
    )

    tools = [knowledge_base_tool]

    current_time = time()

    MOTORHEAD_API_KEY = os.getenv("MOTORHEAD_API_KEY")
    MOTORHEAD_CLIENT_ID = os.getenv("MOTORHEAD_CLIENT_ID")
    MOTORHEAD_SESSION_ID = str(current_time)
    MOTORHEAD_MEMORY_KEY = "chat_history"

    # Long Term chat memory
    memory = MotorheadMemory(
        api_key=MOTORHEAD_API_KEY,
        client_id=MOTORHEAD_CLIENT_ID,
        session_id=MOTORHEAD_SESSION_ID,
        memory_key=MOTORHEAD_MEMORY_KEY,
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    #
    async def memory_initialize():
        await memory.init()

    asyncio.run(memory_initialize())

    # Initializing agent
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

    new_prompt = agent.agent.create_prompt(system_message=sys_msg, tools=tools)

    agent.agent.llm_chain.prompt = new_prompt

    response = agent({"input": entered_query},callbacks=[handler])
    return response


# Function to get relevant documents
def relevant_docs(entered_query, retriever):
    relevant_document = "Referred Documents: " + str(
        retriever.get_relevant_documents(entered_query)
    )
    return relevant_document


# Feedback CRUD Operations


# Feedback CRUD Operations
# Create (Insert) operation
def insert_feedback(data, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    result = feedbacks_collection.insert_one(data)
    print(f"Document inserted with ID: {result.inserted_id}")


# Read operation
def read_feedback(query, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    documents = (
        feedbacks_collection.find(search_query)
        if search_query
        else feedbacks_collection.find()
    )
    for document in documents:
        print(document)


# Update operation
def update_feedback(query, new_data, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    result = feedbacks_collection.update_one(search_query, {"$set": new_data})
    print(f"Modified {result.modified_count} document(s)")


# Delete operation
def delete_feedback(query, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    result = feedbacks_collection.delete_one(search_query)
    print(f"Deleted {result.deleted_count} document(s)")


# Questions and Answers CRUD Operations
# Create (Insert) operation
def insert_qna(data, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    result = qna_collection.insert_one(data)
    print(f"Document inserted with ID: {result.inserted_id}")
    return result


# Anirudha Read operation
# def read_qna(query, email_id):
#     qna_collection_today = f"chats_{email_id}"
#     qna_collection = db[qna_collection_today]
#     search_query = {"entered_query": query}
#     documents = qna_collection.find(search_query) if search_query else qna_collection.find()
#     for document in documents:
#         print(document)

# def read_qna(email_id,mongo_qna_id=None):


def read_qna(email_id, conversation_id):
    qna_collection_user = f"chats_{email_id}"
    qna_collection = db[qna_collection_user]
    search_query = {"_id": ObjectId(conversation_id)}
    documents = qna_collection.find_one(search_query)
    return documents


# Update operation
def update_qna(query, new_data, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    search_query = {"entered_query": query}
    result = qna_collection.update_one(search_query, {"$set": new_data})
    print(f"Modified {result.modified_count} document(s)")


# Delete operation
def delete_qna(query, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    search_query = {"entered_query": query}
    result = qna_collection.delete_one(search_query)
    print(f"Deleted {result.deleted_count} document(s)")


def update_chat(new_data, conversation_id, email_id):
    qna_collection_user = f"chats_{email_id}"
    qna_collection = db[qna_collection_user]
    search_query = {"_id": ObjectId(conversation_id)}
    result = qna_collection.update_one(search_query, {"$push": {"chats": new_data}})
    print(f"Modified {result.modified_count} document(s)")


def token_process(authorization_header):
    parts = authorization_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        bearer_token = parts[1]
        try:
            token = client.verify_id_token(bearer_token, JWT_AUDIENCE)
            return (token), 200
        except Exception as e:
            if str(e).split(",")[0] == "Token used too late":
                return jsonify({"message": "Token has expired"}), 403
            else:
                return jsonify({"message": "Token is invalid"}), 403
    else:
        return jsonify({"message": "Token is missing"}), 401


@app.before_request
def before_request():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    if request.method == "OPTIONS" or request.method == "options":
        return jsonify(headers), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    with tracer.start_span('api/feedback'):
        authorization_header = request.headers.get("Authorization")

        # case 5: missing token
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_feedback = request.get_json()

        # extract from query parameters
        conversation_id = request.args.get("conversation-id")
        chat_id = request.args.get("chat-id")

        # extract values from input
        reaction = user_feedback.get("reaction")
        response = user_feedback.get("response")
        user_email = token_data[0]["email"]

        query_time = datetime.utcnow()

        # output to be stored in mongodb collection
        feedback_json = {
            "query_time": query_time,
            "conversation_id": conversation_id,
            "chat_id": chat_id,
            "reaction": reaction,
            "response": response,
        }

        insert_feedback(feedback_json, user_email)
        return {}


@app.route("/start", methods=["POST"])
def start():
    try:
        with tracer.start_span('api/start') as parent0:
            with tracer.start_span('authorization', child_of=parent0):
                start_time = time()
                authorization_header = request.headers.get("Authorization")

                if authorization_header is None:
                    return jsonify({"message": "Token is missing"}), 401

                if authorization_header:
                    token_data = token_process(authorization_header)

                    if token_data[1] == 401 or token_data[1] == 403:
                        return (token_data[0].get_json()), token_data[1]
            
            with tracer.start_span('data-fetching', child_of=parent0):
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Request body is missing or invalid"}), 400
                
                contract_type = data.get("contractType")
                selected_plan = data.get("selectedPlan")
                selected_state = data.get("selectedState")
                gpt_model = data.get("gptModel")
                entered_query = data.get("enteredQuery")
                
                # Validate required fields
                if not all([contract_type, selected_plan, selected_state, gpt_model, entered_query]):
                    return jsonify({"error": "Missing required fields: contractType, selectedPlan, selectedState, gptModel, enteredQuery"}), 400
                
                # user_email = "kartik.dabre@mindstix.com"
                user_email = token_data[0]["email"]
                conversation_id = request.args.get("conversation-id")

                collection_mapping = {
                    "RE": {
                        "ShieldEssential": f"{selected_state}_RE_ShieldEssential",
                        "ShieldPlus": f"{selected_state}_RE_ShieldPlus",
                        "default": f"{selected_state}_RE_ShieldComplete",
                    },
                    "DTC": {
                        "ShieldSilver": f"{selected_state}_DTC_ShieldSilver",
                        "ShieldGold": f"{selected_state}_DTC_ShieldGold",
                        "default": f"{selected_state}_DTC_ShieldPlatinum",
                    },
                }

                # Get the collection name based on contract_type and selected_plan
                selected_collection_name = collection_mapping.get(contract_type, {}).get(
                    selected_plan, collection_mapping.get(contract_type, {}).get("default")
                )
            with tracer.start_span('vector_db-initialization', child_of=parent0):
                # Selecting collection dynamically
                vector_db1: Milvus = Milvus(
                    embed,
                    collection_name=selected_collection_name,
                    connection_args={"host": MILVUS_HOST, "port": "19530"},
                )
            
            # Initialize variables to prevent undefined errors
            agent_resp = None
            relevant_documents = ""

            if gpt_model == "Search":
                with tracer.start_span('Search', child_of=parent0) as parent1:
                    with tracer.start_span('llm-retriever-initialization', child_of=parent1):
                        llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                        retriever = vector_db1.as_retriever(search_kwargs={"k": 4})
                    
                    with tracer.start_span('memory_update', child_of=parent1):
                        memory1.clear()
                        question1 = ""
                        answer1 = ""
                        if conversation_id is not None and conversation_id != "":
                            docs = read_qna(email_id=user_email, conversation_id=conversation_id)
                            if docs and "chats" in docs and len(docs["chats"]) > 0:
                                question1 = docs["chats"][-1]['entered_query']
                                answer1 = docs["chats"][-1]['response']
                                # Store in new memory API format (only if values exist)
                                if question1 and answer1:
                                    memory1.add_message(HumanMessage(content=question1))
                                    memory1.add_message(AIMessage(content=answer1))

                    with tracer.start_span('standalone-prompt-chain', child_of=parent1) as p:
                        standalone_prompt = ChatPromptTemplate.from_template(
                        """
                        Act as an expert in question rephrasing and create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
                        If the current question is not related to previous question and answer, then return the current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based on that create the standalone question.
                        standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
                        previous question: """ + question1 + """
                        answer of previous question: """ + answer1 + """
                        current question: """ + entered_query + """
                        
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
                        )
                        start = int(time())
                        standalone_chain = LLMChain(llm=llm2, prompt=standalone_prompt, verbose=True)

                        standalone_result = standalone_chain.run({"input": entered_query},callbacks=[handler])
                        print(standalone_result)
                        res1, tok1 = handler.infi()
                        llm_trace_to_jaeger(res1, p.span_id, p.trace_id)
                        a = threading.Thread(target=token_calculator, args=(tok1,))
                        a.start()

                        print(f"time taken for standalone = {time() - start}")

                    with tracer.start_span('q_monitor', child_of=parent1) as parentq:
                        t = threading.Thread(target=q_monitor, args=(parentq,entered_query,))
                        t.start()
                        # q_monitor(parentq,entered_query)

                    with tracer.start_span('llm-RetrievalQA-chain', child_of=parent1) as q:
                        # Creating a prompt template
                        prompt_template = """

                        You are assisting a customer care executive. Your role is to review the contract’s contextual information given in the context below.

                        {context}

                        Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
                        You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

                        You'll be asked about repairs, coverage policy and service questions about home appliances, home fixtures, home care, repairs/replacement and cleaning, and also about the renewal, cancellation or refund policies in the contract, whether a certain service is covered under the contract and similar context.

                        The contract context given will have information about contractual details, terms and conditions, renewals, cancellation, refund and service request policies, the coverage limits, limitation and exclusion policies. You will need to use and infer from all the information available in context to analyze and then respond with the final answer.

                        If the question is about a square feet limit, make sure to compare the numerical values properly. 
                        For example,
                        Question: "will my 800 square feet guest house be covered?"
                        The answer to this question will be No, as the square feet limit for guest houses is 750 and 800 is greater than 750.

                        If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question.". For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

                        Always include the appliance name in the answer and provide in depth information.

                        Make the answer as short as possible with in depth information.

                        Question: """ + standalone_result + """
                        Answer: """

                        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
                        chain_type_kwargs = {"prompt": PROMPT}
                        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True,
                                                        chain_type_kwargs=chain_type_kwargs)

                        agent_resp = qa.run(standalone_result,callbacks=[handler])
                        res2, tok2 = handler.infi()
                        llm_trace_to_jaeger(res2, q.span_id, q.trace_id)
                        b = threading.Thread(target=token_calculator, args=(tok2,))
                        b.start()
                    
                    with tracer.start_span('relevant_documents', child_of=parent1):
                        relevant_documents = relevant_docs(entered_query, retriever=retriever)

            elif gpt_model == "Infer":
                with tracer.start_span('Infer', child_of=parent0) as parent1:
                    with tracer.start_span('llm-retriever-initialization', child_of=parent1):
                        llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model='gpt-4o')
                        llm2 = ChatOpenAI(temperature=0.0, model='gpt-4o')
                        retriever = vector_db1.as_retriever(search_kwargs={"k": 4})
                        
                    with tracer.start_span('memory_update', child_of=parent1):
                        memory1.clear()
                        question1 = ""
                        answer1 = ""
                        if conversation_id is not None and conversation_id != "":
                            docs = read_qna(email_id=user_email, conversation_id=conversation_id)
                            if docs and "chats" in docs and len(docs["chats"]) > 0:
                                question1 = docs["chats"][-1]['entered_query']
                                answer1 = docs["chats"][-1]['response']
                                # Store in new memory API format (only if values exist)
                                if question1 and answer1:
                                    memory1.add_message(HumanMessage(content=question1))
                                    memory1.add_message(AIMessage(content=answer1))

                    with tracer.start_span('standalone-prompt-chain', child_of=parent1) as p:
                        standalone_prompt = ChatPromptTemplate.from_template(
                            """
                            Identify if the current question is related to previous question and answer and Create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
                            If the current question is not related to previous question and answer, then return the current question as standalone question. If the previous question and answer is not available, then return current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based on that create the standalone question.
                            standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
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
                        )
                        start = int(time())
                        standalone_chain = LLMChain(llm=llm3, prompt=standalone_prompt, verbose=True)

                        standalone_result = standalone_chain.run({"input": entered_query})
                        print(standalone_result)
                        res1, tok1 = handler.infi()
                        llm_trace_to_jaeger(res1, p.span_id, p.trace_id)
                        a = threading.Thread(target=token_calculator, args=(tok1,))
                        a.start()

                    with tracer.start_span('q_monitor', child_of=parent1) as parentq:
                        t = threading.Thread(target=q_monitor, args=(parentq,entered_query,))
                        t.start()

                    with tracer.start_span('llm-RetrievalQA-chain', child_of=parent1) as q:
                        qa = RetrievalQA.from_chain_type(llm=llm2, retriever=retriever, verbose=True)
                        agent_response = input_prompt(standalone_result, qa, llm)
                        agent_resp = agent_response["output"]
                        res2, tok2 = handler.infi()
                        llm_trace_to_jaeger(res2, q.span_id, q.trace_id)
                        b = threading.Thread(target=token_calculator, args=(tok2,))
                        b.start()
                    
                    with tracer.start_span('relevant_documents', child_of=parent1):
                        knowledge_base_thoughts = [item[0].tool_input for item in agent_response["intermediate_steps"] if
                                            item[0].tool == 'Knowledge Base']
                        relevant_documents = ""
                        for action_input in knowledge_base_thoughts:
                            relevant_documents += relevant_docs(action_input, retriever)
            else:
                return jsonify({"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"}), 400

            with tracer.start_span('output-formating', child_of=parent0):
                # Validate that we have a response
                if agent_resp is None:
                    return jsonify({"error": "Invalid gpt_model. Must be 'Search' or 'Infer'"}), 400
                
                ai_response = agent_resp

                word_count = len(relevant_documents.split())
                latency = time() - start_time

                query_time = datetime.now()

                chat = {
                    "chat_id": str(uuid.uuid4()),
                    "entered_query": entered_query,
                    "response": ai_response,
                    "relevant_docs": relevant_documents,
                    "gpt_model": gpt_model,
                    "chat_timestamp": query_time,
                    "latency": latency,
                    "word_count": word_count
                }

                if conversation_id is None or conversation_id == "":
                    qna_json = {
                        "conversation_name": entered_query,
                        "contract_type": contract_type,
                        "selected_plan": selected_plan,
                        "selected_state": selected_state,
                        "query_time": query_time,
                        "chats": [chat],
                    }

                    conversation_id = insert_qna(email_id=user_email, data=qna_json)
                    conversation_id = conversation_id.inserted_id

                else:
                    add_chat = update_chat(
                        new_data=chat, conversation_id=conversation_id, email_id=user_email
                    )

                output_json = {"aiResponse": ai_response, "conversationId": str(conversation_id), "chatId":chat.get("chat_id")}

        return make_response(jsonify(output_json), 200)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /start endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": "An error occurred while processing your request", "details": str(e)}), 500


@app.route("/history", methods=["GET"])
def chat_history():
    with tracer.start_span('api/history'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        conversation_id = request.args.get("conversation-id")
        user_email = token_data[0]["email"]

        docs = read_qna(email_id=user_email, conversation_id=conversation_id)

        feedback_collection_user = f"feedbacks_{user_email}"
        feedback_collection = db[feedback_collection_user]
        feedback_reaction = feedback_collection.find(
            {"conversation_id": str(conversation_id)}
        )
        feedback_dict = {}

        for doc in feedback_reaction:
            chat_id = str(doc["chat_id"])
            feedback_dict[chat_id] = doc["reaction"]

        chats = docs["chats"]
        for chat in chats:
            chat_id = chat.get("chat_id")
            if chat_id in feedback_dict:
                chat["reaction"] = feedback_dict[chat_id]

        output_json = {
            "conversationName": docs["conversation_name"],
            "contractType": docs["contract_type"],
            "selectedPlan": docs["selected_plan"],
            "selectedState": docs["selected_state"],
            "chats": chats,
            "gptModel": chats[0]["gpt_model"],
        }
        if docs:
            return make_response(jsonify(output_json), 200)
        else:
            return make_response(
                jsonify({"message": "No data found in the specified conversation"}), 404
            )


@app.route("/sidebar", methods=["GET"])
def sidebar_history():
    with tracer.start_span('api/sidebar'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        qna_collection_user = f"chats_{user_email}"
        qna_collection = db[qna_collection_user]

        # projection means setting the key name to 1, i.e we want all ids and names from given collection
        result = qna_collection.find({}, {"_id": 1, "conversation_name": 1})

        output_json = [
                        {
                            "conversationId": str(doc["_id"]),
                            "conversationName": doc["conversation_name"],
                        }
                        for doc in result
                    ][::-1]

        if output_json:
            return make_response(jsonify(output_json), 200)
        else:
            return make_response(jsonify([]), 200)


@app.route("/delete", methods=["DELETE"])
def delete_conversation():
    with tracer.start_span('api/delete'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        qna_collection_user = f"chats_{user_email}"
        qna_collection = db[qna_collection_user]
        conversation_id = request.args.get("conversation-id")

        qna_collection.delete_one({"_id": ObjectId(conversation_id)})
        return {}


@app.route("/edit-conversation-name", methods=["PATCH"])
def edit_name():
    with tracer.start_span('api/edit-conversation-name'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        data = request.get_json()
        new_name = data.get("newName")

        conversation_id = request.args.get("conversation-id")

        try:
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            qna_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"conversation_name": new_name}},
            )
            return jsonify({"message": "Conversation name updated successfully"})

        except Exception as e:
            return jsonify({"error": str(e)})


@app.route("/referred-clauses", methods=["GET"])
def referred_clauses():
    with tracer.start_span('api/referred-clauses'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]
        conversation_id = request.args.get("conversation-id")
        chat_id = request.args.get("chat-id")

        try:
            docs = read_qna(email_id=user_email, conversation_id=conversation_id)

            chat_ans = docs["chats"]
            for chat_obj in chat_ans:
                if chat_obj.get("chat_id") == chat_id:
                    question = chat_obj.get("entered_query")
                    answer = chat_obj.get("response")

            referred_clauses_json = {
                "contractType": docs["contract_type"],
                "selectedState": docs["selected_state"],
                "selectedPlan": docs["selected_plan"],
                "question": question,
                "answer": answer,
                "referredClauses": chat_obj["relevant_docs"],
                "gpt_model": chat_obj.get("gpt_model"),
                "latency": chat_obj.get("latency", None),
                "word_count": chat_obj.get("word_count", None)
            }

            return referred_clauses_json

        except Exception as e:
            return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
