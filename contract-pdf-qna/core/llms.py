from langchain_openai import ChatOpenAI


TRANSCRIPT_QA_AGENT = ChatOpenAI(model="gpt-4o", temperature=0.0)

SEARCH_LLM2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
SEARCH_LLM = ChatOpenAI(temperature=0.0, model="gpt-4o")

INFER_LLM3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
INFER_LLM1 = ChatOpenAI(temperature=0.0, model="gpt-4o")
INFER_LLM2 = ChatOpenAI(temperature=0.0, model="gpt-4o")