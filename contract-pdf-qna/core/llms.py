from langchain_openai import ChatOpenAI
from config import MODEL_INTENT, MODEL_SUGGEST
import os
from utils.helpers import _env_float, _env_int

TRANSCRIPT_QA_AGENT = ChatOpenAI(model="gpt-4o", temperature=0.0)

SEARCH_LLM2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
SEARCH_LLM = ChatOpenAI(temperature=0.0, model="gpt-4o")

INFER_LLM3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
INFER_LLM1 = ChatOpenAI(temperature=0.0, model="gpt-4o")
INFER_LLM2 = ChatOpenAI(temperature=0.0, model="gpt-4o")






INTENT_LLM = ChatOpenAI(
    temperature=0.0,
    model=MODEL_INTENT,
    max_tokens=200,  # Limit response length for speed # type: ignore
    timeout=_env_float("LIVE_COPILOT_LLM_TIMEOUT_INTENT_S", 15.0),
    max_retries=_env_int("LIVE_COPILOT_LLM_MAX_RETRIES", 2),
)


SUGGEST_LLM = ChatOpenAI(
    temperature=0.0,
    model=MODEL_SUGGEST,
    max_tokens=500,  # Limit response length # type: ignore
    # Suggestions sometimes take longer; keep this configurable.
    timeout=_env_float("LIVE_COPILOT_LLM_TIMEOUT_SUGGEST_S", 60.0),
    max_retries=_env_int("LIVE_COPILOT_LLM_MAX_RETRIES", 2),
)


DIAGNOSTIC_LLM = ChatOpenAI(
    temperature=0.2,
    model=MODEL_SUGGEST,
    max_tokens=300,  # Limit response length # type: ignore
    timeout=_env_float("LIVE_COPILOT_LLM_TIMEOUT_DIAGNOSTICS_S", 20.0),
    max_retries=_env_int("LIVE_COPILOT_LLM_MAX_RETRIES", 2),
)
