from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime
from langchain.callbacks.base import BaseCallbackHandler
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.documents import Document
import time
import os
import json
import base64
from pathlib import Path

# Optional: OpenTelemetry child spans for per-step visibility in Jaeger.
# Off by default; enable with:
# - OTEL_TRACE_LLM_CALL_SPANS=1 (per LLM call spans)
# - OTEL_TRACE_TOOL_CALL_SPANS=1 (per tool/retriever call spans)
from opentelemetry import trace


def _env_truthy(name: str, default: str = "0") -> bool:
    raw = (os.getenv(name, default) or "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


_OTEL_TRACE_LLM_CALL_SPANS = _env_truthy("OTEL_TRACE_LLM_CALL_SPANS", "1")
_OTEL_TRACE_TOOL_CALL_SPANS = _env_truthy("OTEL_TRACE_TOOL_CALL_SPANS", "1")
_otel_tracer = trace.get_tracer("csr_copilot.langchain")


def _load_bigquery_credentials():
    """
    Docker-friendly BigQuery auth:
    1) If GOOGLE_APPLICATION_CREDENTIALS points to a file, use it.
    2) Else if BIGQUERY_SERVICE_ACCOUNT_JSON(_BASE64) is set, use it.
    3) Else fall back to ./bigquery.json if present.
    """
    scopes = ["https://www.googleapis.com/auth/bigquery"]

    # Common container mount path (works even if entrypoint didn't set env yet)
    default_mounted = Path("/run/secrets/bigquery.json")
    if default_mounted.exists():
        return service_account.Credentials.from_service_account_file(
            str(default_mounted), scopes=scopes
        )

    cred_path = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if cred_path and Path(cred_path).exists():
        return service_account.Credentials.from_service_account_file(
            cred_path, scopes=scopes
        )

    raw_b64 = (os.getenv("BIGQUERY_SERVICE_ACCOUNT_JSON_BASE64") or "").strip()
    if raw_b64:
        info = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)

    raw_json = (os.getenv("BIGQUERY_SERVICE_ACCOUNT_JSON") or "").strip()
    if raw_json:
        info = json.loads(raw_json)
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)

    local = Path("bigquery.json")
    if local.exists():
        return service_account.Credentials.from_service_account_file(
            str(local), scopes=scopes
        )

    raise FileNotFoundError(
        "BigQuery credentials not found. Provide GOOGLE_APPLICATION_CREDENTIALS "
        "(mounted file), or BIGQUERY_SERVICE_ACCOUNT_JSON(_BASE64), or mount ./bigquery.json."
    )


BIGQUERY_ENABLED = True
_bigquery_init_error = None

try:
    credentials = _load_bigquery_credentials()
    # Initialize the BigQuery client
    client_t = bigquery.Client(
        credentials=credentials, project=os.getenv("BQ_PROJECT") or "data-404309"
    )
except Exception as e:
    # Don't crash the whole app if BigQuery creds aren't provided in Docker.
    BIGQUERY_ENABLED = False
    _bigquery_init_error = str(e)
    client_t = None

# Define the dataset ID, table ID
dataset_id = "data123"
table_id = "Token"

if BIGQUERY_ENABLED:
    # Construct the reference to the table
    table_ref = client_t.dataset(dataset_id).table(table_id)
    table = client_t.get_table(table_ref)
else:
    table_ref = None
    table = None

def token_calculator(dict, session_id=None):
    for i in dict:
        token_insert_to_bigquery(i, session_id=session_id)

def token_insert_to_bigquery(dic, session_id=None):
    if not BIGQUERY_ENABLED:
        # Keep the rest of the app working even if BigQuery is not configured.
        # You can enable by mounting /run/secrets/bigquery.json or setting env vars.
        return
    current_time = datetime.now()
    
    # Format the current time as a string
    time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    data_to_insert = [
        {
            'timestamp' : time,
            'model_name' : dic["model_name"],
            'total_token_count' : dic["total_tokens"],
            'input_token' : dic["prompt_tokens"],
            'output_token' : dic["completion_tokens"]
        },
        # Add more dictionaries for additional rows
    ]
    if session_id is not None:
        data_to_insert[0]['session_id'] = session_id
    # Insert the data into the table
    errors = client_t.insert_rows(table, data_to_insert)
    if not errors:
        print(f"Data inserted successfully into {table_id}.")
    else:
        print('Errors occurred during data insertion:', errors)



class CallbackHandler(BaseCallbackHandler):

    def __init__(
        self,
        model_id: Optional[str] = None,
        model_version: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        
        self.token_usage = []
        self.client = []
        self.result_list = []
        # run_id -> active OTEL span (ended on *_end callbacks)
        self._otel_active_spans: Dict[UUID, Any] = {}
        # self.model_id = model_id
        # self.model_version = model_version
        # self.verbose = verbose
        # self.is_chat_openai_model = False
        # self.chat_openai_model_name = "gpt-3.5-turbo"

    def append_to_list(
        self,
        key: str,
        value: Any,
        start_time: Any,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        is_ts: bool = True,
    ) -> None:
        
        if is_ts:
            payload = {
                "time": time.time(),
                key: value
            }
            
            self.client.append(payload)
        else:
            payload = {
                "chain_name": key,
                "latency": value,
                'start_time': start_time,
                'end_time': start_time + value,
                "run_id":run_id,
                "parent_run_id":parent_run_id
            }

            self.result_list.append(payload)

    def infi(self):
        temp_result_list = self.result_list  
        temp_token_usage = self.token_usage
        self.result_list = []  
        self.token_usage = []
        # print(11111, temp_token_usage)
        return temp_result_list, temp_token_usage


    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if serialized is None:
            llm_name = "<unknown>"
        else:
            llm_name = serialized.get("name") or (serialized.get("id")[-1] if serialized.get("id") else "<unknown>")
        self.append_to_list("chain_name", llm_name,run_id, parent_run_id )

        # Optional: child span per LLM call (shows ReAct loops clearly in Jaeger)
        if _OTEL_TRACE_LLM_CALL_SPANS:
            try:
                span = _otel_tracer.start_span("llm_call")
                span.set_attribute("langchain.run_id", str(run_id))
                if parent_run_id is not None:
                    span.set_attribute("langchain.parent_run_id", str(parent_run_id))
                span.set_attribute("langchain.llm.name", str(llm_name))
                try:
                    span.set_attribute("langchain.prompts.count", int(len(prompts or [])))
                    prompt_chars = 0
                    for p in (prompts or []):
                        if isinstance(p, str):
                            prompt_chars += len(p)
                    span.set_attribute("langchain.prompts.chars", int(prompt_chars))
                except Exception:
                    pass
                self._otel_active_spans[run_id] = span
            except Exception:
                # Never break the main flow for tracing.
                pass


    def on_llm_end(self, response: LLMResult,*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        # Calculate and track the request latency.
        if not self.client:
            return  # No matching start event, skip processing
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        self.append_to_list(last_dict['chain_name'], latency,last_dict['time'],run_id, parent_run_id , is_ts=False)
        prompt_response = []
        for generations in response.generations:
            for generation in generations:
                prompt_response.append(generation.text)

        # Track token usage (for non-chat models).
        if (response.llm_output is not None) and isinstance(response.llm_output, Dict):
            token_usage = response.llm_output["token_usage"]
            if token_usage is not None:
                payload = {
                "prompt_tokens": token_usage["prompt_tokens"],
                "total_tokens": token_usage["total_tokens"],
                'completion_tokens': token_usage["completion_tokens"],
                'model_name': response.llm_output["model_name"]
                }
                self.token_usage.append(payload)

        # End optional span for this LLM call.
        if _OTEL_TRACE_LLM_CALL_SPANS:
            try:
                span = self._otel_active_spans.pop(run_id, None)
                if span is not None:
                    try:
                        if (response.llm_output is not None) and isinstance(response.llm_output, Dict):
                            tu = response.llm_output.get("token_usage") or {}
                            span.set_attribute("llm.model", str(response.llm_output.get("model_name") or ""))
                            span.set_attribute("llm.tokens.prompt", int(tu.get("prompt_tokens") or 0))
                            span.set_attribute("llm.tokens.completion", int(tu.get("completion_tokens") or 0))
                            span.set_attribute("llm.tokens.total", int(tu.get("total_tokens") or 0))
                    except Exception:
                        pass
                    span.end()
            except Exception:
                pass

    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any],*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain starts."""
        if serialized is None:
            chain_name = "<unknown>"
        else:
            chain_name = serialized.get("name") or (serialized.get("id")[-1] if serialized.get("id") else "<unknown>")
        self.append_to_list("chain_name", chain_name,run_id, parent_run_id )

        pass

    def on_chain_end(self, outputs: Dict[str, Any],*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """Do nothing when LLM chain ends."""
        if not self.client:
            return  # No matching start event, skip processing
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        self.append_to_list(last_dict['chain_name'], latency,last_dict['time'],run_id, parent_run_id , is_ts=False)

        pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool starts."""
        tool_name = "<tool>"
        try:
            tool_name = serialized.get("name") or (serialized.get("id")[-1] if serialized.get("id") else tool_name)
        except Exception:
            pass

        self.append_to_list("chain_name", str(tool_name),run_id, parent_run_id )

        if _OTEL_TRACE_TOOL_CALL_SPANS:
            try:
                span = _otel_tracer.start_span("tool_call")
                span.set_attribute("langchain.run_id", str(run_id))
                if parent_run_id is not None:
                    span.set_attribute("langchain.parent_run_id", str(parent_run_id))
                span.set_attribute("langchain.tool.name", str(tool_name))
                if isinstance(input_str, str):
                    span.set_attribute("langchain.tool.input.chars", int(len(input_str)))
                self._otel_active_spans[run_id] = span
            except Exception:
                pass

        pass

    def on_tool_end(
        self,
        output: str,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Do nothing when tool ends."""
        if not self.client:
            return  # No matching start event, skip processing
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        self.append_to_list(last_dict['chain_name'], latency,last_dict['time'],run_id, parent_run_id , is_ts=False)

        if _OTEL_TRACE_TOOL_CALL_SPANS:
            try:
                span = self._otel_active_spans.pop(run_id, None)
                if span is not None:
                    try:
                        if isinstance(output, str):
                            span.set_attribute("langchain.tool.output.chars", int(len(output)))
                    except Exception:
                        pass
                    span.end()
            except Exception:
                pass

        pass

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever starts running."""
        
        self.append_to_list("chain_name", "VectorStoreRetriever",run_id, parent_run_id )

        if _OTEL_TRACE_TOOL_CALL_SPANS:
            try:
                span = _otel_tracer.start_span("retriever_call")
                span.set_attribute("langchain.run_id", str(run_id))
                if parent_run_id is not None:
                    span.set_attribute("langchain.parent_run_id", str(parent_run_id))
                if isinstance(query, str):
                    span.set_attribute("retriever.query.chars", int(len(query)))
                self._otel_active_spans[run_id] = span
            except Exception:
                pass
   
    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        if not self.client:
            return  # No matching start event, skip processing
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        self.append_to_list(last_dict['chain_name'], latency,last_dict['time'],run_id, parent_run_id , is_ts=False)

        if _OTEL_TRACE_TOOL_CALL_SPANS:
            try:
                span = self._otel_active_spans.pop(run_id, None)
                if span is not None:
                    try:
                        span.set_attribute("retriever.docs.count", int(len(list(documents or []))))
                    except Exception:
                        pass
                    span.end()
            except Exception:
                pass
        
