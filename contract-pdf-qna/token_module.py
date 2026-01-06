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

def token_calculator(dict):
    for i in dict:
        token_insert_to_bigquery(i)

def token_insert_to_bigquery(dic):
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
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        
        if is_ts:
            payload = {
                "time": time.time(),
                key: value,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "attrs": attrs or {},
            }
            
            self.client.append(payload)
        else:
            payload = {
                "chain_name": key,
                "latency": value,
                'start_time': start_time,
                'end_time': start_time + value,
                "run_id":run_id,
                "parent_run_id":parent_run_id,
                "attrs": attrs or {},
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
        include_payloads = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            preview_chars = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or "500")
        except Exception:
            preview_chars = 500

        llm_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        attrs: Dict[str, Any] = {
            "langchain.kind": "llm",
            "langchain.llm.name": str(llm_name),
        }
        if include_payloads and prompts:
            attrs["langchain.llm.prompt_preview"] = (prompts[0] or "")[:preview_chars]

        # Note: append_to_list captures start time internally; we still pass run_id correctly.
        self.append_to_list("chain_name", llm_name, time.time(), run_id, parent_run_id, attrs=attrs)


    def on_llm_end(self, response: LLMResult,*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        # Calculate and track the request latency.
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        include_payloads = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            preview_chars = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or "500")
        except Exception:
            preview_chars = 500

        prior_attrs = last_dict.get("attrs") or {}
        end_attrs = dict(prior_attrs)
        if include_payloads:
            # Best-effort: capture the first generation text as a preview.
            try:
                if response.generations and response.generations[0] and getattr(response.generations[0][0], "text", None) is not None:
                    end_attrs["langchain.llm.response_preview"] = (response.generations[0][0].text or "")[:preview_chars]
            except Exception:
                pass

        # Token usage (when provided by the LLM wrapper).
        try:
            if (response.llm_output is not None) and isinstance(response.llm_output, Dict):
                token_usage = response.llm_output.get("token_usage")
                model_name = response.llm_output.get("model_name")
                if isinstance(token_usage, dict):
                    if "prompt_tokens" in token_usage:
                        end_attrs["langchain.llm.tokens.prompt"] = int(token_usage["prompt_tokens"])
                    if "completion_tokens" in token_usage:
                        end_attrs["langchain.llm.tokens.completion"] = int(token_usage["completion_tokens"])
                    if "total_tokens" in token_usage:
                        end_attrs["langchain.llm.tokens.total"] = int(token_usage["total_tokens"])
                if model_name is not None:
                    end_attrs["langchain.llm.model_name"] = str(model_name)
        except Exception:
            pass

        self.append_to_list(
            last_dict['chain_name'],
            latency,
            last_dict['time'],
            run_id,
            parent_run_id,
            is_ts=False,
            attrs=end_attrs,
        )

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

    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any],*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain starts."""
        chain_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        include_payloads = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            preview_chars = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or "500")
        except Exception:
            preview_chars = 500

        attrs: Dict[str, Any] = {"langchain.kind": "chain"}
        if include_payloads and isinstance(inputs, dict):
            try:
                attrs["langchain.chain.inputs_preview"] = (json.dumps(inputs, default=str)[:preview_chars])
            except Exception:
                attrs["langchain.chain.inputs_preview"] = (str(inputs)[:preview_chars])

        self.append_to_list("chain_name", chain_name, time.time(), run_id, parent_run_id, attrs=attrs)

        pass

    def on_chain_end(self, outputs: Dict[str, Any],*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """Do nothing when LLM chain ends."""
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        include_payloads = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            preview_chars = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or "500")
        except Exception:
            preview_chars = 500

        prior_attrs = last_dict.get("attrs") or {}
        end_attrs = dict(prior_attrs)
        if include_payloads and isinstance(outputs, dict):
            try:
                end_attrs["langchain.chain.outputs_preview"] = json.dumps(outputs, default=str)[:preview_chars]
            except Exception:
                end_attrs["langchain.chain.outputs_preview"] = str(outputs)[:preview_chars]

        self.append_to_list(
            last_dict['chain_name'],
            latency,
            last_dict['time'],
            run_id,
            parent_run_id,
            is_ts=False,
            attrs=end_attrs,
        )

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
        tool_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        include_payloads = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            preview_chars = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or "500")
        except Exception:
            preview_chars = 500

        self.append_to_list(
            "chain_name",
            tool_name,
            time.time(),
            run_id,
            parent_run_id,
            attrs={
                "langchain.kind": "tool",
                "langchain.tool.name": str(tool_name),
                "langchain.tool.input_preview": (input_str or "")[:preview_chars] if include_payloads else "",
            },
        )

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
        
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        prior_attrs = last_dict.get("attrs") or {}
        end_attrs = dict(prior_attrs)
        try:
            end_attrs["langchain.tool.output_len"] = len(output or "")
        except Exception:
            pass
        include_payloads = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            preview_chars = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or "500")
        except Exception:
            preview_chars = 500
        if include_payloads:
            try:
                end_attrs["langchain.tool.output_preview"] = (output or "")[:preview_chars]
            except Exception:
                pass
        self.append_to_list(
            last_dict['chain_name'],
            latency,
            last_dict['time'],
            run_id,
            parent_run_id,
            is_ts=False,
            attrs=end_attrs,
        )

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
        include_payloads = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            preview_chars = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or "500")
        except Exception:
            preview_chars = 500

        attrs: Dict[str, Any] = {"langchain.kind": "retriever"}
        if include_payloads:
            attrs["langchain.retriever.query_preview"] = (query or "")[:preview_chars]

        self.append_to_list(
            "chain_name",
            "VectorStoreRetriever",
            time.time(),
            run_id,
            parent_run_id,
            attrs=attrs,
        )
   
    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        
        last_dict = self.client[-1]  # Retrieve the last dictionary in the list
        latency = time.time() - last_dict['time']
        self.client.remove(last_dict)
        self.append_to_list(last_dict['chain_name'], latency,last_dict['time'],run_id, parent_run_id , is_ts=False)
        
