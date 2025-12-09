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


credentials = service_account.Credentials.from_service_account_file(
    r'bigquery.json',
    scopes=['https://www.googleapis.com/auth/bigquery']
)


# Initialize the BigQuery client
client_t = bigquery.Client(credentials=credentials, project='data-404309')

# Define the dataset ID, table ID
dataset_id = 'data123'
table_id = 'Token'

# Construct the reference to the table
table_ref = client_t.dataset(dataset_id).table(table_id)
table = client_t.get_table(table_ref)

def token_calculator(dict):
    for i in dict:
        token_insert_to_bigquery(i)

def token_insert_to_bigquery(dic):
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
        llm_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        self.append_to_list("chain_name", llm_name,run_id, parent_run_id )


    def on_llm_end(self, response: LLMResult,*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        # Calculate and track the request latency.
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

    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any],*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> None:
        """Do nothing when LLM chain starts."""
        chain_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        self.append_to_list("chain_name", chain_name,run_id, parent_run_id )

        pass

    def on_chain_end(self, outputs: Dict[str, Any],*,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """Do nothing when LLM chain ends."""
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
        self.append_to_list("chain_name", "on_tool_start",run_id, parent_run_id )

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
        self.append_to_list(last_dict['chain_name'], latency,last_dict['time'],run_id, parent_run_id , is_ts=False)

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
        
