from whylogs.experimental.core.udf_schema import udf_schema
import whylogs as why
from langkit import toxicity
from langkit import sentiment
from langkit import themes
# from langkit import injections  # Disabled - AWS S3 data file unavailable
from langkit import textstat
from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime
import closest
import atexit
import os
from typing import Any, Dict, Iterable, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path

model = SentenceTransformer( "sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / 'files' / 'ontopic_fd.json', 'r', encoding='utf-8') as file:
    ontopic = json.load(file)
    ontopic = ontopic['jailbreak']

with open(BASE_DIR / 'files' / 'offtopic_fd.json', 'r', encoding='utf-8') as file:
    offtopic = json.load(file)
    offtopic = offtopic['jailbreak']


ontopic_embed = [model.encode(i) for i in ontopic]

offtopic_embed = [model.encode(i) for i in offtopic]


_TRACER = None


def init_tracer():
    global _TRACER
    if _TRACER:
        return _TRACER

    service_name = os.getenv("OTEL_SERVICE_NAME", "Customer Representative Copilot")
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://jaeger:4318",
    )

    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))

    _TRACER = trace.get_tracer(service_name)
    return _TRACER


tracer = init_tracer()


@atexit.register
def _shutdown_tracer():
    # Best-effort shutdown (flushes span processors once at process exit).
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass


#credentials = service_account.Credentials.from_service_account_file(
#    r'bigquery.json',
#    scopes=['https://www.googleapis.com/auth/bigquery']
#)
from google.oauth2 import service_account
BQ_CRED_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "bigquery.json"  # fallback for local non-docker runs
)

credentials = service_account.Credentials.from_service_account_file(
    BQ_CRED_PATH,
    scopes=["https://www.googleapis.com/auth/bigquery"]
)


# Initialize the BigQuery client
client = bigquery.Client(credentials=credentials, project='data-404309')

# Define the dataset ID, table ID
dataset_id = 'data123'
table_id = 'Score'

# Construct the reference to the table
table_ref = client.dataset(dataset_id).table(table_id)
table = client.get_table(table_ref)

text_schema = udf_schema()




def func_Binsert(parent1, dicts,prompt):
    ctx = trace.set_span_in_context(parent1) if parent1 else None
    with tracer.start_as_current_span("func_Binsert", context=ctx):
        # Get the current time
        current_time = datetime.now()

        # Format the current time as a string
        time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        data_to_insert = [

            {
                'Prompt': prompt,
                'timestamp': time,
                'toxicity':dicts['prompt.toxicity'],
                'sentiment':dicts['prompt.sentiment_nltk'],
                'jailbreak':dicts['prompt.jailbreak_similarity'],
                'injection':dicts.get('prompt.injection'),
                'flesch_reading_ease':dicts['prompt.flesch_reading_ease'],
                'automated_readability_index':dicts['prompt.automated_readability_index'],
                'aggregate_reading_level':dicts['prompt.aggregate_reading_level'],
                'syllable_count':dicts['prompt.syllable_count'],
                'lexicon_count':dicts['prompt.lexicon_count'],
                'character_count':dicts['prompt.character_count'],
                'difficult_words':dicts['prompt.difficult_words'],
                'ontopic':dicts['ontopic'],
                'offtopic':dicts["offtopic"],
                'Products':dicts['closest_topic']

            },
            # Add more dictionaries for additional rows
        ]

        # Insert the data into the table
        errors = client.insert_rows(table, data_to_insert)

        if not errors:
            print(f"Data inserted successfully into {table_id}.")
        else:
            print('Errors occurred during data insertion:', errors)


def closest_t(child1, question):
    ctx = trace.set_span_in_context(child1) if child1 else None
    with tracer.start_as_current_span("closest", context=ctx):
        topic = closest.classify_topic(closest.arr,question,closest.Embed)
        return topic


def score_calculator(child1, question):
    ctx = trace.set_span_in_context(child1) if child1 else None
    with tracer.start_as_current_span("score_calculator", context=ctx):
        dicts = {}
       #results = why.log({"prompt": question}, schema = text_schema)
        results = why.log(
            {"prompt": question},
            schema=text_schema
        )

        score = results.view()
        for i in score.get_columns():    
            # Skip injection metric (unstable in whylogs)
            if i == "prompt.injection":
                continue

            val = score.get_column(i).to_summary_dict().get('distribution/mean')
            dicts[i] = val
        return dicts


def ontopic_fun(child1, query):
    ctx = trace.set_span_in_context(child1) if child1 else None
    with tracer.start_as_current_span("ontopic_fun", context=ctx):
        query_embedding = model.encode(query)
        val = -10
        for i in ontopic_embed:
            t_val = util.pytorch_cos_sim(query_embedding, i)[0][0]
            if(t_val>val):
                val = t_val
        return float(val)

def offtopic_fun(child1, query):
    ctx = trace.set_span_in_context(child1) if child1 else None
    with tracer.start_as_current_span("offtopic_fun", context=ctx):
        query_embedding = model.encode(query)
        val = -10
        for i in offtopic_embed:
            t_val = util.pytorch_cos_sim(query_embedding, i)[0][0]
            if(t_val>val):
                val = t_val
        return float(val)


def security_scores(parent1, question):
    ctx = trace.set_span_in_context(parent1) if parent1 else None
    with tracer.start_as_current_span("security_scores", context=ctx) as child1:

        dicts = {}
        
        scores = score_calculator(child1, question)
        dicts.update(scores)
        dicts['closest_topic'] = closest_t(child1, question)
        

        # ontopic f3
        dicts["ontopic"] = ontopic_fun(child1, question)

        # offtopic f4
        dicts["offtopic"] = offtopic_fun(child1, question)

        return dicts


def q_monitor(parent1, question):
    # Guard against empty / bad input
    if not question or not isinstance(question, str):
        return
    ctx = trace.set_span_in_context(parent1) if parent1 else None
    # Give the monitor thread its own span; deterministic lifecycle inside the thread.
    with tracer.start_as_current_span("q_monitor", context=ctx) as span:
        dicts = security_scores(span, question)
        func_Binsert(span, dicts, question)


def llm_trace_to_otel(data: Iterable[Dict[str, Any]], parent_span=None) -> None:
    """
    Recreate a LangChain run tree as OpenTelemetry spans, exported once after aggregation.

    - Keeps one "aggregation at end" call site (see app.py)
    - Ensures parent/child relationships via OpenTelemetry context propagation
    - Preserves timing (start/end) using epoch seconds -> nanoseconds
    """
    if not data:
        return

    # Sort so parents are likely created before children (best-effort).
    items = list(data)
    items.sort(key=lambda x: (x.get("start_time") is None, x.get("start_time") or 0))

    spans_by_run_id: Dict[Any, Any] = {}
    root_ctx = trace.set_span_in_context(parent_span) if parent_span else None

    for item in items:
        name = (item.get("chain_name") or "langchain").strip()
        run_id = item.get("run_id")
        parent_run_id = item.get("parent_run_id")
        attrs = item.get("attrs") or {}

        if parent_run_id is not None and parent_run_id in spans_by_run_id:
            parent_ctx = trace.set_span_in_context(spans_by_run_id[parent_run_id])
        else:
            parent_ctx = root_ctx

        start_s = item.get("start_time")
        end_s = item.get("end_time")
        start_ns = int(start_s * 1_000_000_000) if isinstance(start_s, (int, float)) else None
        end_ns = int(end_s * 1_000_000_000) if isinstance(end_s, (int, float)) else None

        span_cm = tracer.start_as_current_span(
            name,
            context=parent_ctx,
            start_time=start_ns,
            end_on_exit=False,
        )
        span_obj = span_cm.__enter__()
        try:
            # Helpful attributes for debugging / correlating back to LangChain.
            if run_id is not None:
                span_obj.set_attribute("langchain.run_id", str(run_id))
            if parent_run_id is not None:
                span_obj.set_attribute("langchain.parent_run_id", str(parent_run_id))
            if item.get("latency") is not None:
                try:
                    span_obj.set_attribute("langchain.latency_s", float(item["latency"]))
                except Exception:
                    pass

            # Attach any extra callback-derived attributes (tool name/input/output, etc.)
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    if v is None:
                        continue
                    try:
                        if isinstance(v, (bool, int, float, str)):
                            span_obj.set_attribute(str(k), v)
                        else:
                            span_obj.set_attribute(str(k), str(v)[:1000])
                    except Exception:
                        pass

            if run_id is not None:
                spans_by_run_id[run_id] = span_obj
        finally:
            # End span with recorded end_time if present.
            try:
                if end_ns is not None:
                    span_obj.end(end_time=end_ns)
                else:
                    span_obj.end()
                span_cm.__exit__(None, None, None)
            except Exception:
                # Never let tracing break the request.
                try:
                    span_cm.__exit__(None, None, None)
                except Exception:
                    pass