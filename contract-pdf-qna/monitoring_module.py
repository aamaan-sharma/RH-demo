import os
import json
from datetime import datetime
from pathlib import Path
import socket
from urllib.parse import urlparse

# OpenTelemetry (single tracing implementation)
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _otlp_http_endpoint() -> str:
    """
    OTLP/HTTP exporter endpoint.
    Accepts either:
      - http://host:4318
      - http://host:4318/v1/traces
    """
    base = _env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318")
    if base.endswith("/v1/traces"):
        return base
    return base.rstrip("/") + "/v1/traces"


def _is_truthy_env(name: str) -> bool:
    val = _env(name, "").lower()
    return val in {"1", "true", "yes", "y", "on"}


def _can_resolve_export_host(endpoint: str) -> bool:
    """
    Best-effort guard so local runs don't spam exporter errors when `jaeger`
    (docker-compose hostname) isn't resolvable.
    """
    try:
        parsed = urlparse(endpoint)
        host = (parsed.hostname or "").strip()
        if not host:
            return False
        socket.getaddrinfo(host, parsed.port or 80, proto=socket.IPPROTO_TCP)
        return True
    except Exception:
        return False


def _init_tracer_provider() -> None:
    # Safety: never create multiple tracer providers.
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return

    service_name = _env("OTEL_SERVICE_NAME", "CSR Copilot") or "CSR Copilot"
    protocol = _env("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").lower()
    if protocol and protocol != "http/protobuf":
        # Per target state: we only support OTLP HTTP/protobuf here.
        # Keep app running; exporter will still be configured for HTTP/protobuf.
        pass

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # Allow disabling exports explicitly for local dev.
    # Standard env: OTEL_TRACES_EXPORTER=none
    if _env("OTEL_TRACES_EXPORTER", "").lower() == "none" or _is_truthy_env("DISABLE_OTEL_EXPORT"):
        trace.set_tracer_provider(provider)
        return

    endpoint = _otlp_http_endpoint()
    if not _can_resolve_export_host(endpoint):
        # Keep tracing API working, but skip exporter to avoid background errors.
        # (Common when running app locally without docker-compose `jaeger` service.)
        trace.set_tracer_provider(provider)
        return

    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


_init_tracer_provider()
tracer = trace.get_tracer("csr_copilot.tracing")


def _ctx_from_parent(parent_span):
    if parent_span is None:
        return None
    try:
        return trace.set_span_in_context(parent_span)
    except Exception:
        return None


def _is_answer_fallback(answer_text: str) -> bool:
    """
    Answer-quality helper: detect common fallback/refusal phrases.
    Used only for resolution_score and relevance_score; must not affect security metrics.
    """
    if not answer_text or not isinstance(answer_text, str):
        return True
    t = (answer_text or "").strip().lower()
    if not t:
        return True
    fallback_phrases = [
        "i couldn't find", "i could not find", "i don't have", "i'm unable to",
        "i am unable to", "i can't", "i cannot", "no relevant", "not enough information",
        "i don't have access", "i'm not able", "i am not able", "no policy language",
        "couldn't find relevant", "no supporting",
    ]
    for p in fallback_phrases:
        if p in t:
            return True
    return False


_MONITORING_AVAILABLE = False

# Monitoring stack (whylogs/langkit/sentence-transformers/bigquery) is optional.
# The core service and tracing must still import and run without these dependencies.
try:
    from whylogs.experimental.core.udf_schema import udf_schema
    import whylogs as why
    from langkit import toxicity
    from langkit import sentiment
    from langkit import themes
    # from langkit import injections  # Disabled - AWS S3 data file unavailable
    from langkit import textstat
    from google.oauth2 import service_account
    from google.cloud import bigquery
    import closest
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    BASE_DIR = Path(__file__).resolve().parent
    with open(BASE_DIR / "files" / "ontopic_fd.json", "r", encoding="utf-8") as file:
        ontopic = json.load(file)
        ontopic = ontopic["jailbreak"]

    with open(BASE_DIR / "files" / "offtopic_fd.json", "r", encoding="utf-8") as file:
        offtopic = json.load(file)
        offtopic = offtopic["jailbreak"]

    ontopic_embed = [model.encode(i) for i in ontopic]
    offtopic_embed = [model.encode(i) for i in offtopic]

    BQ_CRED_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "bigquery.json")
    credentials = service_account.Credentials.from_service_account_file(
        BQ_CRED_PATH, scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    client = bigquery.Client(credentials=credentials, project="data-404309")
    dataset_id = "data123"
    table_id = "Score"
    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref)
    text_schema = udf_schema()

    _MONITORING_AVAILABLE = True
except Exception:
    # Keep module importable even when monitoring deps are missing.
    _MONITORING_AVAILABLE = False
    model = None
    ontopic_embed = []
    offtopic_embed = []
    client = None
    table = None
    text_schema = None




def func_Binsert(parent1, dicts,prompt, session_id=None, user_email=None, answer_text=None, feature_name=None, agent_name=None, flow_type=None):
    if not _MONITORING_AVAILABLE:
        return
    with tracer.start_as_current_span('func_Binsert', context=_ctx_from_parent(parent1)) as child2:
        # Feature Usage Insights: agent_name from span attribute agent.name if not passed.
        if agent_name is None and parent1 is not None:
            try:
                attrs = getattr(parent1, 'attributes', None) or getattr(parent1, '_attributes', None)
                if attrs and isinstance(attrs, dict):
                    agent_name = attrs.get('agent.name')
            except Exception:
                pass
        # Answer-quality metrics (computed in app.py / live_copilot.py); default 0 if missing.
        relevance_score = 0
        resolution_score = 0
        try:
            relevance_score = int(dicts.get('relevance_score', 0) or 0)
            resolution_score = int(dicts.get('resolution_score', 0) or 0)
        except (TypeError, ValueError):
            pass
        relevance_score = 1 if relevance_score else 0
        resolution_score = 1 if resolution_score else 0

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
                'Products':dicts['closest_topic'],
                'relevance_score': relevance_score,
                'resolution_score': resolution_score,

            },
            # Add more dictionaries for additional rows
        ]
        if session_id is not None:
            data_to_insert[0]['session_id'] = session_id
        if user_email is not None:
            data_to_insert[0]['user_email'] = user_email
        if answer_text is not None:
            data_to_insert[0]['answer_text'] = answer_text
        # Feature Usage Insights (nullable; append only when present).
        if feature_name is not None:
            data_to_insert[0]['feature_name'] = feature_name
        if agent_name is not None:
            data_to_insert[0]['agent_name'] = agent_name
        if flow_type is not None:
            data_to_insert[0]['flow_type'] = flow_type

        # Attach answer-quality metrics to current span (no new spans).
        try:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                span.set_attribute("score.relevance_score", relevance_score)
                span.set_attribute("score.resolution_score", resolution_score)
        except Exception:
            pass

        # Insert the data into the table
        errors = client.insert_rows(table, data_to_insert)

        if not errors:
            print(f"Data inserted successfully into {table_id}.")
        else:
            print('Errors occurred during data insertion:', errors)


def closest_t(child1, question):
    if not _MONITORING_AVAILABLE:
        return None
    with tracer.start_as_current_span('closest', context=_ctx_from_parent(child1)) as child1_1:
        topic = closest.classify_topic(closest.arr,question,closest.Embed)
        return topic


def score_calculator(child1, question):
    if not _MONITORING_AVAILABLE:
        return {}
    with tracer.start_as_current_span('score_calculator', context=_ctx_from_parent(child1)) as child1_2:
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
    if not _MONITORING_AVAILABLE:
        return 0.0
    with tracer.start_as_current_span('ontopic_fun', context=_ctx_from_parent(child1)) as child1_3:
        query_embedding = model.encode(query)
        val = -10
        for i in ontopic_embed:
            t_val = util.pytorch_cos_sim(query_embedding, i)[0][0]
            if(t_val>val):
                val = t_val
        return float(val)

def offtopic_fun(child1, query):
    if not _MONITORING_AVAILABLE:
        return 0.0
    with tracer.start_as_current_span('offtopic_fun', context=_ctx_from_parent(child1)) as child1_4:
        query_embedding = model.encode(query)
        val = -10
        for i in offtopic_embed:
            t_val = util.pytorch_cos_sim(query_embedding, i)[0][0]
            if(t_val>val):
                val = t_val
        return float(val)


def security_scores(parent1, question):
    if not _MONITORING_AVAILABLE:
        return {}
    with tracer.start_as_current_span('security_scores', context=_ctx_from_parent(parent1)) as child1:

        dicts = {}
        
        scores = score_calculator(child1, question)
        dicts.update(scores)
        dicts['closest_topic'] = closest_t(child1, question)
        

        # ontopic f3
        dicts["ontopic"] = ontopic_fun(child1, question)

        # offtopic f4
        dicts["offtopic"] = offtopic_fun(child1, question)

        return dicts


def q_monitor(parent1, question, session_id=None, user_email=None, answer_text=None, feature_name=None, agent_name=None, flow_type=None):
    if not _MONITORING_AVAILABLE:
        return
    # Guard against empty / bad input
    if not question or not isinstance(question, str):
        return
    dicts = security_scores(parent1,question)
    func_Binsert(parent1,dicts,question, session_id=session_id, user_email=user_email, answer_text=answer_text, feature_name=feature_name, agent_name=agent_name, flow_type=flow_type)


def llm_trace_to_jaeger(data, token_usage=None):
    """
    OpenTelemetry bridge for LangChain callback data.
    We do NOT create spans here (no new spans allowed in this migration).
    We only attach metadata + token totals to the current span.
    """
    span = trace.get_current_span()
    try:
        if span is None or not span.get_span_context().is_valid:
            return
    except Exception:
        return

    # Attach run metadata (best-effort, keep small to avoid oversized attributes)
    try:
        runs = list(data or [])
        span.set_attribute("langchain.runs.count", len(runs))
        names = []
        total_latency = 0.0
        for r in runs:
            if not isinstance(r, dict):
                continue
            nm = str(r.get("chain_name") or "")
            if nm:
                names.append(nm)
            try:
                total_latency += float(r.get("latency") or 0.0)
            except Exception:
                pass
        if names:
            span.set_attribute("langchain.runs.names_csv", ",".join(names)[:1024])
        span.set_attribute("langchain.runs.total_latency_s", float(total_latency))
    except Exception:
        pass

    # Attach token totals (from handler.infi token_usage list)
    try:
        toks = list(token_usage or [])
        prompt = 0
        completion = 0
        total = 0
        for t in toks:
            if not isinstance(t, dict):
                continue
            prompt += int(t.get("prompt_tokens") or 0)
            completion += int(t.get("completion_tokens") or 0)
            total += int(t.get("total_tokens") or 0)
        span.set_attribute("langchain.tokens.prompt", int(prompt))
        span.set_attribute("langchain.tokens.completion", int(completion))
        span.set_attribute("langchain.tokens.total", int(total))
    except Exception:
        pass