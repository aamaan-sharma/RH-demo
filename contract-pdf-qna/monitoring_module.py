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
from jaeger_client import Config, span, span_context, constants
from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer( "sentence-transformers/all-MiniLM-L6-v2")

with open('files/ontopic_fd.json', 'r', encoding='utf-8') as file:
    ontopic = json.load(file)
    ontopic = ontopic['jailbreak']

with open('files/offtopic_fd.json', 'r', encoding='utf-8') as file:
    offtopic = json.load(file)
    offtopic = offtopic['jailbreak']


ontopic_embed = [model.encode(i) for i in ontopic]

offtopic_embed = [model.encode(i) for i in offtopic]


def init_tracer(service_name):
    config = Config(
        config={
            'sampler': {'type': 'const', 'param': 1},
            'logging': True,
        },
        service_name=service_name,
    )
    return config.initialize_tracer()


tracer = init_tracer('AHS Customer Rep Copilot')


credentials = service_account.Credentials.from_service_account_file(
    r'bigquery.json',
    scopes=['https://www.googleapis.com/auth/bigquery']
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
    with tracer.start_span('func_Binsert', child_of=parent1) as child2:
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
                'injection':dicts['prompt.injection'],
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
    with tracer.start_span('closest', child_of=child1) as child1_1:
        topic = closest.classify_topic(closest.arr,question,closest.Embed)
        return topic


def score_calculator(child1, question):
    with tracer.start_span('score_calculator', child_of=child1) as child1_2:
        dicts = {}
        results = why.log({"prompt": question}, schema = text_schema)
        score = results.view()
        for i in score.get_columns():    
            val = score.get_column(i).to_summary_dict()['distribution/mean']
            dicts[i] = val
        return dicts


def ontopic_fun(child1, query):
    with tracer.start_span('ontopic_fun', child_of=child1) as child1_3:
        query_embedding = model.encode(query)
        val = -10
        for i in ontopic_embed:
            t_val = util.pytorch_cos_sim(query_embedding, i)[0][0]
            if(t_val>val):
                val = t_val
        return float(val)

def offtopic_fun(child1, query):
    with tracer.start_span('offtopic_fun', child_of=child1) as child1_4:
        query_embedding = model.encode(query)
        val = -10
        for i in offtopic_embed:
            t_val = util.pytorch_cos_sim(query_embedding, i)[0][0]
            if(t_val>val):
                val = t_val
        return float(val)


def security_scores(parent1, question):
    with tracer.start_span('security_scores', child_of=parent1) as child1:

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
    dicts = security_scores(parent1,question)
    func_Binsert(parent1,dicts,question)


def llm_trace_to_jaeger(data, span_id, trace_id):
    # trace_id = random.getrandbits(64)
    for x in data[::-1]:
        if x['parent_run_id'] is not None:
            context = span_context.SpanContext(trace_id = trace_id,span_id = x['run_id'].int & 0xFFFFFFFFFFFFFFFF,parent_id=x['parent_run_id'].int & 0xFFFFFFFFFFFFFFFF,flags = 1)
        else:
            context = span_context.SpanContext(trace_id = trace_id,span_id = x['run_id'].int & 0xFFFFFFFFFFFFFFFF,parent_id=span_id,flags = 1)
        
        a = span.Span(context=context,tracer=tracer, operation_name=x["chain_name"], start_time=x["start_time"])
        a.finish(finish_time = x['end_time'])