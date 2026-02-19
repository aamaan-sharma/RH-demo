import json
from glob import glob




def processFile(filepath):
    print("Processing File: ", filepath)
    collect = []
    data = None
    with open(filepath) as file:
        data = json.load(file)
    for i, conv_process in enumerate(data):
        intent = conv_process["intent"]
        time = conv_process["timing"]
        if intent == "OTHER": continue
        collect += [ (i, intent, time) ]
    max_conv = max(collect, key=lambda x: x[2])
    total_time = sum([x[2] for x in collect]) 
    avg_time =  total_time / len(collect)
    print(f"{avg_time=}, {total_time=}")
    print(max_conv)





for filepath in glob("tests/converse_results/*.json"):
    processFile(filepath)
