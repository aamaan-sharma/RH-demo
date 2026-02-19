from dataclasses import dataclass
import json
import live_copilot as lc
from token_module import token_calculator, CallbackHandler

    

@dataclass
class Payload:
    sessionId: str = ""
    speaker: str = "agent"
    text: str = ""
    phoneNumber: str = ""
    contractType: str = ""
    plan: str = ""
    state: str = ""
    isPartial: bool = False


conversation = None
from glob import glob
import os

@dataclass
class Conversation:
    speaker: str
    text: str

import time


print("Started the Test")

def processFile(phoneNumber, conversation):
    results = []
    for i, conv in enumerate(conversation):
        speaker = Conversation(**conv).speaker.lower()
        text = Conversation(**conv).text.lower()
        s = time.time()
        result = lc.handle_transcript_event(Payload(text=text, speaker=speaker, phoneNumber=phoneNumber, sessionId=phoneNumber).__dict__)
        end = time.time() - s
        print(f"{result=}")

        if result == None:
            continue
        result["conversation"] = conv
        result["timing"] = end
        results += [ result ]
    with open(f"tests/converse_results/results_{phoneNumber}.json", "w") as file:
        json.dump(results, file)


    


files = glob("tests/converse/*.json")
for name in files:
    print(f'Processing file {name=}')
    phoneNumber = os.path.basename(name).split(".")[0]
    with open(name) as file:
        conversation = json.load(file)
        processFile(phoneNumber, conversation)
