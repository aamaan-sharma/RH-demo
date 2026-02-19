from dataclasses import dataclass
import json
import live_copilot as lc

    

@dataclass
class Payload:
    sessionId: str = "12345678"
    speaker: str = "agent"
    text: str = ""
    phoneNumber: str = "9876500001"
    contractType: str = "DTC"
    plan: str = "ShieldGold"
    state: str = "Georgia"
    isPartial: bool = False


conversation = None
with open("tests/converse.json") as file:
    conversation = json.load(file)

@dataclass
class Conversation:
    speaker: str
    text: str

results = []
import time


print("Started the Test")

for i, conv in enumerate(conversation):
    speaker = Conversation(**conv).speaker.lower()
    text = Conversation(**conv).text.lower()
    s = time.time()
    result = lc.handle_transcript_event(Payload(text=text, speaker=speaker).__dict__)
    end = time.time() - s
    if result == None:
        continue
    result["timing"] = end
    results += [ result ]
with open("results.json", "w") as file:
    json.dump(results, file)


    




    


