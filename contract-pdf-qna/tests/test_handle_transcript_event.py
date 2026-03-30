from dataclasses import dataclass
import json
import live_copilot as lc
from core.schemas import CopilotSessionData
from threading import Thread



class socketioFactory:
    def __init__(self):
        self.result = []

    def emit(self, event, data, room):
        self.result += [ data ]
        print(data)


socketio = socketioFactory()

thread = Thread(target=lc.process_transcript_event_loop, args=(socketio, ))
thread.start()

    



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
        lc.handle_transcript_event(CopilotSessionData(text=text, speaker=speaker, phoneNumber=phoneNumber, sessionId=phoneNumber, isPartial=False,contactId=phoneNumber,contractType="", state="",plan="", beginOffsetMillis=0, endOffsetMillis=0))
        end = time.time() - s
    lc.handle_transcript_event(None)
    thread.join()
    with open(f"tests/converse_results/results_{phoneNumber}.json", "w") as file:
        json.dump(socketio.result, file)


    


files = glob("tests/converse/*.json")
for name in files:
    print(f'Processing file {name=}')
    phoneNumber = os.path.basename(name).split(".")[0]
    with open(name) as file:
        conversation = json.load(file)
        processFile(phoneNumber, conversation)

