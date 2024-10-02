import threading
import mido 
import midi_tokenizer
import time 

class RingBuffer:
    """
    circular / ring buffer
    """
    def __init__(self, size):
        self.size = size
        self.array = [None] * size
        self.index = 0
        self.lock = threading.Lock()

    def addEvent(self, event):
        with self.lock:
            self.array[self.index] = event
            self.index = (self.index + 1) % self.size

    def getEvents(self):
        with self.lock:
            return list(self.array)

def midiMsgToTokenFormat(msg:mido.Message):
    """
    convert the mido MIDI message into 
    a format suitable for the midi tokenizer
    ["note", offset_in_ticks, duration_in_ticks, channel, note, velocity]
    """
    pass


def processMIDIMsg(msg:mido.Message, ring_buffer:RingBuffer):
    pass
    # ring_buffer.addEvent(msg)

def generate_sequence(model, tokenizer, context_len, buffer):
    # pad it if needed
    if len(buffer) < context_len:
        buffer.append(0) # some zeroes

    tokens = tokenizer.tokenize(buffer)
    model_out = [] # model.forward(tokens)
    detokened = tokenizer.detokenize(model_out)
    return detokened

def gen_runtime():
    """
    mapping out the generation runtime
    """
    tokenizer = midi_tokenizer.MIDITokenizer()
    context_len = 1024 # frame length passed to model
    model = []
    buffer = list()
    elapsed_time = 0
    max_time_frame = 5 # how long to wait between calls to generate 

    def receive_midi(msg:mido.Message):
        token_in = midiMsgToTokenFormat(msg)
        buffer.append(token_in)
        elapsed_time = elapsed_time + time.time()

        if (len(buffer) == context_len) or (elapsed_time >= max_time_frame):
            messages = generate_sequence(model,tokenizer,context_len,buffer)
            # schedule the messages to send ...
            


