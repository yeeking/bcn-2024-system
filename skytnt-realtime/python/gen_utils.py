import threading
import mido 
import midi_tokenizer
import midi_model
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
            
class MidoWrapper():
    def __init__(self, midiCallback):
        self.midiCallback = midiCallback
   
    def initMIDI(self):
        mido.set_backend('mido.backends.rtmidi')
        print("Available MIDI input devices:")
        midi_devices = mido.get_input_names()
        for i, device in enumerate(midi_devices):
            print(f"{i}: {device}")

        device_index = int(input("Enter the number of the MIDI device you want to connect to: "))
        # could do something more elegant where they get another chance... 
        assert (device_index >= 0) and (device_index < len(midi_devices)), "Invalid MIDI device chosen"
        selected_device = midi_devices[device_index]
        print(f"Connecting to: {selected_device} callback {self.midiCallback}")

        def midi_thread_function():
            with mido.open_input(selected_device, callback=self.midiCallback) as inport:
                input("MIDI open! Press Enter to stop listening...\n")

        # Start a new thread to run the thread_function
        thread = threading.Thread(target=midi_thread_function)
        thread.start()


        # with mido.open_input(selected_device, callback=self.midiCallback) as inport:
        #     print(f"Listening for messages from {selected_device}...")
        #     try:
        #         # Wait for MIDI messages to come in via callback
        #         input("Press Enter to stop listening...\n")
        #     except KeyboardInterrupt:
        #         print("MIDI listener stopped.")

class ImproviserAgent():
    def __init__(self):
        self.midoWrapper = MidoWrapper(self.receiveMIDI)
        self.noteBuffer = RingBuffer(8)
        self.start_time_s = time.time()
        self.bpm = 120 # not sure what to do with this one! 
        self.ticks_per_beat = 96 # 96 is a typical old-skool MIDI file tpb. 480 is more modern.
        
    def setModel(self, model:midi_model.MIDIModel):
        self.model = model 

    def initMIDI(self):
        self.midoWrapper.initMIDI()

    def receiveMIDI(self, msg:mido.Message):
        if msg.type == "note_on":
            print(f"ImproviserAgent receiveMIDI note on {msg}")
            # ['note', start_time, duration, channel, note, velocity]
            # ["note", offset_in_ticks, duration_in_ticks, channel, note, velocity ]
            # time since the start of this generation cycle 
            offset_secs = time.time() - self.start_time_s
            offset_in_ticks = (offset_secs / (60/self.bpm)) * self.ticks_per_beat
            event = ['note', offset_in_ticks, self.ticks_per_beat, 0, msg.note, msg.velocity]
            self.noteBuffer.addEvent(event)

    def generate(self):
        print(f"Generating improvisation... {self.noteBuffer.array}")

    def run(self):
        # Define the function that will run in the thread
        def thread_function():
            while True:
                time.sleep(1)  # Wait for 5 seconds
                self.generate()  # Call the generate function after waiting

        # Start a new thread to run the thread_function
        thread = threading.Thread(target=thread_function)
        thread.start()

improviser = ImproviserAgent() 
improviser.initMIDI()
improviser.run()
# improviserGlobal.initMIDI()

