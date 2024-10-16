import threading
import mido 
import midi_tokenizer
import midi_model
import time 
import numpy as np
import torch 
from midi_model import MIDIModel, MIDITokenizer
import tqdm # for the progress bar
from contextlib import nullcontext # for the anti-progress bar :)   
import os
import copy 
import heapq
from collections import defaultdict
import torch.functional as F

class ModelHandler:
    def __init__():
        pass


    def load_model(path, model:MIDIModel):
        """
        load a python model from the sent checkpoint
        """
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return "success"


    @torch.inference_mode()
    def infer(model:MIDIModel, tokenizer:MIDITokenizer, prompt=None, max_len=512, temp=1.0, top_p=0.98, top_k=20,
                disable_patch_change=False, disable_control_change=False, disable_channels=None, amp=True, show_bar=False):
        """
        actually sends the input to the model and prepares the output 
        """
        # print(f"infer max len {max_len}")
        if disable_channels is not None:
            disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
        else:
            disable_channels = []
        max_token_seq = tokenizer.max_token_seq
        if prompt is None:
            input_tensor = torch.full((1, max_token_seq), tokenizer.pad_id, dtype=torch.long, device=model.device)
            input_tensor[0, 0] = tokenizer.bos_id  # bos
        else:
            prompt = prompt[:, :max_token_seq]
            if prompt.shape[-1] < max_token_seq:
                prompt = np.pad(prompt, ((0, 0), (0, max_token_seq - prompt.shape[-1])),
                                mode="constant", constant_values=tokenizer.pad_id)
            input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=model.device)
        input_tensor = input_tensor.unsqueeze(0)
        cur_len = input_tensor.shape[1]
        if model.device == 'cuda':
            autocast = torch.cuda.amp.autocast  # Use CUDA autocast for GPU
        else:
            autocast = torch.cpu.amp.autocast  # Use CPU autocast for CPU

        # print(f"Entering model forard call loop. in tensor shape: {input_tensor.shape} max len {max_len}")
        if show_bar == False:
            bar = tqdm.tqdm(desc="generating", total=max_len - cur_len, disable=True)
        else:
            bar = tqdm.tqdm(desc="generating", total=max_len - cur_len)
                
        with bar, autocast(enabled=amp):
        # with autocast(enabled=amp):
            # while cur_len < max_len: 
            for step in range(0, max_len): # ensure we get the full length output
                # print(f"Calling forward length is {cur_len} of {max_len} input shape is {input_tensor.shape} ")
                end = False
                hidden = model.forward(input_tensor)[0, -1].unsqueeze(0)
                next_token_seq = None
                event_name = ""
                for param_ind in range(max_token_seq):
                    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.int64, device=model.device)
                    if param_ind == 0:
                        mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                        if disable_patch_change:
                            mask_ids.remove(tokenizer.event_ids["patch_change"])
                        if disable_control_change:
                            mask_ids.remove(tokenizer.event_ids["control_change"])
                        mask[mask_ids] = 1
                    else:
                        param_name = tokenizer.events[event_name][param_ind - 1]
                        mask_ids = tokenizer.parameter_ids[param_name]
                        if param_name == "channel":
                            mask_ids = [i for i in mask_ids if i not in disable_channels]
                        mask[mask_ids] = 1
                    logits = model.forward_token(hidden, next_token_seq)[:, -1:]
                    scores = torch.softmax(logits / temp, dim=-1) * mask
                    sample = model.sample_top_p_k(scores, top_p, top_k)
                    if param_ind == 0:
                        next_token_seq = sample
                        eid = sample.item()
                        if eid == tokenizer.eos_id:
                            end = True
                            break
                        event_name = tokenizer.id_events[eid]
                    else:
                        next_token_seq = torch.cat([next_token_seq, sample], dim=1)
                        # if len(tokenizer.events[event_name]) == i:
                        #     break
                if next_token_seq.shape[1] < max_token_seq:
                    next_token_seq = F.pad(next_token_seq, (0, max_token_seq - next_token_seq.shape[1]),
                                        "constant", value=tokenizer.pad_id)
                next_token_seq = next_token_seq.unsqueeze(1)
                # print(next_token_seq)
                input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
                cur_len += 1
                if show_bar:
                    bar.update(1)
                yield next_token_seq.reshape(-1).cpu().numpy()
                if end:
                    break


    def generate_midi_seq(model:MIDIModel, tokenizer:MIDITokenizer, score_format_input, output_len, temp, top_p, top_k, allow_cc, amp, use_model=True, show_bar=False):
        """
        controller function for inference. Takes score format input, prepares it then sends it over to the model
        It is possible I can cut this one out and just go straight to the infer function 
        returns data in the format produced by MidiTokenizer.detokenize, which is 'score' format 
        """
        # print("generate_mid_seq")
        # prepare variables
        mid_seq = []
        max_len = int(output_len)
        disable_patch_change = False
        disable_channels = None

        tokens = tokenizer.tokenize(score_format_input)

        mid = np.asarray(tokens, dtype=np.int64)
        # print(f"Final midi format for model. Shape: {mid.shape}")
        # mid = mid[:int(max_input_len)] # if want to use a subset of the inputs 

        if use_model == False: # give up here...
            return 
        # print(f"Calling infer with max len {max_len}")
        generator = ModelHandler.infer(model, tokenizer, mid, max_len=max_len, 
                            temp=temp, top_p=top_p, top_k=top_k,
                            disable_patch_change=disable_patch_change, disable_control_change=not allow_cc,
                            disable_channels=disable_channels, amp=amp, show_bar=show_bar)
        for i, token_seq in enumerate(generator):
            # print(f"Gen step {i} of {len(token_seq)}")
            mid_seq.append(token_seq)
            ## this bit is for outputting as you generate ... might be useful! 
            # event = tokenizer.tokens2event(token_seq.tolist())

        score_format_data = tokenizer.detokenize(mid_seq)
        return score_format_data



class MIDINoteState:
    """
    Represents the state of a MIDI keyboard (which notes are held)
    as well as remembering note velocity and onset time 
    """
    def __init__(self):
        self.notes = []
        self.times = []
        self.vels = []
        self.reset()
    
    def reset(self):
        self.notes = [False for n in range(0, 128)]
        self.times = [0 for n in range(0, 128)]
        self.vels = [0 for n in range(0, 128)]

    def note_on(self, note, vel, tick_offset):
        """
        set the state of the sent note to on with the sent velocity
        """ 
        self.notes[note] = True
        self.vels[note] = vel
        self.times[note] = tick_offset
    
    def note_off(self, note, tick_offset):
        """
        a note off occured. Returns onset time of note on, elapsed time since note on, note on velocity 
        """
        self.notes[note] = False
        # self.vels[note] = 0
        assert tick_offset > self.times[note], "MidiState: note off happened before note on somehow"
        length = tick_offset - self.times[note]
        return self.times[note], length, self.vels[note] # onset time and length 
    

class RingBuffer:
    """
    circular / ring buffer
    """
    def __init__(self, size):
        self.size = size
        self.lock = threading.Lock()
        self.reset()

    def addEvent(self, event):
        with self.lock:
            self.array[self.index] = event
            self.index = (self.index + 1) % self.size

    def getEvents(self):
        with self.lock:
            return list(self.array)
    
    def isFull(self):
        """
        return true if the index is pointing to the last position
        """
        if self.index == len(self.array) - 1:
            return True
        else:
            return False
        
    def reset(self):
        with self.lock:
            self.index = 0
            self.array = [None] * self.size

class MidiDeviceHandler():
    def __init__(self, midiCallback):
        assert midiCallback is not None, "MidoWrapper: You need to pass a callback"
        self.midiCallback = midiCallback
        self.input_device = None
        self.output_device = None
        self.output_port = None
        self.output_lock = threading.Lock()  # Lock to ensure thread safety for output
        self.stop_event = threading.Event()  # 

    def getMIDIDevicesFromUser(self):
        mido.set_backend('mido.backends.rtmidi')
        
        # Get and choose MIDI input device
        print("Available MIDI input devices:")
        midi_input_devices = mido.get_input_names()
        for i, device in enumerate(midi_input_devices):
            print(f"{i}: {device}")

        input_device_index = int(input("Enter the number of the MIDI input device you want to connect to: "))
        assert (input_device_index >= 0) and (input_device_index < len(midi_input_devices)), "Invalid MIDI input device chosen"
        self.input_device = midi_input_devices[input_device_index]
        print(f"Connecting to MIDI input: {self.input_device}")

        # Get and choose MIDI output device
        print("\nAvailable MIDI output devices:")
        midi_output_devices = mido.get_output_names()
        for i, device in enumerate(midi_output_devices):
            print(f"{i}: {device}")

        output_device_index = int(input("Enter the number of the MIDI output device you want to connect to: "))
        assert (output_device_index >= 0) and (output_device_index < len(midi_output_devices)), "Invalid MIDI output device chosen"
        self.output_device = midi_output_devices[output_device_index]
        print(f"Connecting to MIDI output: {self.output_device}")

    def initMIDI(self):
        def midi_input_thread_function():
            with mido.open_input(self.input_device, callback=self.midiCallback) as inport:
                # input("MIDI input open! Press Enter to stop listening...\n")
                while not self.stop_event.is_set():  
                    time.sleep(1)  # Keeps the thread running

            print("MIDI input closing")

        def midi_output_thread_function():
            # Open MIDI output
            with mido.open_output(self.output_device) as outport:
                self.output_port = outport
                print("MIDI output open! You can now send messages to the output device.")

                # Keep the thread alive to allow sending messages
                while not self.stop_event.is_set():  
                    time.sleep(1)  # Keeps the thread running
            print("MIDI output closing")
            print("Now wait for the generation to complete and I will shut down")


        # Start MIDI input thread
        input_thread = threading.Thread(target=midi_input_thread_function)
        input_thread.start()

        # Start MIDI output thread (to handle output connection)
        output_thread = threading.Thread(target=midi_output_thread_function)
        output_thread.start()
        return input_thread,output_thread



    def sendMIDI(self, message: mido.Message):
        """Send a mido.Message to the output device."""
        with self.output_lock:
            if self.output_port:
                self.output_port.send(message)
                # print(f"MIDI HANDLER Sent MIDI message: {message}")
            else:
                print("MIDI HANDLER MIDI output port is not initialized.")
        
    def stop(self):
        print("Stopping MIDI I/O...")
        self.stop_event.set()  # Signal all threads to stop


class MIDIScheduler:
    """
    maintains a queue of MIDI messages with time stamps and 
    automatically sends them out via the midi handler at the correct time
    """
    def __init__(self, midiHandler:MidiDeviceHandler):
        self.message_queue = []  # Priority queue (min-heap) for (time, msg) tuples
        self.queue_lock = threading.Lock()  # To ensure thread safety for the queue
        self.send_lock = threading.Lock()   # To ensure thread safety for sending MIDI
        self.running = True
        self.thread = threading.Thread(target=self._clock_thread, daemon=True)
        self.thread.start()
        self.midiHandler = midiHandler

    def addMIDIMsg(self, msg, delay_ms):
        """Adds a MIDI message with a delay to the scheduler."""
        # print(f"MIDIQ adding message {msg} {delay_ms}")
        send_time = time.time() * 1000 + delay_ms  # Calculate absolute send time in ms
        # print(f"Q put a message {msg.type} {round(delay_ms/1000, 2)}ms in the future")
        with self.queue_lock:
            heapq.heappush(self.message_queue, (send_time, msg))

    def isEmpty(self):
        with self.queue_lock:
            return len(self.message_queue) == 0

    def _clock_thread(self):
        """Background thread to regularly check and send MIDI messages."""
        while self.running:
            current_time = time.time() * 1000  # Current time in milliseconds
            to_send = []
            with self.queue_lock:
                # Collect all messages whose time is less than or equal to the current time
                while self.message_queue and self.message_queue[0][0] <= current_time:
                    send_time, msg = heapq.heappop(self.message_queue)
                    to_send.append(msg)

            # Send messages outside the queue lock to avoid blocking it
            if to_send:
                self._send_midi_messages(to_send)

            # Sleep for a short time to ensure high clock accuracy
            time.sleep(0.001)

    def _send_midi_messages(self, messages):
        """Thread-safe method to send MIDI messages."""
        with self.send_lock:
            for msg in messages:
                self.sendMIDI(msg)

    def sendMIDI(self, msg):
        """Sends the MIDI message (you can replace this with actual sending logic)."""
        # Placeholder for actual MIDI sending code, e.g., using mido
        # print(f"MIDI Q: Sending MIDI message: {msg}")
        self.midiHandler.sendMIDI(msg)


    def stop(self):
        """Stops the scheduler and the background thread."""
        self.running = False
        self.thread.join()



from enum import IntFlag

class ImproviserStatus(IntFlag):
    OFF = 1        # 00001
    GETTING_MIDI = 2     # 00010
    LISTENING = 4   # 00100
    GENERATING = 8      # 01000
    SHUTTING_DOWN = 16    # 10000
    STARTING_UP = 32 

    @staticmethod
    def status_to_text(status:ImproviserStatus):
        return ""

class ImproviserAgent():
    def __init__(self, input_length:int, output_length:int, model:MIDIModel, tokenizer:MIDITokenizer, allow_gen_overlap=False, test_mode=False):
        self.midiHandler = MidiDeviceHandler(self.receiveMIDI)
        self.noteBuffer = RingBuffer(8192) # that should be plenty given it has a 4096 max context :)
        self.midiNoteState = MIDINoteState()
        self.midiQ = MIDIScheduler(self.midiHandler)

        self.start_time_s = time.time()
        self.bpm = 120 # not sure what to do with this one! 
        self.ticks_per_beat = 480 # 96 is a typical old-skool MIDI file tpb. 480 is more modern.
        self.setModel(model)
        self.tokenizer = tokenizer
        self.lock = threading.Lock() # to lock gneerate from multiple threads calling it
        self.stop_event = threading.Event()  # 
        self.test_mode = test_mode

        self.gen_thread = None
        self.allow_gen_overlap = allow_gen_overlap

        self.set_status(ImproviserStatus.OFF)
        
        self.input_length = 32
        self.output_length = 32 
        
        
    def setModel(self, model:midi_model.MIDIModel):
        self.model = model 

    def initMIDI(self):
        """
        initialises midi input and output and returns 
        the two threads controlling them so you can control
        threads outside this function 
        """
        self.set_status(ImproviserStatus.GETTING_MIDI)
        self.midiHandler.getMIDIDevicesFromUser()
        return self.midiHandler.initMIDI()
        
    def receiveMIDI(self, msg:mido.Message):
   
        offset_secs = time.time() - self.start_time_s
        offset_in_ticks = (offset_secs / (60/self.bpm)) * self.ticks_per_beat
        if msg.type == "note_on":
            if msg.velocity > 0:
                self.midiNoteState.note_on(msg.note, msg.velocity, offset_in_ticks)
        if msg.type == "note_off" or msg.velocity == 0:
            onset_tick, len_ticks, on_vel = self.midiNoteState.note_off(msg.note, offset_in_ticks)
            # ["note", offset_in_ticks, duration_in_ticks, channel, note, velocity ]
            # time since the start of this generation cycle 
            # if msg.velocity == 0:print("receiveMIDI zero velocity note mate!")
            event = ['note', int(onset_tick), int(len_ticks), 0, msg.note, on_vel]
            # print(f"Adding event to ring buffer {event}")
            self.noteBuffer.addEvent(event)
            # if self.noteBuffer.isFull():# generate when the buffer is full
            #     self.generate()

    def sendMIDI(self, msg):
        """
        assumes message format  ['note', start_time, duration, channel, note, velocity] 
        """
        # print("Sending a midi message", msg)
        if msg[0] != 'note': return # ignore non-note messages for now
        assert len(msg) == 6, "sendMIDI received bad message " + str(msg) # note message looks bad
        note = msg[4]
        vel = msg[5]
        start = msg[1]
        dur = msg[2]

        # work out the time delta in ms
        secs_per_tick = (60.0 / self.bpm) / self.ticks_per_beat
        beat_offset = start / self.ticks_per_beat # in beats
        note_on_offset_s = (60.0 / self.bpm) * beat_offset
        note_off_offset_s = note_on_offset_s + (secs_per_tick * dur)
        
        onMsg = mido.Message('note_on', note=note, velocity=vel)
        offMsg = mido.Message('note_off', note=note, velocity=vel)
        
        self.midiQ.addMIDIMsg(onMsg, note_on_offset_s * 1000)
        self.midiQ.addMIDIMsg(offMsg, note_off_offset_s * 1000)

    def analyse_output(self, output):
        """
        print some info about the de-tokenized output of the model
        """
        if len(output) < 2:
            print(f"Bad output {output}")
            return 
        tpb = output[0]
        ch_count = len(output) - 1 # first entry is not a channel it is tpb
        ch1_events = len(output[1])
        beat_offsets = [round(eve[1]/tpb, 2) for eve in output[1] if eve[0] == 'note']
        print(f"analyse_output: chans {ch_count} ch1 events {ch1_events} tpb {tpb} beats {beat_offsets}")
  
    def set_status(self, status:ImproviserStatus):
        self.status = status

    def get_status(self):
        # print("Impro get status called")
        return self.status 
    
    def call_the_model(self):
        """
        convert the current contents of the ring buffer into 
        a score format event list as in MIDI.py score 
        then tokenize it and send it to the model 
        """

        gen_events = None
        if self.lock.acquire(blocking=False):
            try:
            # Critical section that should be thread-safe
                print("Generating improvisation...")
                input_events = [copy.copy(e) for e in self.noteBuffer.array if e != None]
                if len(input_events) == 0:
                    print("No context yet...")
                    return  
                # now reset the ring buffer so we can capture more events whilst they continue to play
                self.noteBuffer.reset()
                # and reset the start time for the events we will now capture 
                self.start_time_s = time.time() # reset start time for next frame 
                print(f"Sending {len(input_events)} to the model")
                input_events = [self.ticks_per_beat] + [input_events]
                # output_events = self.generate(input_events)
                gen_events = ModelHandler.generate_midi_seq(self.model, self.tokenizer, 
                            input_events,
                            output_len=self.noteBuffer.size, # generate as much as we give you
                            temp=0.7, 
                            top_p=0.5, #0.1 to 1.0
                            top_k=1, #1 to 20 
                            allow_cc=False, # True or False
                            amp=True, use_model=(self.test_mode == False), show_bar=False) # True or False  
                self.analyse_output(gen_events)
                for track in gen_events[1:]:# first one is tpb
                    for score_msg in track:
                        self.sendMIDI(score_msg)
            finally:
                # Release the lock after completion
                self.lock.release()
                print("Generation complete.") 

                return gen_events
        else:
            # Lock is already acquired, reject the call
            print("Generate is already running, rejecting call.")
            return 
        

        
    def _run(self):
        """
        starts the improvisers runtime loop in a thread
        returns the thread for external thread management 
        """
        while not self.stop_event.is_set():
            self.set_status(ImproviserStatus.LISTENING)
            time.sleep(5)  # Wait for note collection
            self.set_status(ImproviserStatus.GENERATING)
 
            if (self.allow_gen_overlap) or (self.midiQ.isEmpty()):
                print(f"Q empty: {self.midiQ.isEmpty()} allow overlap {self.allow_gen_overlap}")
                gen_events = self.call_the_model() # try to generate every x seconds regardless of what has come in
            else:
                print("Q not empty or no overlap allowed")
            self.midiNoteState.reset() # clear off any outstanding notes    

    
    def start(self):
        self.set_status(ImproviserStatus.STARTING_UP)
        self.stop_event.set() 
        if self.gen_thread is not None: self.gen_thread.join()
        self.stop_event.clear()
        # Start a new thread to run the thread_function
        self.gen_thread = threading.Thread(target=self._run)
        self.gen_thread.start()
    

    def stop(self):
        self.set_status(ImproviserStatus.SHUTTING_DOWN)
        self.stop_event.set() 
        self.midiHandler.stop()
        self.midiQ.stop()

        self.gen_thread.join()
        self.set_status(ImproviserStatus.OFF)

        
    def setInputLength(self, length):
        self.input_length = length 
        print(f"Input Length set to: {length}")
    
    def setOutputLength(self, length):
        self.output_length = length 
        print(f"Output Length set to: {length}")
